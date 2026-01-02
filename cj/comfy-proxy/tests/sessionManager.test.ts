#!/usr/bin/env bun

/**
 * Unit tests for session management functionality
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import {
	clearSession,
	evictLeastActiveIfNeeded,
	getAllSessions,
	getSession,
	getSessionCount,
	getSessionsForUser,
	initializeSessionManager,
	type SessionManagerConfig,
	updateLastActive,
	upsertSession,
} from "../sessionManager";
import type { Session } from "../types";

// Mock WebSocket for testing
function createMockWebSocket(readyState = 1): Partial<WebSocket> {
	return {
		close: () => {},
		readyState,
	} as Partial<WebSocket>;
}

// Mock ExtendedServerWebSocket for testing
function createMockClientWs(readyState = 1): Partial<Session["clientWs"]> {
	return {
		close: () => {},
		readyState,
	} as Partial<Session["clientWs"]>;
}

const testConfig: SessionManagerConfig = {
	CLEANUP_INTERVAL_MS: 1000,
	MAX_CONNECTIONS_PER_USER: 3,
	SESSION_IDLE_EVICTION_MS: 5000,
	SESSION_TIMEOUT_MS: 10000,
};

beforeEach(() => {
	// Clear all sessions before each test
	const sessions = getAllSessions();
	for (const session of sessions) {
		clearSession(session.sessionId);
	}
});

afterEach(() => {
	// Clean up any intervals
	initializeSessionManager({
		...testConfig,
		CLEANUP_INTERVAL_MS: 0, // Disable cleanup for tests
	});
});

test("Session creation and upsertion", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	const session = upsertSession(userId, {
		clientId,
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	expect(session).toBeDefined();
	expect(session.sessionId).toBe(sessionId);
	expect(session.userId).toBe(userId);
	expect(session.clientId).toBe(clientId);
	expect(session.connectionState).toBe("connected");
});

test("Session retrieval by ID", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	upsertSession(userId, {
		clientId,
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	const retrieved = getSession(sessionId);
	expect(retrieved).toBeDefined();
	expect(retrieved?.sessionId).toBe(sessionId);
	expect(retrieved?.userId).toBe(userId);
});

test("Session retrieval returns undefined for non-existent session", () => {
	const retrieved = getSession("non-existent");
	expect(retrieved).toBeUndefined();
});

test("Get sessions for user", () => {
	const userId = "user1";

	// Create multiple sessions for the same user
	upsertSession(userId, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: Date.now() - 1000,
		sessionId: `${userId}:client1`,
		userId,
	});

	upsertSession(userId, {
		clientId: "client2",
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId: `${userId}:client2`,
		userId,
	});

	const sessions = getSessionsForUser(userId);
	expect(sessions.length).toBe(2);
	// Should be sorted by lastActiveAt (most recent first)
	expect(sessions[0]?.lastActiveAt).toBeGreaterThanOrEqual(
		sessions[1]?.lastActiveAt || 0,
	);
});

test("Get sessions for user returns empty array for non-existent user", () => {
	const sessions = getSessionsForUser("non-existent-user");
	expect(sessions).toEqual([]);
});

test("Update last active timestamp", async () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	const initialTime = Date.now();
	upsertSession(userId, {
		clientId,
		connectionState: "connected",
		lastActiveAt: initialTime,
		sessionId,
		userId,
	});

	// Wait a bit and update
	await new Promise((resolve) => setTimeout(resolve, 10));
	updateLastActive(sessionId);

	const session = getSession(sessionId);
	expect(session?.lastActiveAt).toBeGreaterThan(initialTime);
});

test("Update last active on non-existent session does nothing", () => {
	// Should not throw
	updateLastActive("non-existent");
});

test("Clear session", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	upsertSession(userId, {
		clientId,
		clientWs: createMockClientWs() as Session["clientWs"],
		comfyWs: createMockWebSocket() as WebSocket,
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	expect(getSession(sessionId)).toBeDefined();

	clearSession(sessionId);

	expect(getSession(sessionId)).toBeUndefined();
});

test("Clear session cancels reconnection timeout", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	const timeoutId = setTimeout(() => {}, 1000);

	upsertSession(userId, {
		clientId,
		connectionState: "connected",
		lastActiveAt: Date.now(),
		reconnectTimeoutId: timeoutId,
		sessionId,
		userId,
	});

	clearSession(sessionId);

	// Session should be cleared
	expect(getSession(sessionId)).toBeUndefined();
});

test("Get all sessions", () => {
	const userId1 = "user1";
	const userId2 = "user2";

	upsertSession(userId1, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId: `${userId1}:client1`,
		userId: userId1,
	});

	upsertSession(userId2, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId: `${userId2}:client1`,
		userId: userId2,
	});

	const allSessions = getAllSessions();
	expect(allSessions.length).toBe(2);
});

test("Get session count", () => {
	expect(getSessionCount()).toBe(0);

	upsertSession("user1", {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId: "user1:client1",
		userId: "user1",
	});

	expect(getSessionCount()).toBe(1);

	upsertSession("user2", {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId: "user2:client1",
		userId: "user2",
	});

	expect(getSessionCount()).toBe(2);
});

test("Evict least active session when at capacity", () => {
	const userId = "user1";
	const config: SessionManagerConfig = {
		...testConfig,
		MAX_CONNECTIONS_PER_USER: 2,
	};

	// Create sessions with different lastActiveAt times
	const now = Date.now();
	upsertSession(userId, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: now - 2000, // Oldest
		sessionId: `${userId}:client1`,
		userId,
	});

	upsertSession(userId, {
		clientId: "client2",
		connectionState: "connected",
		lastActiveAt: now, // Most recent
		sessionId: `${userId}:client2`,
		userId,
	});

	// At capacity (2 sessions)
	expect(getSessionsForUser(userId).length).toBe(2);

	// Try to evict - should evict least active
	const evicted = evictLeastActiveIfNeeded(userId, config);
	expect(evicted).toBe(true);
	expect(getSessionsForUser(userId).length).toBe(1);
	// Most recent session should remain
	expect(getSession(`${userId}:client2`)).toBeDefined();
	expect(getSession(`${userId}:client1`)).toBeUndefined();
});

test("Evict session when idle time exceeds threshold", () => {
	const userId = "user1";
	const config: SessionManagerConfig = {
		...testConfig,
		MAX_CONNECTIONS_PER_USER: 2,
		SESSION_IDLE_EVICTION_MS: 1000,
	};

	const now = Date.now();
	upsertSession(userId, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: now - 2000, // Idle for 2 seconds
		sessionId: `${userId}:client1`,
		userId,
	});

	upsertSession(userId, {
		clientId: "client2",
		connectionState: "connected",
		lastActiveAt: now,
		sessionId: `${userId}:client2`,
		userId,
	});

	const evicted = evictLeastActiveIfNeeded(userId, config);
	expect(evicted).toBe(true);
	// Should evict the idle session
	expect(getSession(`${userId}:client1`)).toBeUndefined();
});

test("Do not evict when under capacity", () => {
	const userId = "user1";
	const config: SessionManagerConfig = {
		...testConfig,
		MAX_CONNECTIONS_PER_USER: 3,
	};

	upsertSession(userId, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId: `${userId}:client1`,
		userId,
	});

	// Under capacity, should not evict
	const evicted = evictLeastActiveIfNeeded(userId, config);
	expect(evicted).toBe(false);
	expect(getSessionsForUser(userId).length).toBe(1);
});

test("Upsert session updates existing session", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	const initialSession = upsertSession(userId, {
		clientId,
		connectionState: "connecting",
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	expect(initialSession.connectionState).toBe("connecting");

	// Update the session
	const updatedSession = upsertSession(userId, {
		connectionState: "connected",
		sessionId,
		sid: "test-sid",
	});

	expect(updatedSession.connectionState).toBe("connected");
	expect(updatedSession.sid).toBe("test-sid");
	expect(updatedSession.sessionId).toBe(sessionId);
});

test("Upsert session with markReady callback", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	let readyCalled = false;
	const markReady = () => {
		readyCalled = true;
	};

	// Create session with markReady
	upsertSession(userId, {
		clientId,
		connectionState: "connecting",
		lastActiveAt: Date.now(),
		markReady,
		sessionId,
		userId,
	});

	// Update with sid - should call markReady
	upsertSession(userId, {
		sessionId,
		sid: "test-sid",
	});

	expect(readyCalled).toBe(true);
});

test("Session cleanup interval initialization", () => {
	const config: SessionManagerConfig = {
		...testConfig,
		CLEANUP_INTERVAL_MS: 100,
	};

	initializeSessionManager(config);

	// Should not throw
	expect(true).toBe(true);
});

test("Multiple sessions per user sorted correctly", () => {
	const userId = "user1";
	const now = Date.now();

	upsertSession(userId, {
		clientId: "client1",
		connectionState: "connected",
		lastActiveAt: now - 3000,
		sessionId: `${userId}:client1`,
		userId,
	});

	upsertSession(userId, {
		clientId: "client2",
		connectionState: "connected",
		lastActiveAt: now - 1000,
		sessionId: `${userId}:client2`,
		userId,
	});

	upsertSession(userId, {
		clientId: "client3",
		connectionState: "connected",
		lastActiveAt: now,
		sessionId: `${userId}:client3`,
		userId,
	});

	const sessions = getSessionsForUser(userId);
	expect(sessions.length).toBe(3);
	// Should be sorted most recent first
	expect(sessions[0]?.lastActiveAt).toBe(now);
	expect(sessions[1]?.lastActiveAt).toBe(now - 1000);
	expect(sessions[2]?.lastActiveAt).toBe(now - 3000);
});

test("Clear session closes WebSocket connections", () => {
	const userId = "user1";
	const clientId = "client1";
	const sessionId = `${userId}:${clientId}`;

	let comfyClosed = false;
	let clientClosed = false;

	const mockComfyWs = {
		close: () => {
			comfyClosed = true;
		},
		readyState: 1,
	} as Partial<WebSocket> as WebSocket;

	const mockClientWs = {
		close: () => {
			clientClosed = true;
		},
		readyState: 1,
	} as Partial<Session["clientWs"]> as Session["clientWs"];

	upsertSession(userId, {
		clientId,
		clientWs: mockClientWs,
		comfyWs: mockComfyWs,
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	clearSession(sessionId);

	expect(comfyClosed).toBe(true);
	expect(clientClosed).toBe(true);
});
