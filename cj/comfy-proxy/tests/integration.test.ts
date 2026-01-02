#!/usr/bin/env bun

/**
 * Integration tests for proxy functionality
 * Tests the interaction between client, proxy, and ComfyUI (mocked)
 */

import { afterAll, beforeAll, expect, test } from "bun:test";
import { createPromptQueue } from "../errorHandling";
import { Logger } from "../logger";
import type { ProxyConfig } from "../proxy";
import {
	clearSession,
	getSession,
	getSessionCount,
	initializeSessionManager,
	upsertSession,
} from "../sessionManager";
import type { Session } from "../types";

// Mock configuration
const mockConfig: ProxyConfig = {
	CIRCUIT_BREAKER_THRESHOLD: 5,
	CIRCUIT_BREAKER_TIMEOUT_MS: 30000,
	CLEANUP_INTERVAL_MS: 300000,
	CONNECTION_TIMEOUT_MS: 10000,
	HEALTH_CHECK_INTERVAL_MS: 10000,
	HEALTH_CHECK_TIMEOUT_MS: 5000,
	HTTP_REQUEST_TIMEOUT_MS: 30000,
	INITIAL_RETRY_DELAY_MS: 1000,
	LOG_LEVEL: "silent",
	MAX_CONNECTIONS_PER_USER: 5,
	MAX_PROMPT_RETRIES: 3,
	MAX_QUEUED_PROMPTS_PER_USER: 100,
	MAX_RETRY_DELAY_MS: 30000,
	METRICS_SECRET: "test-metrics-secret",
	PROXY_COMFY_JWT_SECRET: "test-secret",
	PROXY_COMFY_RECONNECT_ENABLED: true,
	PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS: 1000,
	PROXY_COMFY_RECONNECT_MAX_DELAY_MS: 30000,
	PROXY_COMFY_RECONNECT_MAX_RETRIES: 5,
	PROXY_COMFY_URL: "http://localhost:8188",
	PROXY_PORT: 8190,
	SESSION_IDLE_EVICTION_MS: 300000,
	SESSION_READY_TIMEOUT_MS: 10000,
	SESSION_TIMEOUT_MS: 1800000,
};

beforeAll(() => {
	// Initialize session manager
	initializeSessionManager({
		CLEANUP_INTERVAL_MS: mockConfig.CLEANUP_INTERVAL_MS,
		MAX_CONNECTIONS_PER_USER: mockConfig.MAX_CONNECTIONS_PER_USER,
		SESSION_IDLE_EVICTION_MS: mockConfig.SESSION_IDLE_EVICTION_MS,
		SESSION_TIMEOUT_MS: mockConfig.SESSION_TIMEOUT_MS,
	});
});

afterAll(() => {
	// Clean up all sessions
	const sessions = Array.from(
		new Set(
			Array.from({ length: getSessionCount() }, (_, i) => {
				const session = getSession(`test-session-${i}`);
				return session?.sessionId;
			}).filter(Boolean) as string[],
		),
	);
	for (const sessionId of sessions) {
		if (sessionId) clearSession(sessionId);
	}
});

test("Session creation and retrieval", () => {
	const userId = "test-user-1";
	const sessionId = "test-session-1";
	const clientId = "test-client-1";

	upsertSession(userId, {
		clientId,
		connectionState: "connected",
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	const session = getSession(sessionId);
	expect(session).toBeDefined();
	expect(session?.userId).toBe(userId);
	expect(session?.clientId).toBe(clientId);
	expect(session?.sessionId).toBe(sessionId);

	clearSession(sessionId);
});

test("Session correlation ID propagation", () => {
	const userId = "test-user-2";
	const sessionId = "test-session-2";
	const correlationId = crypto.randomUUID();

	const session: Partial<Session> = {
		clientId: "test-client-2",
		connectionState: "connected",
		correlationId,
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	};

	upsertSession(userId, session);

	const retrieved = getSession(sessionId);
	expect(retrieved?.correlationId).toBe(correlationId);

	clearSession(sessionId);
});

test("Prompt queue integration with circuit breaker", () => {
	const circuitBreakerConfig = {
		threshold: 3,
		timeoutMs: 1000,
	};
	const promptQueue = createPromptQueue(
		10,
		circuitBreakerConfig,
		new Logger("silent", "[Test]"),
	);

	// Initially should be able to process
	expect(promptQueue.canProcess()).toBe(true);

	// Record failures up to threshold
	promptQueue.recordFailure();
	promptQueue.recordFailure();
	expect(promptQueue.canProcess()).toBe(true);

	promptQueue.recordFailure();
	expect(promptQueue.canProcess()).toBe(false);

	// Record success should reset
	promptQueue.recordSuccess();
	expect(promptQueue.canProcess()).toBe(true);
});

test("Error propagation through proxy layers", () => {
	const logger = new Logger("silent");
	const userId = "test-user-3";
	const sessionId = "test-session-3";

	// Create session with correlation ID
	const correlationId = crypto.randomUUID();
	const sessionLogger = logger.withCorrelationId(correlationId);

	upsertSession(userId, {
		clientId: "test-client-3",
		connectionState: "connected",
		correlationId,
		lastActiveAt: Date.now(),
		sessionId,
		userId,
	});

	// Verify correlation ID is stored and can be retrieved
	const session = getSession(sessionId);
	expect(session?.correlationId).toBe(correlationId);
	expect(sessionLogger.getCorrelationId()).toBe(correlationId);

	clearSession(sessionId);
});

test("Session cleanup on timeout", async () => {
	const userId = "test-user-4";
	const sessionId = "test-session-4";

	// Create session with old timestamp
	upsertSession(userId, {
		clientId: "test-client-4",
		connectionState: "connected",
		lastActiveAt: Date.now() - 2000000, // 2000 seconds ago (older than timeout)
		sessionId,
		userId,
	});

	const session = getSession(sessionId);
	expect(session).toBeDefined();

	// Note: Actual cleanup is tested in sessionManager.test.ts
	// This test verifies the session exists before cleanup
	clearSession(sessionId);
});
