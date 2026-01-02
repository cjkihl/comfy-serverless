import { Logger } from "./logger";
import type { Session } from "./types";

const sessions = new Map<string, Session>();
let cleanupInterval: Timer | null = null;
const logger = new Logger("info", "[SessionManager]");

export interface SessionManagerConfig {
	CLEANUP_INTERVAL_MS: number;
	MAX_CONNECTIONS_PER_USER: number;
	SESSION_IDLE_EVICTION_MS: number;
	SESSION_TIMEOUT_MS: number;
}

export function initializeSessionManager(config: SessionManagerConfig): void {
	// Clear any existing interval
	if (cleanupInterval) {
		clearInterval(cleanupInterval);
	}

	// Start cleanup timer
	cleanupInterval = setInterval(() => {
		cleanupInactiveSessions(config);
	}, config.CLEANUP_INTERVAL_MS);
}

export function getSession(sessionId: string) {
	return sessions.get(sessionId);
}

export function upsertSession(
	userId: string,
	partial: Partial<Session>,
): Session {
	const sessionId =
		partial.sessionId ||
		`${userId}:${partial.clientId || crypto.randomUUID().replace(/-/g, "")}`;
	const existing = sessions.get(sessionId);
	const now = Date.now();

	// If sid is being set and we have a markReady callback, call it
	if (partial.sid && !existing?.sid && existing?.markReady) {
		logger.debug(
			`🎯 Calling markReady for session ${sessionId} (got sid: ${partial.sid})`,
		);
		existing.markReady();
	} else if (partial.sid && existing) {
		logger.warn(
			`⚠️ Session ${sessionId} got sid ${partial.sid} but no markReady callback (existing=${!!existing})`,
		);
	}

	const merged: Session = {
		clientId: existing?.clientId || crypto.randomUUID().replace(/-/g, ""),
		clientWs: existing?.clientWs,
		comfyWs: existing?.comfyWs,
		connectionState: existing?.connectionState || "connecting",
		currentPromptId: existing?.currentPromptId,
		lastActiveAt: now,
		markReady: partial.markReady || existing?.markReady,
		readyPromise: partial.readyPromise || existing?.readyPromise,
		sessionId,
		sid: partial.sid !== undefined ? partial.sid : existing?.sid, // Use partial.sid if provided
		userId,
		...partial,
	};
	sessions.set(sessionId, merged);
	return merged;
}

export function clearSession(sessionId: string) {
	const session = sessions.get(sessionId);
	if (session) {
		// Cancel any pending reconnection attempts
		if (session.reconnectTimeoutId) {
			clearTimeout(session.reconnectTimeoutId);
		}
		// Close WebSocket connections
		try {
			session.comfyWs?.close();
		} catch {}
		try {
			session.clientWs?.close();
		} catch {}
		sessions.delete(sessionId);
		logger.debug(`Cleared session ${sessionId} for user ${session.userId}`);
	}
}

export function updateLastActive(sessionId: string) {
	const session = sessions.get(sessionId);
	if (session) {
		session.lastActiveAt = Date.now();
	}
}

/**
 * Get all sessions for a specific user, sorted by lastActiveAt (most recent first)
 */
export function getSessionsForUser(userId: string): Session[] {
	return Array.from(sessions.values())
		.filter((s) => s.userId === userId)
		.sort((a, b) => b.lastActiveAt - a.lastActiveAt);
}

/**
 * Get the least recently active session for a user
 */
function getLeastActiveSession(userId: string): Session | null {
	const userSessions = getSessionsForUser(userId);
	return userSessions[userSessions.length - 1] || null;
}

/**
 * Evict the least recently active session if user is at capacity
 * Returns true if eviction happened, false otherwise
 */
export function evictLeastActiveIfNeeded(
	userId: string,
	config: SessionManagerConfig,
): boolean {
	const userSessions = getSessionsForUser(userId);

	// If under capacity, no eviction needed
	if (userSessions.length < config.MAX_CONNECTIONS_PER_USER) {
		return false;
	}

	// Find least active session
	const leastActive = getLeastActiveSession(userId);
	if (!leastActive) {
		return false;
	}

	// Check if it should be evicted based on idle time
	const now = Date.now();
	const idleTime = now - leastActive.lastActiveAt;

	// Evict if idle for more than 5 minutes OR if at capacity
	if (
		idleTime > config.SESSION_IDLE_EVICTION_MS ||
		userSessions.length >= config.MAX_CONNECTIONS_PER_USER
	) {
		logger.info(
			`Evicting least active session ${leastActive.sessionId} for user ${userId} (idle: ${Math.round(idleTime / 1000)}s)`,
		);
		clearSession(leastActive.sessionId);
		return true;
	}

	return false;
}

export function getAllSessions(): Session[] {
	return Array.from(sessions.values());
}

export function getSessionCount(): number {
	return sessions.size;
}

function cleanupInactiveSessions(config: SessionManagerConfig) {
	const now = Date.now();
	let cleanedCount = 0;

	for (const [sessionId, session] of sessions.entries()) {
		// Safety check: ensure session is actually inactive before cleanup
		// Check both timeout and that WebSocket connections are closed/invalid
		const isTimedOut = now - session.lastActiveAt > config.SESSION_TIMEOUT_MS;

		// Check if WebSocket connections are still active
		const comfyWsActive = session.comfyWs?.readyState === 1; // OPEN
		const clientWsActive = session.clientWs?.readyState === 1; // OPEN

		// Only cleanup if timed out AND no active connections
		if (isTimedOut && !comfyWsActive && !clientWsActive) {
			logger.debug(
				`Cleaning up inactive session ${sessionId} (idle for ${Math.round((now - session.lastActiveAt) / 1000)}s)`,
			);
			clearSession(sessionId);
			cleanedCount++;
		} else if (isTimedOut && (comfyWsActive || clientWsActive)) {
			logger.warn(
				`Session ${sessionId} timed out but has active connections, skipping cleanup (comfyWs: ${comfyWsActive}, clientWs: ${clientWsActive})`,
			);
		}
	}

	if (cleanedCount > 0) {
		logger.debug(`Cleaned up ${cleanedCount} inactive sessions`);
	}
}
