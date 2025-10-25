import { env } from "./env";
import type { Session } from "./types";

const sessions = new Map<string, Session>();
// Track users currently creating sessions to prevent race conditions
const sessionCreationLocks = new Set<string>();
// Start cleanup timer
setInterval(() => {
	cleanupInactiveSessions();
}, env.CLEANUP_INTERVAL_MS);

export function getSession(userId: string) {
	return sessions.get(userId);
}

export function upsertSession(
	userId: string,
	partial: Partial<Session>,
): Session {
	const existing = sessions.get(userId);
	const now = Date.now();
	const merged: Session = {
		clientId: existing?.clientId || crypto.randomUUID().replace(/-/g, ""),
		clientWs: existing?.clientWs,
		comfyWs: existing?.comfyWs,
		connectionState: existing?.connectionState || "connecting",
		currentPromptId: existing?.currentPromptId,
		lastActiveAt: now,
		sid: existing?.sid,
		userId,
		...partial,
	};
	sessions.set(userId, merged);
	return merged;
}

export function clearSession(userId: string) {
	const session = sessions.get(userId);
	if (session) {
		// Close WebSocket connections
		try {
			session.comfyWs?.close();
		} catch {}
		try {
			session.clientWs?.close();
		} catch {}
		sessions.delete(userId);
		// Release any creation lock
		sessionCreationLocks.delete(userId);
		console.log(`Cleared session for user ${userId}`);
	}
}

export function updateLastActive(userId: string) {
	const session = sessions.get(userId);
	if (session) {
		session.lastActiveAt = Date.now();
	}
}

export function canCreateSession(userId: string): boolean {
	// Check if another connection is already being created for this user
	if (sessionCreationLocks.has(userId)) {
		return false;
	}

	const existing = sessions.get(userId);
	if (!existing) return true;

	// Check if existing session is stale
	const now = Date.now();
	if (now - existing.lastActiveAt > env.SESSION_TIMEOUT_MS) {
		clearSession(userId);
		return true;
	}

	// Check connection limit
	const activeSessions = Array.from(sessions.values()).filter(
		(s) => s.userId === userId && s.connectionState === "connected",
	).length;

	return activeSessions < env.MAX_CONNECTIONS_PER_USER;
}

/**
 * Atomically acquire a session creation lock for a user.
 * Returns true if lock was acquired, false if already locked.
 * Caller MUST call releaseSessionCreationLock when done.
 */
export function acquireSessionCreationLock(userId: string): boolean {
	if (sessionCreationLocks.has(userId)) {
		return false;
	}
	sessionCreationLocks.add(userId);
	return true;
}

/**
 * Release a session creation lock for a user.
 * Must be called after acquireSessionCreationLock.
 */
export function releaseSessionCreationLock(userId: string): void {
	sessionCreationLocks.delete(userId);
}

export function getAllSessions(): Session[] {
	return Array.from(sessions.values());
}

export function getSessionCount(): number {
	return sessions.size;
}

function cleanupInactiveSessions() {
	const now = Date.now();
	let cleanedCount = 0;

	for (const [userId, session] of sessions.entries()) {
		if (now - session.lastActiveAt > env.SESSION_TIMEOUT_MS) {
			clearSession(userId);
			cleanedCount++;
		}
	}

	if (cleanedCount > 0) {
		console.log(`Cleaned up ${cleanedCount} inactive sessions`);
	}
}
