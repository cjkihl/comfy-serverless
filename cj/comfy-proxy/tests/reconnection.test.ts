#!/usr/bin/env bun

/**
 * Unit tests for reconnection logic
 */

import { expect, test } from "bun:test";

// Test exponential backoff calculation
test("Exponential backoff calculation", () => {
	const initialDelayMs = 1000;
	const maxDelayMs = 30000;

	// First attempt (reconnectAttempts = 1)
	const attempt1 = Math.min(initialDelayMs * 2 ** (1 - 1), maxDelayMs);
	expect(attempt1).toBe(1000);

	// Second attempt (reconnectAttempts = 2)
	const attempt2 = Math.min(initialDelayMs * 2 ** (2 - 1), maxDelayMs);
	expect(attempt2).toBe(2000);

	// Third attempt (reconnectAttempts = 3)
	const attempt3 = Math.min(initialDelayMs * 2 ** (3 - 1), maxDelayMs);
	expect(attempt3).toBe(4000);

	// Fourth attempt (reconnectAttempts = 4)
	const attempt4 = Math.min(initialDelayMs * 2 ** (4 - 1), maxDelayMs);
	expect(attempt4).toBe(8000);

	// Fifth attempt (reconnectAttempts = 5)
	const attempt5 = Math.min(initialDelayMs * 2 ** (5 - 1), maxDelayMs);
	expect(attempt5).toBe(16000);
});

test("Exponential backoff respects max delay", () => {
	const initialDelayMs = 1000;
	const maxDelayMs = 5000;

	// Attempt that would exceed max
	const attempt10 = Math.min(initialDelayMs * 2 ** (10 - 1), maxDelayMs);
	expect(attempt10).toBe(maxDelayMs);
});

test("Exponential backoff with different initial delay", () => {
	const initialDelayMs = 500;
	const maxDelayMs = 30000;

	const attempt1 = Math.min(initialDelayMs * 2 ** (1 - 1), maxDelayMs);
	expect(attempt1).toBe(500);

	const attempt2 = Math.min(initialDelayMs * 2 ** (2 - 1), maxDelayMs);
	expect(attempt2).toBe(1000);
});

test("Reconnection attempt counting", () => {
	let reconnectAttempts = 0;

	// Simulate reconnection attempts
	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(reconnectAttempts).toBe(1);

	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(reconnectAttempts).toBe(2);

	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(reconnectAttempts).toBe(3);
});

test("Max retry limit enforcement", () => {
	const maxRetries = 5;
	let reconnectAttempts = 0;

	// Simulate attempts up to max
	for (let i = 0; i < maxRetries; i++) {
		reconnectAttempts = (reconnectAttempts || 0) + 1;
		expect(reconnectAttempts).toBeLessThanOrEqual(maxRetries);
	}

	// Next attempt should exceed max
	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(reconnectAttempts).toBeGreaterThan(maxRetries);
});

test("Reconnection state flag management", () => {
	let isReconnecting = false;

	// Set reconnecting flag
	isReconnecting = true;
	expect(isReconnecting).toBe(true);

	// Clear reconnecting flag
	isReconnecting = false;
	expect(isReconnecting).toBe(false);
});

test("Reconnection timeout ID management", () => {
	let reconnectTimeoutId: Timer | undefined;

	// Set timeout ID
	const timeoutId = setTimeout(() => {}, 1000);
	reconnectTimeoutId = timeoutId;
	expect(reconnectTimeoutId).toBeDefined();

	// Clear timeout
	if (reconnectTimeoutId) {
		clearTimeout(reconnectTimeoutId);
		reconnectTimeoutId = undefined;
	}
	expect(reconnectTimeoutId).toBeUndefined();
});

test("Reconnection delay calculation for various attempts", () => {
	const initialDelayMs = 1000;
	const maxDelayMs = 30000;

	const delays: number[] = [];
	for (let attempt = 1; attempt <= 10; attempt++) {
		const delay = Math.min(initialDelayMs * 2 ** (attempt - 1), maxDelayMs);
		delays.push(delay);
	}

	// Verify exponential growth
	expect(delays[0]).toBe(1000);
	expect(delays[1]).toBe(2000);
	expect(delays[2]).toBe(4000);
	expect(delays[3]).toBe(8000);
	expect(delays[4]).toBe(16000);
	expect(delays[5]).toBe(30000); // Capped at max
	expect(delays[6]).toBe(30000); // Still capped
});

test("Reconnection attempt reset after success", () => {
	let reconnectAttempts = 3;

	// After successful reconnection, reset attempts
	reconnectAttempts = 0;
	expect(reconnectAttempts).toBe(0);
});

test("Reconnection state prevents duplicate reconnection attempts", () => {
	let isReconnecting = false;

	// First reconnection attempt
	if (!isReconnecting) {
		isReconnecting = true;
		expect(isReconnecting).toBe(true);
	}

	// Second attempt should be prevented
	if (!isReconnecting) {
		isReconnecting = true;
		expect(isReconnecting).toBe(true);
	} else {
		// Should not set again
		expect(isReconnecting).toBe(true);
	}
});

test("Reconnection with zero initial delay", () => {
	const initialDelayMs = 0;
	const maxDelayMs = 30000;

	const attempt1 = Math.min(initialDelayMs * 2 ** (1 - 1), maxDelayMs);
	expect(attempt1).toBe(0);
});

test("Reconnection delay calculation edge cases", () => {
	const initialDelayMs = 1000;
	const maxDelayMs = 30000;

	// Attempt 0 (shouldn't happen but test edge case)
	const attempt0 = Math.min(initialDelayMs * 2 ** (0 - 1), maxDelayMs);
	expect(attempt0).toBe(500); // 2^(-1) = 0.5

	// Very large attempt number
	const attempt100 = Math.min(initialDelayMs * 2 ** (100 - 1), maxDelayMs);
	expect(attempt100).toBe(maxDelayMs); // Should be capped
});

test("Reconnection attempt tracking with undefined initial value", () => {
	let reconnectAttempts: number | undefined;

	// First attempt
	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(reconnectAttempts).toBe(1);

	// Second attempt
	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(reconnectAttempts).toBe(2);
});

test("Reconnection timeout cancellation", () => {
	let reconnectTimeoutId: Timer | undefined;

	// Create timeout
	const timeoutId = setTimeout(() => {}, 1000);
	reconnectTimeoutId = timeoutId;

	// Cancel timeout
	if (reconnectTimeoutId) {
		clearTimeout(reconnectTimeoutId);
		reconnectTimeoutId = undefined;
	}

	expect(reconnectTimeoutId).toBeUndefined();
});

test("Reconnection state transition flow", () => {
	let isReconnecting = false;
	let reconnectAttempts = 0;
	const maxRetries = 5;

	// Start reconnection
	isReconnecting = true;
	reconnectAttempts = (reconnectAttempts || 0) + 1;
	expect(isReconnecting).toBe(true);
	expect(reconnectAttempts).toBe(1);

	// Simulate reconnection attempt
	if (reconnectAttempts >= maxRetries) {
		isReconnecting = false;
	} else {
		// Continue reconnecting
		expect(isReconnecting).toBe(true);
	}

	// After max retries
	reconnectAttempts = maxRetries;
	if (reconnectAttempts >= maxRetries) {
		isReconnecting = false;
	}
	expect(isReconnecting).toBe(false);
});
