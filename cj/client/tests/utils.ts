#!/usr/bin/env bun

/**
 * Test utilities for @cj/comfy-client package
 * Adapted from proxy/tests/utils.ts
 */

import type { ComfyPrompt } from "../src/types";
import { testPrompt } from "./test-prompt";

/**
 * Generate a unique test user ID with timestamp and random component
 * This ensures test isolation and prevents conflicts between test runs
 */
export function generateUniqueUserId(prefix = "test-user"): string {
	const timestamp = Date.now();
	const random = Math.random().toString(36).substr(2, 9);
	return `${prefix}-${timestamp}-${random}`;
}

/**
 * Generate a test JWT token with the given userId in the 'sub' claim
 * This creates an unsigned JWT that will be accepted when NO_VERIFY=true
 */
export function generateTestJWT(userId: string): string {
	console.log(`[JWT] Generating test JWT for user: ${userId}`);

	const header = {
		alg: "none",
		typ: "JWT",
	};

	const payload = {
		aud: "test-audience",
		exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour
		iat: Math.floor(Date.now() / 1000),
		iss: "test-issuer",
		sub: userId,
	};

	// Create unsigned JWT (no signature)
	const encodedHeader = Buffer.from(JSON.stringify(header)).toString(
		"base64url",
	);
	const encodedPayload = Buffer.from(JSON.stringify(payload)).toString(
		"base64url",
	);

	const token = `${encodedHeader}.${encodedPayload}.`;
	console.log(
		`[JWT] Generated token for user ${userId}: ${token.substring(0, 50)}...`,
	);
	return token;
}

/**
 * Measure the execution time of an async function
 */
export async function measureTime<T>(
	fn: () => Promise<T>,
): Promise<{ result: T; duration: number }> {
	const start = performance.now();
	const result = await fn();
	const duration = performance.now() - start;
	return { duration, result };
}

/**
 * Get the known working test prompt for ComfyUI
 * This prompt is tested and known to work successfully
 */
export function createTestPrompt(): ComfyPrompt {
	return testPrompt;
}

/**
 * Validate that all expected event types were received
 */
export function validateEventSequence(events: unknown[]): {
	valid: boolean;
	missingEvents: string[];
	extraEvents: string[];
} {
	const expectedEvents = [
		"prompt_accepted",
		"status",
		"executing",
		"progress_state",
		"executed",
		"execution_success",
	];

	const receivedEventTypes = events.map((e) => (e as { type: string }).type);
	const missingEvents = expectedEvents.filter(
		(eventType) => !receivedEventTypes.includes(eventType),
	);
	const extraEvents = receivedEventTypes.filter(
		(eventType) => !expectedEvents.includes(eventType),
	);

	return {
		extraEvents,
		missingEvents,
		valid: missingEvents.length === 0,
	};
}

/**
 * Format duration in milliseconds to human readable format
 */
export function formatDuration(ms: number): string {
	if (ms < 1000) {
		return `${ms.toFixed(0)}ms`;
	}
	if (ms < 60000) {
		return `${(ms / 1000).toFixed(1)}s`;
	}
	const minutes = Math.floor(ms / 60000);
	const seconds = ((ms % 60000) / 1000).toFixed(1);
	return `${minutes}m ${seconds}s`;
}

/**
 * Check if binary data looks like a valid image
 */
export function isValidImageData(data: ArrayBuffer): boolean {
	if (data.byteLength < 4) return false;

	const view = new Uint8Array(data);

	// Check for common image format signatures
	// PNG: 89 50 4E 47
	if (
		view[0] === 0x89 &&
		view[1] === 0x50 &&
		view[2] === 0x4e &&
		view[3] === 0x47
	) {
		return true;
	}

	// JPEG: FF D8 FF
	if (view[0] === 0xff && view[1] === 0xd8 && view[2] === 0xff) {
		return true;
	}

	// WebP: 52 49 46 46 ... 57 45 42 50
	if (
		view[0] === 0x52 &&
		view[1] === 0x49 &&
		view[2] === 0x46 &&
		view[3] === 0x46
	) {
		// Check for WEBP signature further in the file
		for (let i = 8; i < Math.min(view.length - 4, 20); i++) {
			if (
				view[i] === 0x57 &&
				view[i + 1] === 0x45 &&
				view[i + 2] === 0x42 &&
				view[i + 3] === 0x50
			) {
				return true;
			}
		}
	}

	return false;
}

/**
 * Test result interface
 */
export interface TestResult {
	name: string;
	success: boolean;
	duration: number;
	error?: string;
	details?: unknown;
}

/**
 * Run a test and return formatted result
 */
export async function runTest(
	name: string,
	testFn: () => Promise<void>,
): Promise<TestResult> {
	const start = performance.now();
	try {
		await testFn();
		const duration = performance.now() - start;
		return {
			duration,
			name,
			success: true,
		};
	} catch (error) {
		const duration = performance.now() - start;
		return {
			duration,
			error: error instanceof Error ? error.message : String(error),
			name,
			success: false,
		};
	}
}

/**
 * Print test results in a formatted way
 */
export function printTestResults(results: TestResult[]): void {
	console.log("\n📊 Test Results:");
	console.log("================");

	let passed = 0;
	let failed = 0;

	for (const result of results) {
		const status = result.success ? "✅" : "❌";
		const duration = formatDuration(result.duration);
		console.log(`${status} ${result.name} (${duration})`);

		if (!result.success && result.error) {
			console.log(`   Error: ${result.error}`);
		}

		if (result.success) {
			passed++;
		} else {
			failed++;
		}
	}

	console.log("================");
	console.log(`✅ Passed: ${passed}`);
	console.log(`❌ Failed: ${failed}`);
	console.log(`📈 Total: ${results.length}`);

	if (failed > 0) {
		process.exit(1);
	}
}
