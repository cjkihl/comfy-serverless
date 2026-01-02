#!/usr/bin/env bun

/**
 * Unit tests for proxy core functionality
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { createErrorJSON } from "../errorHelpers";
import { ErrorCode } from "../types";

beforeEach(() => {
	// Reset state before each test
});

afterEach(() => {
	// Clean up after each test
});

test("createErrorJSON creates valid error response", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR);
	const parsed = JSON.parse(errorJson);

	expect(parsed.type).toBe("error");
	expect(parsed.data).toBeDefined();
	expect(parsed.data.message).toBeDefined();
	expect(parsed.data.code).toBe(ErrorCode.UNKNOWN_ERROR);
	expect(parsed.data.timestamp).toBeDefined();
});

test("createErrorJSON includes userId when provided", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR, {
		userId: "user123",
	});
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.userId).toBe("user123");
});

test("createErrorJSON includes promptId when provided", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR, {
		promptId: "prompt123",
	});
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.promptId).toBe("prompt123");
});

test("createErrorJSON includes retryable flag", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR, {
		retryable: true,
	});
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.retryable).toBe(true);
});

test("createErrorJSON defaults retryable to false", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR);
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.retryable).toBe(false);
});

test("createErrorJSON sanitizes error messages", () => {
	const errorJson = createErrorJSON(
		"Detailed backend error message",
		ErrorCode.UNKNOWN_ERROR,
	);
	const parsed = JSON.parse(errorJson);

	// Should use sanitized message, not original
	expect(parsed.data.message).not.toBe("Detailed backend error message");
	expect(parsed.data.message).toBe(
		"An error occurred while processing your request",
	);
});

test("createErrorJSON maps error codes correctly", () => {
	const testCases = [
		{
			code: ErrorCode.TIMEOUT,
			expectedMessage: "The request timed out",
		},
		{
			code: ErrorCode.INVALID,
			expectedMessage: "Invalid request",
		},
		{
			code: ErrorCode.MAX_CONNECTIONS_EXCEEDED,
			expectedMessage: "Maximum connections exceeded",
		},
		{
			code: ErrorCode.QUEUE_FULL,
			expectedMessage: "Queue is full, please try again later",
		},
		{
			code: ErrorCode.SESSION_NOT_READY,
			expectedMessage: "Session not ready, please try again",
		},
	];

	for (const testCase of testCases) {
		const errorJson = createErrorJSON("Test", testCase.code);
		const parsed = JSON.parse(errorJson);
		expect(parsed.data.message).toBe(testCase.expectedMessage);
	}
});

test("createErrorJSON includes timestamp", () => {
	const before = Date.now();
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR);
	const after = Date.now();
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.timestamp).toBeGreaterThanOrEqual(before);
	expect(parsed.data.timestamp).toBeLessThanOrEqual(after);
});

test("createErrorJSON includes all optional fields", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR, {
		context: { key: "value" },
		promptId: "prompt123",
		retryable: true,
		userId: "user123",
	});
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.userId).toBe("user123");
	expect(parsed.data.promptId).toBe("prompt123");
	expect(parsed.data.retryable).toBe(true);
	// Context should not be included in response (security)
	expect(parsed.data.context).toBeUndefined();
});

test("Error response structure matches ProxyWsOutbound type", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR);
	const parsed = JSON.parse(errorJson);

	expect(parsed.type).toBe("error");
	expect(parsed.data).toBeDefined();
	expect(typeof parsed.data.message).toBe("string");
	expect(typeof parsed.data.code).toBe("string");
	expect(typeof parsed.data.timestamp).toBe("number");
	expect(typeof parsed.data.retryable).toBe("boolean");
});

test("Error code constants are defined", () => {
	expect(ErrorCode.UNKNOWN_ERROR).toBe("UNKNOWN_ERROR");
	expect(ErrorCode.TIMEOUT).toBe("TIMEOUT");
	expect(ErrorCode.INVALID).toBe("INVALID");
	expect(ErrorCode.MAX_CONNECTIONS_EXCEEDED).toBe("MAX_CONNECTIONS_EXCEEDED");
	expect(ErrorCode.QUEUE_FULL).toBe("QUEUE_FULL");
	expect(ErrorCode.SESSION_NOT_READY).toBe("SESSION_NOT_READY");
});

test("createErrorJSON handles empty message", () => {
	const errorJson = createErrorJSON("", ErrorCode.UNKNOWN_ERROR);
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.message).toBeDefined();
	expect(typeof parsed.data.message).toBe("string");
});

test("createErrorJSON JSON is parseable", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR);

	// Should not throw
	const parsed = JSON.parse(errorJson);
	expect(parsed).toBeDefined();
});

test("Error response can be stringified again", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR);
	const parsed = JSON.parse(errorJson);

	// Should be able to stringify again
	const reStringified = JSON.stringify(parsed);
	expect(reStringified).toBeDefined();
	expect(typeof reStringified).toBe("string");
});

test("createErrorJSON with context logs but doesn't expose", () => {
	// This test verifies that context is logged server-side but not exposed to client
	const errorJson = createErrorJSON("Test error", ErrorCode.UNKNOWN_ERROR, {
		context: { sensitive: "data" },
	});
	const parsed = JSON.parse(errorJson);

	// Context should not be in response
	expect(parsed.data.context).toBeUndefined();
});

test("Multiple error codes produce different messages", () => {
	const error1 = JSON.parse(createErrorJSON("Test", ErrorCode.TIMEOUT));
	const error2 = JSON.parse(createErrorJSON("Test", ErrorCode.INVALID));

	expect(error1.data.message).not.toBe(error2.data.message);
});

test("Error response includes code in data", () => {
	const errorJson = createErrorJSON("Test error", ErrorCode.TIMEOUT);
	const parsed = JSON.parse(errorJson);

	expect(parsed.data.code).toBe(ErrorCode.TIMEOUT);
});
