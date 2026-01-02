#!/usr/bin/env bun

/**
 * Unit tests for health check functionality
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import type { RequestInfo, RequestInit } from "undici";
import { Logger } from "../logger";
import {
	checkComfyHealth,
	startPeriodicHealthCheck,
	stopPeriodicHealthCheck,
} from "../proxy";

// Mock fetch globally
const originalFetch = globalThis.fetch;

beforeEach(() => {
	// Reset fetch mock before each test
	globalThis.fetch = originalFetch;
});

afterEach(() => {
	// Clean up any intervals
	stopPeriodicHealthCheck();
	// Restore original fetch
	globalThis.fetch = originalFetch;
});

test("Health check succeeds with valid response", async () => {
	const mockFetch = async () => {
		return new Response(JSON.stringify({ memory: "usage", system: "stats" }), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(true);
});

test("Health check fails with non-200 status", async () => {
	const mockFetch = async () => {
		return new Response("Error", { status: 500 });
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(false);
});

test("Health check fails on timeout", async () => {
	const mockFetch = async () => {
		await new Promise((resolve) => setTimeout(resolve, 10000));
		return new Response(JSON.stringify({}), { status: 200 });
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 100);
	expect(isHealthy).toBe(false);
});

test("Health check fails on network error", async () => {
	const mockFetch = async () => {
		throw new Error("Network error");
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(false);
});

test("Health check fails with invalid JSON", async () => {
	const mockFetch = async () => {
		return new Response("not json", {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(false);
});

test("Health check fails with non-object JSON", async () => {
	const mockFetch = async () => {
		return new Response("null", {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(false);
});

test("Health check fails with array JSON", async () => {
	const mockFetch = async () => {
		return new Response("[]", {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(false);
});

test("Health check succeeds with object JSON", async () => {
	const mockFetch = async () => {
		return new Response(JSON.stringify({ status: "ok" }), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(true);
});

test("Health check uses correct URL", async () => {
	let capturedUrl = "";
	const mockFetch = async (url: RequestInfo | URL) => {
		capturedUrl = url.toString();
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	await checkComfyHealth("http://localhost:8188", 5000);
	expect(capturedUrl).toBe("http://localhost:8188/system_stats");
});

test("Health check handles missing content-type header", async () => {
	const mockFetch = async () => {
		return new Response(JSON.stringify({}), {
			headers: {},
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	// Should still succeed if body is valid JSON object
	expect(isHealthy).toBe(true);
});

test("Start periodic health check", () => {
	const logger = new Logger("silent");
	const comfyUrl = "http://localhost:8188";
	const intervalMs = 100;
	const timeoutMs = 5000;

	// Mock fetch to succeed
	const mockFetch = async () => {
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	// Should not throw
	startPeriodicHealthCheck(comfyUrl, intervalMs, timeoutMs, logger);

	// Clean up
	stopPeriodicHealthCheck(logger);
});

test("Stop periodic health check", () => {
	const logger = new Logger("silent");
	const comfyUrl = "http://localhost:8188";
	const intervalMs = 100;
	const timeoutMs = 5000;

	const mockFetch = async () => {
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	startPeriodicHealthCheck(comfyUrl, intervalMs, timeoutMs, logger);
	stopPeriodicHealthCheck(logger);

	// Should not throw when stopping
	expect(true).toBe(true);
});

test("Start periodic health check replaces existing interval", () => {
	const logger = new Logger("silent");
	const comfyUrl = "http://localhost:8188";
	const intervalMs = 100;
	const timeoutMs = 5000;

	const mockFetch = async () => {
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	// Start first interval
	startPeriodicHealthCheck(comfyUrl, intervalMs, timeoutMs, logger);

	// Start second interval - should replace first
	startPeriodicHealthCheck(comfyUrl, intervalMs, timeoutMs, logger);

	// Clean up
	stopPeriodicHealthCheck(logger);
});

test("Health check timeout uses Promise.race", async () => {
	const mockFetch = async () => {
		await new Promise((resolve) => setTimeout(resolve, 200));
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	// Short timeout
	const isHealthy = await checkComfyHealth("http://localhost:8188", 50);
	expect(isHealthy).toBe(false);
});

test("Health check validates response body structure", async () => {
	const mockFetch = async () => {
		return new Response(JSON.stringify({ valid: "object" }), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(true);
});

test("Health check handles different HTTP methods", async () => {
	let capturedMethod = "";
	const mockFetch = async (_input: RequestInfo | URL, init?: RequestInit) => {
		capturedMethod = init?.method || "GET";
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	await checkComfyHealth("http://localhost:8188", 5000);
	expect(capturedMethod).toBe("GET");
});

test("Health check with HTTPS URL", async () => {
	let capturedUrl = "";
	const mockFetch = async (url: RequestInfo | URL) => {
		capturedUrl = url.toString();
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	await checkComfyHealth("https://example.com:8188", 5000);
	expect(capturedUrl).toBe("https://example.com:8188/system_stats");
});

test("Health check handles empty object response", async () => {
	const mockFetch = async () => {
		return new Response(JSON.stringify({}), {
			headers: { "content-type": "application/json" },
			status: 200,
		});
	};

	globalThis.fetch = mockFetch as unknown as typeof fetch;

	const isHealthy = await checkComfyHealth("http://localhost:8188", 5000);
	expect(isHealthy).toBe(true);
});
