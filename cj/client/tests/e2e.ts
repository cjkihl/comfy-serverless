#!/usr/bin/env bun

/**
 * Comprehensive E2E tests for ComfyUI WebSocket proxy
 * Tests real workflows against remote ComfyUI server
 *
 * Run with: bun tests/e2e.ts
 */

import { env } from "../src/env";
import {
	BunWebSocketAdapter,
	ComfyClient,
	type ComfyPrompt,
} from "../src/index";
import {
	createTestPrompt,
	formatDuration,
	generateTestJWT,
	generateUniqueUserId,
	isValidImageData,
	printTestResults,
	runTest,
	validateEventSequence,
} from "./utils";

interface MetricsResponse {
	active_sessions: number;
	active_connections: number;
	circuit_breaker_state: string;
	queued_prompts: Record<string, number>;
}

/**
 * Test 1: Single User Complete Workflow
 */
async function testSingleUserWorkflow(): Promise<void> {
	console.log("🔍 Testing single user complete workflow...");

	const userId = generateUniqueUserId("e2e-single");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "info" },
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.PROXY_URL,
	});

	try {
		// Connect
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		// Submit prompt
		const prompt = createTestPrompt();
		const submitResult = await client.submitPrompt(prompt, {
			extraData: {
				test: "e2e-single",
				timestamp: Date.now(),
				userId,
			},
			promptId: `e2e-single-${Date.now()}`,
		});

		if (!submitResult.success) {
			throw new Error(`Prompt submission failed: ${submitResult.error}`);
		}

		console.log("✅ Prompt submitted successfully");

		// Wait for acceptance
		const acceptResult = await client.waitForEvent("prompt_accepted");
		if (!acceptResult.success) {
			throw new Error(`Failed to get prompt acceptance: ${acceptResult.error}`);
		}

		console.log("✅ Prompt accepted");

		// Collect all events
		const collectResult = await client.collectAllEvents({
			timeout: 120000, // 2 minutes
			waitForCompletion: true,
		});

		if (!collectResult.success) {
			throw new Error(`Event collection failed: ${collectResult.error}`);
		}

		const { events, binaryData, completed, error } = collectResult.data;

		console.log(`📊 Collected ${events.length} events`);
		console.log(`🖼️  Received ${binaryData.length} binary data chunks`);

		if (error) {
			throw new Error(`Execution error: ${error}`);
		}

		if (!completed) {
			throw new Error("Execution did not complete successfully");
		}

		// Validate event sequence
		const validation = validateEventSequence(events);
		if (!validation.valid) {
			console.warn(
				`⚠️  Event sequence validation failed: missing ${validation.missingEvents.join(", ")}`,
			);
		}

		// Check for valid image data
		let validImages = 0;
		for (const data of binaryData) {
			if (isValidImageData(data)) {
				validImages++;
			}
		}

		console.log(`🖼️  Found ${validImages} valid images`);

		if (validImages === 0) {
			console.warn("⚠️  No valid images found in binary data");
		}

		console.log("✅ Single user workflow completed successfully");
	} finally {
		client.disconnect();
	}
}

/**
 * Test 2: Concurrent Users
 */
async function testConcurrentUsers(): Promise<void> {
	console.log("🔍 Testing concurrent users...");

	const userCount = 3;
	const promises: Promise<void>[] = [];

	for (let i = 0; i < userCount; i++) {
		promises.push(testConcurrentUser(i));
	}

	await Promise.all(promises);
	console.log("✅ Concurrent users test completed");
}

async function testConcurrentUser(userIndex: number): Promise<void> {
	const userId = generateUniqueUserId(`concurrent-${userIndex}`);
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "silent" },
		reconnect: { enabled: true, maxRetries: 2 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.PROXY_URL,
	});

	try {
		// Connect
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(
				`User ${userIndex} connection failed: ${connectResult.error}`,
			);
		}

		// Submit prompt
		const prompt = createTestPrompt();
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `concurrent-${userIndex}-${Date.now()}`,
		});

		if (!submitResult.success) {
			throw new Error(
				`User ${userIndex} prompt submission failed: ${submitResult.error}`,
			);
		}

		// Wait for acceptance
		const acceptResult = await client.waitForEvent("prompt_accepted");
		if (!acceptResult.success) {
			throw new Error(
				`User ${userIndex} failed to get prompt acceptance: ${acceptResult.error}`,
			);
		}

		console.log(`✅ User ${userIndex} prompt accepted`);
	} finally {
		client.disconnect();
	}
}

/**
 * Test 3: Error Handling
 */
async function testErrorHandling(): Promise<void> {
	console.log("🔍 Testing error handling...");

	// Test invalid JWT
	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt: "invalid-jwt" },
		logging: { level: "silent" },
		url: env.PROXY_URL,
	});

	const connectResult = await client.connect();
	if (connectResult.success) {
		console.warn(
			"⚠️  Connection with invalid JWT succeeded (may be expected if NO_VERIFY=true)",
		);
	} else {
		console.log("✅ Invalid JWT properly rejected");
	}

	client.disconnect();

	// Test invalid prompt
	const validClient = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: env.PROXY_URL,
	});

	await validClient.connect();

	const invalidPrompt = {} as ComfyPrompt;
	const submitResult = await validClient.submitPrompt(invalidPrompt);

	if (submitResult.success) {
		throw new Error("Invalid prompt should have been rejected");
	}

	console.log("✅ Invalid prompt properly rejected");

	validClient.disconnect();
}

/**
 * Test 4: Connection Pooling
 */
async function testConnectionPooling(): Promise<void> {
	console.log("🔍 Testing connection pooling...");

	// Test metrics endpoint
	try {
		const response = await fetch(env.METRICS_URL, {
			headers: {
				Authorization: `Bearer ${process.env.METRICS_SECRET || "test-secret"}`,
			},
		});

		if (response.ok) {
			const metrics = (await response.json()) as MetricsResponse;
			console.log("✅ Metrics endpoint working:", {
				active_connections: metrics.active_connections,
				active_sessions: metrics.active_sessions,
				circuit_breaker_state: metrics.circuit_breaker_state,
			});
		} else {
			console.error(
				"❌ Metrics endpoint failed:",
				response.status,
				response.statusText,
			);
		}
	} catch (error) {
		console.error("❌ Metrics test failed:", error);
	}
}

/**
 * Test 5: Performance Benchmark
 */
async function testPerformanceBenchmark(): Promise<void> {
	console.log("🔍 Running performance benchmark...");

	const iterations = 3;
	const results: number[] = [];

	for (let i = 0; i < iterations; i++) {
		const userId = generateUniqueUserId(`perf-${i}`);
		const jwt = generateTestJWT(userId);

		const adapter = new BunWebSocketAdapter();
		const client = new ComfyClient({
			adapter,
			auth: { jwt },
			logging: { level: "silent" },
			timeout: { connect: 10000, message: 30000, operation: 120000 },
			url: env.PROXY_URL,
		});

		const start = performance.now();

		try {
			// Connect
			const connectResult = await client.connect();
			if (!connectResult.success) {
				throw new Error(`Connection failed: ${connectResult.error}`);
			}

			// Submit prompt
			const prompt = createTestPrompt();
			const submitResult = await client.submitPrompt(prompt, {
				promptId: `perf-${i}-${Date.now()}`,
			});

			if (!submitResult.success) {
				throw new Error(`Prompt submission failed: ${submitResult.error}`);
			}

			// Wait for completion
			const collectResult = await client.collectAllEvents({
				timeout: 120000,
				waitForCompletion: true,
			});

			if (!collectResult.success) {
				throw new Error(`Event collection failed: ${collectResult.error}`);
			}

			const duration = performance.now() - start;
			results.push(duration);

			console.log(`📊 Iteration ${i + 1}: ${formatDuration(duration)}`);
		} finally {
			client.disconnect();
		}
	}

	const avgDuration = results.reduce((a, b) => a + b, 0) / results.length;
	const minDuration = Math.min(...results);
	const maxDuration = Math.max(...results);

	console.log("📈 Performance Results:");
	console.log(`   Average: ${formatDuration(avgDuration)}`);
	console.log(`   Min: ${formatDuration(minDuration)}`);
	console.log(`   Max: ${formatDuration(maxDuration)}`);
}

/**
 * Test 6: Connection Limit Enforcement
 */
async function testConnectionLimit(): Promise<void> {
	console.log("🔍 Testing connection limit enforcement...");

	const userId = generateUniqueUserId("connection-limit");
	const jwt = generateTestJWT(userId);

	// Create first connection
	const adapter1 = new BunWebSocketAdapter();
	const client1 = new ComfyClient({
		adapter: adapter1,
		auth: { jwt },
		logging: { level: "silent" },
		url: env.PROXY_URL,
	});

	try {
		const connectResult1 = await client1.connect();
		if (!connectResult1.success) {
			throw new Error(`First connection failed: ${connectResult1.error}`);
		}

		console.log("✅ First connection established");

		// Try to create second connection with same user
		const adapter2 = new BunWebSocketAdapter();
		const client2 = new ComfyClient({
			adapter: adapter2,
			auth: { jwt },
			logging: { level: "silent" },
			url: env.PROXY_URL,
		});

		const connectResult2 = await client2.connect();
		if (connectResult2.success) {
			throw new Error("Second connection should have been rejected");
		}

		console.log("✅ Second connection properly rejected");

		// Disconnect first client
		client1.disconnect();
		await new Promise((resolve) => setTimeout(resolve, 500));

		// Now third connection should succeed
		const adapter3 = new BunWebSocketAdapter();
		const client3 = new ComfyClient({
			adapter: adapter3,
			auth: { jwt },
			logging: { level: "silent" },
			url: env.PROXY_URL,
		});

		const connectResult3 = await client3.connect();
		if (!connectResult3.success) {
			throw new Error(`Third connection should have succeeded after disconnect: ${connectResult3.error}`);
		}

		console.log("✅ Third connection succeeded after disconnect");
		client3.disconnect();
	} finally {
		client1.disconnect();
	}
}

async function main() {
	console.log("🧪 Running ComfyClient E2E Tests...\n");

	const tests = [
		runTest("Single User Workflow", testSingleUserWorkflow),
		runTest("Concurrent Users", testConcurrentUsers),
		runTest("Error Handling", testErrorHandling),
		runTest("Connection Pooling", testConnectionPooling),
		runTest("Performance Benchmark", testPerformanceBenchmark),
		runTest("Connection Limit", testConnectionLimit),
	];

	const results = await Promise.all(tests);
	printTestResults(results);
}

if (import.meta.main) {
	main().catch(console.error);
}
