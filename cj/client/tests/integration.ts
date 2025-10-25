#!/usr/bin/env bun

/**
 * Integration tests for @cj/comfy-client package
 * Tests full workflows against the proxy server (which connects to ComfyUI)
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

// Test configuration - Integration tests connect to proxy, not ComfyUI directly
const PROXY_URL = env.PROXY_URL;

async function testSingleUserWorkflow(): Promise<void> {
	console.log("🔍 Testing single user complete workflow...");

	const userId = generateUniqueUserId("integration");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "info" },
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: PROXY_URL,
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
				test: "integration",
				timestamp: Date.now(),
				userId,
			},
			promptId: `integration-${Date.now()}`,
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
		url: PROXY_URL,
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

async function testErrorHandling(): Promise<void> {
	console.log("🔍 Testing error handling...");

	// Test invalid JWT
	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt: "invalid-jwt" },
		logging: { level: "silent" },
		url: PROXY_URL,
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
		url: PROXY_URL,
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

async function testReconnection(): Promise<void> {
	console.log("🔍 Testing reconnection behavior...");

	const userId = generateUniqueUserId("reconnect");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "info" },
		reconnect: {
			backoffMultiplier: 2,
			enabled: true,
			initialDelay: 1000,
			maxDelay: 5000,
			maxRetries: 3,
		},
		url: PROXY_URL,
	});

	try {
		// Connect
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		console.log("✅ Initial connection successful");

		// Simulate connection loss by closing the adapter
		adapter.close();

		// Wait a bit for reconnection to attempt
		await new Promise((resolve) => setTimeout(resolve, 2000));

		// Try to reconnect manually
		const reconnectResult = await client.connect();
		if (!reconnectResult.success) {
			throw new Error(`Reconnection failed: ${reconnectResult.error}`);
		}

		console.log("✅ Reconnection successful");
	} finally {
		client.disconnect();
	}
}

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
			url: PROXY_URL,
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

async function main() {
	console.log("🧪 Running ComfyClient Integration Tests...\n");

	const tests = [
		runTest("Single User Workflow", testSingleUserWorkflow),
		runTest("Concurrent Users", testConcurrentUsers),
		runTest("Error Handling", testErrorHandling),
		runTest("Reconnection", testReconnection),
		runTest("Performance Benchmark", testPerformanceBenchmark),
	];

	const results = await Promise.all(tests);
	printTestResults(results);
}

if (import.meta.main) {
	main().catch(console.error);
}
