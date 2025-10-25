#!/usr/bin/env bun

/**
 * Session management tests for @cj/comfy-client package
 * Tests session limits, cleanup, and concurrent connections
 */

import { env } from "../src/env";
import { BunWebSocketAdapter, ComfyClient } from "../src/index";
import {
	createTestPrompt,
	generateTestJWT,
	generateUniqueUserId,
	printTestResults,
	runTest,
} from "./utils";

const PROXY_URL = env.PROXY_URL;

async function testMaxConnectionsLimit(): Promise<void> {
	console.log("🔍 Testing MAX_CONNECTIONS_PER_USER limit...");

	const userId = generateUniqueUserId("connection-limit");
	const jwt = generateTestJWT(userId);

	// Create first connection
	const adapter1 = new BunWebSocketAdapter();
	const client1 = new ComfyClient({
		adapter: adapter1,
		auth: { jwt },
		logging: { level: "silent" },
		url: PROXY_URL,
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
			url: PROXY_URL,
		});

		const connectResult2 = await client2.connect();
		if (connectResult2.success) {
			throw new Error("Second connection should have been rejected");
		}

		console.log("✅ Second connection properly rejected");
	} finally {
		client1.disconnect();
	}
}

async function testDisconnectFreesSlot(): Promise<void> {
	console.log("🔍 Testing that disconnect frees connection slot...");

	const userId = generateUniqueUserId("disconnect-slot");
	const jwt = generateTestJWT(userId);

	// Create and connect first client
	const adapter1 = new BunWebSocketAdapter();
	const client1 = new ComfyClient({
		adapter: adapter1,
		auth: { jwt },
		logging: { level: "silent" },
		url: PROXY_URL,
	});

	try {
		const connectResult1 = await client1.connect();
		if (!connectResult1.success) {
			throw new Error(`First connection failed: ${connectResult1.error}`);
		}

		console.log("✅ First connection established");

		// Disconnect first client
		client1.disconnect();
		console.log("✅ First client disconnected");

		// Wait a moment for cleanup
		await new Promise((resolve) => setTimeout(resolve, 500));

		// Now try to connect again - should succeed
		const adapter2 = new BunWebSocketAdapter();
		const client2 = new ComfyClient({
			adapter: adapter2,
			auth: { jwt },
			logging: { level: "silent" },
			url: PROXY_URL,
		});

		const connectResult2 = await client2.connect();
		if (!connectResult2.success) {
			throw new Error(
				`Second connection should have succeeded after disconnect: ${connectResult2.error}`,
			);
		}

		console.log("✅ Second connection succeeded after first disconnect");
		client2.disconnect();
	} finally {
		client1.disconnect();
	}
}

async function testConcurrentUsers(): Promise<void> {
	console.log("🔍 Testing concurrent users (should not conflict)...");

	const promises: Promise<void>[] = [];

	// Create multiple connections with different users
	for (let i = 0; i < 3; i++) {
		promises.push(testConcurrentUser(i));
	}

	await Promise.all(promises);
	console.log("✅ Concurrent users handled correctly");
}

async function testConcurrentUser(index: number): Promise<void> {
	const userId = generateUniqueUserId(`concurrent-${index}`);
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "silent" },
		url: PROXY_URL,
	});

	try {
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(
				`User ${index} connection failed: ${connectResult.error}`,
			);
		}

		const prompt = createTestPrompt();
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `concurrent-${index}-${Date.now()}`,
		});

		if (!submitResult.success) {
			throw new Error(
				`User ${index} prompt submission failed: ${submitResult.error}`,
			);
		}

		console.log(`✅ User ${index} connected and submitted prompt`);
	} finally {
		client.disconnect();
	}
}

async function main() {
	console.log("🧪 Running Session Management Tests...\n");

	const tests = [
		runTest("Max Connections Limit", testMaxConnectionsLimit),
		runTest("Disconnect Frees Slot", testDisconnectFreesSlot),
		runTest("Concurrent Users", testConcurrentUsers),
	];

	const results = await Promise.all(tests);
	printTestResults(results);
}

if (import.meta.main) {
	main().catch(console.error);
}
