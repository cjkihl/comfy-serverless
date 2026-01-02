#!/usr/bin/env bun

/**
 * Session management tests for @cj/comfy-client package
 * Tests session limits, cleanup, and concurrent connections
 */

import { expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ComfyClient } from "@cj/comfy-client";
import { UniversalWebSocketAdapter } from "@cj/comfy-client/adapters/universal";
import { loadEnv } from "@cjkihl/with-env";
import { generateTestJWT } from "../src/shared/jwt";
import { getLightTestPrompt, getTestPrompt } from "../src/shared/test-prompt";
import { generateUniqueUserId } from "./utils-test";

await loadEnv();

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEST_IMAGE_PATH = join(__dirname, "bear-kid.png");
const PROXY_URL = process.env.PROXY_URL || "ws://localhost:8190/ws";

function loadTestImageBase64(): string {
	if (!existsSync(TEST_IMAGE_PATH)) {
		throw new Error(`Test image not found at ${TEST_IMAGE_PATH}`);
	}
	const imageBuffer = readFileSync(TEST_IMAGE_PATH);
	const bytes = new Uint8Array(imageBuffer);
	let binary = "";
	for (let i = 0; i < bytes.byteLength; i++) {
		binary += String.fromCharCode(bytes[i]!);
	}
	const base64 = btoa(binary);
	return `data:image/png;base64,${base64}`;
}

test("Max Connections Limit", async () => {
	console.log("🔍 Testing MAX_CONNECTIONS_PER_USER limit...");
	console.log(
		"ℹ️  Note: Default MAX_CONNECTIONS_PER_USER is 5, so multiple connections are allowed",
	);

	const userId = generateUniqueUserId("connection-limit");
	const jwt = generateTestJWT(userId);

	// Create first connection
	const adapter1 = new UniversalWebSocketAdapter(PROXY_URL);
	const client1 = new ComfyClient({
		adapter: adapter1,
		auth: { jwt },
		logging: { level: "silent" },
		url: PROXY_URL,
	});

	try {
		const connectResult1 = await client1.connect();
		expect(connectResult1.success).toBe(true);

		console.log("✅ First connection established");

		// Create second connection with same user (should succeed since limit is 5 by default)
		const adapter2 = new UniversalWebSocketAdapter(PROXY_URL);
		const client2 = new ComfyClient({
			adapter: adapter2,
			auth: { jwt },
			logging: { level: "silent" },
			url: PROXY_URL,
		});

		try {
			const connectResult2 = await client2.connect();
			// With default limit of 5, second connection should succeed
			expect(connectResult2.success).toBe(true);

			console.log("✅ Second connection established (within limit)");
		} finally {
			client2.disconnect();
		}
	} finally {
		client1.disconnect();
	}
});

test("Disconnect Frees Slot", async () => {
	console.log("🔍 Testing that disconnect frees connection slot...");

	const userId = generateUniqueUserId("disconnect-slot");
	const jwt = generateTestJWT(userId);

	// Create and connect first client
	const adapter1 = new UniversalWebSocketAdapter(PROXY_URL);
	const client1 = new ComfyClient({
		adapter: adapter1,
		auth: { jwt },
		logging: { level: "silent" },
		url: PROXY_URL,
	});

	try {
		const connectResult1 = await client1.connect();
		expect(connectResult1.success).toBe(true);

		console.log("✅ First connection established");

		// Disconnect first client
		client1.disconnect();
		console.log("✅ First client disconnected");

		// Wait a moment for cleanup
		await new Promise((resolve) => setTimeout(resolve, 500));

		// Now try to connect again - should succeed
		const adapter2 = new UniversalWebSocketAdapter(PROXY_URL);
		const client2 = new ComfyClient({
			adapter: adapter2,
			auth: { jwt },
			logging: { level: "silent" },
			url: PROXY_URL,
		});

		const connectResult2 = await client2.connect();
		expect(connectResult2.success).toBe(true);

		console.log("✅ Second connection succeeded after first disconnect");
		client2.disconnect();
	} finally {
		client1.disconnect();
	}
});

test("Concurrent Users", async () => {
	console.log("🔍 Testing concurrent users (should not conflict)...");

	const promises: Promise<void>[] = [];

	// Create multiple connections with different users
	for (let i = 0; i < 3; i++) {
		promises.push(testConcurrentUser(i));
	}

	await Promise.all(promises);
	console.log("✅ Concurrent users handled correctly");
});

async function testConcurrentUser(index: number): Promise<void> {
	const userId = generateUniqueUserId(`concurrent-${index}`);
	const jwt = generateTestJWT(userId);

	const adapter = new UniversalWebSocketAdapter(PROXY_URL);
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "silent" },
		url: PROXY_URL,
	});

	try {
		const connectResult = await client.connect();
		expect(connectResult.success).toBe(true);

		// Wait a bit for connection to be fully established
		await new Promise((resolve) => setTimeout(resolve, 100));

		const imageBase64 = loadTestImageBase64();
		const prompt =
			process.env.VITE_LIGHT_PROMPT === "true"
				? getLightTestPrompt(imageBase64)
				: getTestPrompt(imageBase64);
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `concurrent-${index}-${Date.now()}`,
		});

		if (!submitResult.success) {
			console.error(
				`❌ User ${index} prompt submission failed: ${submitResult.error}`,
			);
		}
		expect(submitResult.success).toBe(true);

		console.log(`✅ User ${index} connected and submitted prompt`);
	} finally {
		client.disconnect();
	}
}
