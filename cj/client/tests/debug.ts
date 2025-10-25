#!/usr/bin/env bun

/**
 * Debug test to see what messages are received from ComfyUI
 * Adapted from proxy/tests/debug.ts
 */

import { env } from "../src/env";
import { BunWebSocketAdapter, ComfyClient } from "../src/index";
import { generateTestJWT, generateUniqueUserId } from "./utils";

async function debugTest() {
	console.log("🔍 Debug test - connecting and logging all messages...");

	const userId = generateUniqueUserId("debug");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "debug" },
		onConnectionChange: (state) => {
			console.log("🔌 Connection state changed:", state);
		},
		onError: (err) => {
			console.error("❌ Error:", err);
		},
		onMessage: (msg) => {
			console.log("📨 Message:", JSON.stringify(msg, null, 2));
		},
		url: env.PROXY_URL, // Connect to proxy, not ComfyUI directly
	});

	try {
		// Connect
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		console.log("✅ Connected to ComfyUI");

		// Wait for messages for 15 seconds
		console.log("⏳ Listening for messages for 15 seconds...");
		await new Promise((resolve) => setTimeout(resolve, 15000));

		console.log("🔍 Debug test completed");
	} catch (error) {
		console.error("💥 Debug test failed:", error);
		process.exit(1);
	} finally {
		client.disconnect();
	}
}

if (import.meta.main) {
	debugTest().catch(console.error);
}
