#!/usr/bin/env bun

/**
 * Simple example client using @cj/comfy-client
 * Adapted from proxy/tests/client.example.ts
 */

import { env } from "../src/env";
import { BunWebSocketAdapter, ComfyClient } from "../src/index";
import {
	createTestPrompt,
	generateTestJWT,
	generateUniqueUserId,
} from "./utils";

async function example() {
	console.log("🚀 Starting ComfyClient example...");

	// Create client with test configuration
	const userId = generateUniqueUserId("example");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "info" },
		onConnectionChange: (state) => {
			console.log("🔌 Connection state:", state);
		},
		onError: (err) => {
			console.error("❌ Client error:", err);
		},
		onMessage: (msg) => {
			console.log("📨 Received message:", msg);
		},
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.PROXY_URL, // Connect to proxy, not ComfyUI directly
	});

	try {
		// Connect
		console.log("🔗 Connecting to ComfyUI...");
		const connectResult = await client.connect();

		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		console.log("✅ Connected successfully!");

		// Submit a prompt
		console.log("📤 Submitting prompt...");
		const prompt = createTestPrompt();
		const submitResult = await client.submitPrompt(prompt, {
			extraData: {
				example: true,
				timestamp: Date.now(),
			},
			promptId: `example-${Date.now()}`,
		});

		if (!submitResult.success) {
			throw new Error(`Prompt submission failed: ${submitResult.error}`);
		}

		console.log("✅ Prompt submitted successfully!");

		// Wait for acceptance
		console.log("⏳ Waiting for prompt acceptance...");
		const acceptResult = await client.waitForEvent("prompt_accepted");

		if (!acceptResult.success) {
			throw new Error(`Failed to get prompt acceptance: ${acceptResult.error}`);
		}

		console.log("✅ Prompt accepted!");

		// Optional: Collect events for a short time
		console.log("📊 Collecting events for 10 seconds...");
		const collectResult = await client.collectAllEvents({
			timeout: 10000,
			waitForCompletion: false,
		});

		if (collectResult.success) {
			const { events, binaryData } = collectResult.data;
			console.log(
				`📈 Collected ${events.length} events and ${binaryData.length} binary chunks`,
			);
		}

		console.log("🎉 Example completed successfully!");
	} catch (error) {
		console.error("💥 Example failed:", error);
		process.exit(1);
	} finally {
		// Clean up
		console.log("🧹 Disconnecting...");
		client.disconnect();
		console.log("👋 Goodbye!");
	}
}

if (import.meta.main) {
	example().catch(console.error);
}
