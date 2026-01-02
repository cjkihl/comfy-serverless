#!/usr/bin/env bun

/**
 * Debug test to see what messages are received from ComfyUI
 */

import { ComfyClient } from "@cj/comfy-client";
import { UniversalWebSocketAdapter } from "@cj/comfy-client/adapters/universal";
import { loadEnv } from "@cjkihl/with-env";
import { generateJWT } from "../../comfy-auth/index.pub";
import { Logger } from "../src/shared/logger";
import { generateUniqueUserId } from "../tests/utils-test";

const logger = new Logger("debug", "[debug]");

await loadEnv();

async function debugTest() {
	logger.info("🔍 Debug test - connecting and logging all messages...");

	const userId = generateUniqueUserId("debug");
	if (!process.env.PROXY_COMFY_JWT_SECRET) {
		throw new Error("PROXY_COMFY_JWT_SECRET is not set");
	}
	const jwt = generateJWT(userId, process.env.PROXY_COMFY_JWT_SECRET);

	const adapter = new UniversalWebSocketAdapter(
		process.env.PROXY_URL || "ws://localhost:8190/ws",
	);
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "debug" },
		onConnectionChange: (state) => {
			logger.debug("🔌 Connection state changed:", state);
		},
		onError: (err) => {
			logger.error("❌ Error:", err);
		},
		onMessage: (msg) => {
			logger.debug("📨 Message:", JSON.stringify(msg, null, 2));
		},
		url: process.env.PROXY_URL || "ws://localhost:8190/ws", // Connect to proxy, not ComfyUI directly
	});

	try {
		// Connect
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		logger.info("✅ Connected to ComfyUI");

		// Wait for messages for 15 seconds
		logger.info("⏳ Listening for messages for 15 seconds...");
		await new Promise((resolve) => setTimeout(resolve, 15000));

		logger.info("🔍 Debug test completed");
	} catch (error) {
		logger.error("💥 Debug test failed:", error);
		process.exit(1);
	} finally {
		client.disconnect();
	}
}

if (import.meta.main) {
	debugTest().catch((error) => {
		logger.error("Unhandled error:", error);
		process.exit(1);
	});
}
