#!/usr/bin/env bun

/**
 * Script to run the test prompt and save the output image
 * This allows visual comparison of the generated image
 *
 * ⚠️ IMPORTANT: ComfyUI runs on a REMOTE server!
 * Make sure PROXY_URL is set in your environment
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ComfyClient } from "@cj/comfy-client";
import { UniversalWebSocketAdapter } from "@cj/comfy-client/adapters/universal";
import { loadEnv } from "@cjkihl/with-env";
import { generateJWT } from "../../comfy-auth/index.pub";
import { Logger } from "../src/shared/logger";
import { getLightTestPrompt, getTestPrompt } from "../src/shared/test-prompt";
import { loadTestImageBase64 } from "../tests/utils-bun";
import { generateUniqueUserId } from "../tests/utils-test";

const logger = new Logger("info", "[run-test-prompt]");

await loadEnv();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OUTPUT_DIR = join(__dirname, "../test-results");
const timestamp = Date.now();

// Ensure output directory exists
if (!existsSync(OUTPUT_DIR)) {
	mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function main() {
	logger.info("🚀 Running test prompt...\n");

	const userId = generateUniqueUserId("test-prompt");
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
		logging: { level: "info" }, // Set to info for cleaner output
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 5000, message: 10000, operation: 20000 }, // Max 20 seconds for operations
		url: process.env.PROXY_URL || "ws://localhost:8190/ws",
	});

	try {
		// Connect
		logger.info("📡 Connecting to server...");
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}
		logger.info("✅ Connected successfully\n");

		// Submit prompt
		logger.info("📤 Submitting prompt...");
		const imageBase64 = loadTestImageBase64();
		const testPrompt =
			process.env.VITE_LIGHT_PROMPT === "true"
				? getLightTestPrompt(imageBase64)
				: getTestPrompt(imageBase64);
		logger.debug("   Prompt has", Object.keys(testPrompt).length, "nodes");

		const submitResult = await client.submitPrompt(testPrompt, {
			promptId: `test-prompt-${timestamp}`,
		});

		if (!submitResult.success) {
			logger.error("❌ Submit result:", submitResult.error);
			throw new Error(`Prompt submission failed: ${submitResult.error}`);
		}
		logger.info("✅ Prompt submitted and accepted\n");

		// Collect all events
		logger.info("📥 Collecting events and binary data...");
		const collectResult = await client.collectAllEvents({
			timeout: 20000, // 20 seconds max
			waitForCompletion: true,
		});

		if (!collectResult.success) {
			throw new Error(`Event collection failed: ${collectResult.error}`);
		}

		const { events, binaryData, completed, error } = collectResult.data;

		logger.info(`\n📊 Collected ${events.length} events`);
		logger.info(`🖼️  Received ${binaryData.length} binary data chunks`);

		if (error) {
			throw new Error(`Execution error: ${error}`);
		}

		if (!completed) {
			throw new Error("Execution did not complete successfully");
		}

		let imageIndex = 1;

		// Save binary data as images (for binary images)
		if (binaryData.length > 0) {
			logger.info("\n💾 Saving output images (binary)...");
			binaryData.forEach((data) => {
				const outputPath = join(
					OUTPUT_DIR,
					`bear-kid-real-${timestamp}-${imageIndex}.webp`,
				);
				writeFileSync(outputPath, Buffer.from(data));
				logger.info(`✅ Saved: ${outputPath}`);
				imageIndex++;
			});
		}

		// Extract base64 images from events (SaveImageBase64 outputs base64 strings)
		const base64Images: string[] = [];
		events.forEach((event) => {
			if (typeof event === "object" && event !== null && "type" in event) {
				const typedEvent = event as { type: string; data?: unknown };
				if (typedEvent.type === "executed" && typedEvent.data) {
					const data = typedEvent.data as {
						node?: string;
						output?: { result?: unknown[] };
					};
					if (data.output?.result && Array.isArray(data.output.result)) {
						data.output.result.forEach((item) => {
							if (
								typeof item === "object" &&
								item !== null &&
								"image" in item
							) {
								const imageData = (item as { image?: string }).image;
								if (typeof imageData === "string") {
									// Handle both data: URLs and raw base64 strings
									if (imageData.startsWith("data:")) {
										base64Images.push(imageData);
									} else {
										// Assume it's a raw base64 string for WebP
										base64Images.push(`data:image/webp;base64,${imageData}`);
									}
								}
							}
						});
					}
				}
			}
		});

		if (base64Images.length > 0) {
			logger.info("\n💾 Saving output images (base64)...");
			base64Images.forEach((base64Image) => {
				// Extract the base64 data and mime type
				const match = base64Image.match(/^data:image\/(\w+);base64,(.+)$/);
				if (match?.[1] && match[2]) {
					const mimeType = match[1];
					const base64Data = match[2];
					const outputPath = join(
						OUTPUT_DIR,
						`bear-kid-real-${timestamp}-${imageIndex}.${mimeType}`,
					);
					writeFileSync(outputPath, Buffer.from(base64Data, "base64"));
					logger.info(`✅ Saved: ${outputPath}`);
					imageIndex++;
				}
			});
		} else if (binaryData.length === 0) {
			logger.warn(
				"\n⚠️  No images received. Check that SaveImageBase64 node is in the workflow.",
			);
		}

		// Print event summary
		logger.info("\n📋 Event Summary:");
		const eventTypes = events.map((e) => (e as { type: string }).type);
		const uniqueTypes = [...new Set(eventTypes)];
		uniqueTypes.forEach((type) => {
			const count = eventTypes.filter((t) => t === type).length;
			logger.info(`  ${type}: ${count}`);
		});

		logger.info("\n✅ Test completed successfully!");
	} catch (error) {
		logger.error("\n❌ Error:", error);
		process.exit(1);
	} finally {
		client.disconnect();
		logger.info("\n👋 Disconnected");
	}
}

if (import.meta.main) {
	main().catch((error) => {
		logger.error("Unhandled error:", error);
		process.exit(1);
	});
}
