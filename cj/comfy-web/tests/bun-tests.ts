#!/usr/bin/env bun

/**
 * Simplified full test for Bun environment
 * Connects to proxy, sends test prompt, waits for base64 result, saves output
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ComfyClient } from "@cj/comfy-client";
import { UniversalWebSocketAdapter } from "@cj/comfy-client/adapters/universal";
import { loadEnv } from "@cjkihl/with-env";
import { generateTestJWT } from "../src/shared/jwt";
import { Logger } from "../src/shared/logger";
import { getLightTestPrompt, getTestPrompt } from "../src/shared/test-prompt";
import { loadTestImageBase64 } from "./utils-bun";
import { generateUniqueUserId } from "./utils-test";

const logger = new Logger("info", "[bun-test]");

await loadEnv();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const OUTPUT_DIR = join(__dirname, "../test-results");
const PROXY_URL = process.env.PROXY_URL || "ws://localhost:8190/ws";

// Ensure output directory exists
if (!existsSync(OUTPUT_DIR)) {
	mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function main(): Promise<void> {
	logger.info("🚀 Running simplified full test (Bun)...\n");
	logger.info(`Output directory: ${OUTPUT_DIR}\n`);

	const startTime = performance.now();
	const userId = generateUniqueUserId("full-test");
	const jwt = generateTestJWT(userId);

	const client = new ComfyClient({
		adapter: new UniversalWebSocketAdapter(PROXY_URL),
		auth: { jwt },
		autoConnect: false,
		logging: { level: "info" },
		timeout: { connect: 10000, message: 30000, operation: 30000 }, // 30 seconds max
		url: PROXY_URL,
	});

	try {
		// 1. Connect to proxy
		logger.info("📡 Connecting to proxy...");
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}
		logger.info("✅ Connected successfully\n");

		// 2. Get test prompt (uses bear-kid.png from file)
		logger.info("📤 Preparing test prompt...");
		const imageBase64 = loadTestImageBase64();
		const prompt =
			process.env.VITE_LIGHT_PROMPT === "true"
				? getLightTestPrompt(imageBase64)
				: getTestPrompt(imageBase64);
		const saveImageEntry = Object.entries(prompt).find(
			([, node]) => node.class_type === "SaveImageBase64",
		);
		if (!saveImageEntry) {
			throw new Error("SaveImageBase64 node not found in test prompt");
		}
		const [saveImageNodeId] = saveImageEntry;
		logger.info(
			`✅ Prompt prepared with ${Object.keys(prompt).length} nodes (SaveImageBase64 node: ${saveImageNodeId})\n`,
		);

		// 3. Submit prompt
		logger.info("📤 Submitting prompt...");
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `full-test-${Date.now()}`,
		});
		if (!submitResult.success) {
			throw new Error(`Prompt submission failed: ${submitResult.error}`);
		}
		logger.info("✅ Prompt submitted and accepted\n");

		// 4. Collect events and wait for base64 result
		logger.info("📥 Collecting events (waiting for base64 result)...");
		const collectResult = await client.collectAllEvents({
			timeout: 30000, // 30 seconds max (increased for debugging)
			waitForCompletion: true,
		});

		if (!collectResult.success) {
			throw new Error(`Event collection failed: ${collectResult.error}`);
		}

		const { events, completed, error } = collectResult.data;

		// Debug: log received events
		logger.info(`📊 Received ${events.length} events`);
		for (const event of events.slice(0, 10)) {
			if (typeof event === "object" && event !== null && "type" in event) {
				logger.debug(`  - Event type: ${(event as { type: string }).type}`);
			}
		}

		if (error) {
			throw new Error(`Execution error: ${error}`);
		}
		if (!completed) {
			throw new Error("Execution did not complete successfully");
		}

		// 5. Extract base64 image from SaveImageBase64 node output
		let base64Image: string | null = null;
		let saveImageNodeExecuted = false;
		for (const event of events) {
			if (
				typeof event === "object" &&
				event !== null &&
				"type" in event &&
				event.type === "executed"
			) {
				const typedEvent = event as { type: string; data?: unknown };
				const data = typedEvent.data as
					| {
							node?: string;
							output?: { result?: Array<{ image?: unknown }> };
					  }
					| undefined;
				if (data?.node === saveImageNodeId) {
					saveImageNodeExecuted = true;
					if (data.output?.result && Array.isArray(data.output.result)) {
						for (const item of data.output.result) {
							if (
								item &&
								typeof item === "object" &&
								"image" in item &&
								typeof item.image === "string"
							) {
								const candidate = item.image.trim();
								const normalized = candidate.startsWith("data:")
									? candidate.split(",")[1] || ""
									: candidate;
								if (normalized.length === 0) {
									continue;
								}
								const buffer = Buffer.from(normalized, "base64");
								if (buffer.length === 0) {
									continue;
								}
								base64Image = normalized;
								break;
							}
						}
					}
				}
				if (base64Image) break;
			}
		}

		if (!saveImageNodeExecuted) {
			throw new Error("SaveImageBase64 node never executed");
		}

		if (!base64Image) {
			throw new Error("No valid base64 image found in SaveImageBase64 output");
		}

		logger.info("✅ Base64 image received\n");

		// 6. Save to output folder
		const timestamp = Date.now();
		const outputPath = join(OUTPUT_DIR, `generated-${timestamp}.webp`);
		writeFileSync(outputPath, Buffer.from(base64Image, "base64"));
		logger.info(`💾 Saved image: ${outputPath}\n`);

		// 7. Assert completion time < 30 seconds (actual execution may vary)
		const duration = performance.now() - startTime;
		logger.info(`⏱️  Total execution time: ${(duration / 1000).toFixed(2)}s`);
		if (duration >= 30000) {
			throw new Error(
				`Test took ${(duration / 1000).toFixed(2)}s, expected < 30s`,
			);
		}
		if (duration < 10000) {
			logger.info("✅ Test completed successfully in less than 10 seconds\n");
		} else {
			logger.warn(
				`⚠️  Test completed in ${(duration / 1000).toFixed(2)}s (target was < 10s, but within 30s limit)\n`,
			);
		}

		process.exit(0);
	} catch (error) {
		const duration = performance.now() - startTime;
		logger.error(
			`❌ Test failed after ${(duration / 1000).toFixed(2)}s:`,
			error,
		);
		process.exit(1);
	} finally {
		client.disconnect();
	}
}

if (import.meta.main) {
	main().catch((error) => {
		logger.error("Unhandled error:", error);
		process.exit(1);
	});
}
