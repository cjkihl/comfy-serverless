#!/usr/bin/env bun

/**
 * Script to run the test prompt and save the output image
 * This allows visual comparison of the generated image
 *
 * ⚠️ IMPORTANT: ComfyUI runs on a REMOTE server!
 * Make sure COMFY_URL is set in your proxy environment
 */

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { env } from "../src/env";
import { BunWebSocketAdapter, ComfyClient } from "../src/index";
import { testPrompt } from "./test-prompt";
import { generateTestJWT, generateUniqueUserId } from "./utils";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main() {
	console.log("🚀 Running test prompt...\n");

	const userId = generateUniqueUserId("test-prompt");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "info" }, // Set to info for cleaner output
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 5000, message: 10000, operation: 20000 }, // Max 20 seconds for operations
		url: env.PROXY_URL || "ws://localhost:8190/ws",
	});

	try {
		// Connect
		console.log("📡 Connecting to server...");
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}
		console.log("✅ Connected successfully\n");

		// Submit prompt
		console.log("📤 Submitting prompt...");
		console.log("   Prompt has", Object.keys(testPrompt).length, "nodes");

		const submitResult = await client.submitPrompt(testPrompt, {
			promptId: `test-prompt-${Date.now()}`,
		});

		if (!submitResult.success) {
			console.error("❌ Submit result:", submitResult.error);
			throw new Error(`Prompt submission failed: ${submitResult.error}`);
		}
		console.log("✅ Prompt submitted and accepted\n");

		// Collect all events
		console.log("📥 Collecting events and binary data...");
		const collectResult = await client.collectAllEvents({
			timeout: 20000, // 20 seconds max
			waitForCompletion: true,
		});

		if (!collectResult.success) {
			throw new Error(`Event collection failed: ${collectResult.error}`);
		}

		const { events, binaryData, completed, error } = collectResult.data;

		console.log(`\n📊 Collected ${events.length} events`);
		console.log(`🖼️  Received ${binaryData.length} binary data chunks`);

		if (error) {
			throw new Error(`Execution error: ${error}`);
		}

		if (!completed) {
			throw new Error("Execution did not complete successfully");
		}

		// Save binary data as images (for binary images)
		if (binaryData.length > 0) {
			console.log("\n💾 Saving output images (binary)...");
			binaryData.forEach((data, index) => {
				const outputPath = join(
					__dirname,
					`bear-kid-generated-${index + 1}.webp`,
				);
				writeFileSync(outputPath, Buffer.from(data));
				console.log(`✅ Saved: ${outputPath}`);
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
			console.log("\n💾 Saving output images (base64)...");
			base64Images.forEach((base64Image, index) => {
				// Extract the base64 data and mime type
				const match = base64Image.match(/^data:image\/(\w+);base64,(.+)$/);
				if (match?.[1] && match[2]) {
					const mimeType = match[1];
					const base64Data = match[2];
					const outputPath = join(
						__dirname,
						`bear-kid-generated-${index + 1}.${mimeType}`,
					);
					writeFileSync(outputPath, Buffer.from(base64Data, "base64"));
					console.log(`✅ Saved: ${outputPath}`);
				}
			});
		} else if (binaryData.length === 0) {
			console.log(
				"\n⚠️  No images received. Check that SaveImageBase64 node is in the workflow.",
			);
		}

		// Print event summary
		console.log("\n📋 Event Summary:");
		const eventTypes = events.map((e) => (e as { type: string }).type);
		const uniqueTypes = [...new Set(eventTypes)];
		uniqueTypes.forEach((type) => {
			const count = eventTypes.filter((t) => t === type).length;
			console.log(`  ${type}: ${count}`);
		});

		console.log("\n✅ Test completed successfully!");
	} catch (error) {
		console.error("\n❌ Error:", error);
		process.exit(1);
	} finally {
		client.disconnect();
		console.log("\n👋 Disconnected");
	}
}

if (import.meta.main) {
	main().catch(console.error);
}
