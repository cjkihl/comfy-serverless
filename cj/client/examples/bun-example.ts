#!/usr/bin/env bun

/**
 * Example usage of @cj/comfy-client with Bun adapter
 *
 * This example demonstrates how to use the ComfyClient in a Bun environment
 * for server-side ComfyUI automation.
 */

import { env } from "../src/env";
import {
	BunWebSocketAdapter,
	ComfyClient,
	type ComfyPrompt,
} from "../src/index";

// Example ComfyUI prompt
const examplePrompt: ComfyPrompt = {
	"1": {
		class_type: "CLIPTextEncode",
		inputs: {
			clip: ["4", 0],
			text: "a beautiful landscape with mountains and a lake",
		},
	},
	"2": {
		class_type: "EmptyLatentImage",
		inputs: {
			batch_size: 1,
			height: 512,
			width: 512,
		},
	},
	"3": {
		class_type: "KSampler",
		inputs: {
			cfg: 8,
			denoise: 1,
			latent_image: ["2", 0],
			model: ["4", 0],
			negative: ["5", 0],
			positive: ["1", 0],
			sampler_name: "euler",
			scheduler: "normal",
			seed: 156680208700286,
			steps: 20,
		},
	},
	"4": {
		class_type: "CheckpointLoaderSimple",
		inputs: {
			ckpt_name: "v1-5-pruned-emaonly.ckpt",
		},
	},
	"5": {
		class_type: "CLIPTextEncode",
		inputs: {
			clip: ["4", 0],
			text: "blurry, low quality",
		},
	},
	"6": {
		class_type: "VAEDecode",
		inputs: {
			samples: ["3", 0],
			vae: ["4", 2],
		},
	},
	"7": {
		class_type: "SaveImage",
		inputs: {
			filename_prefix: "ComfyUI",
			images: ["6", 0],
		},
	},
};

async function main() {
	console.log("🚀 Starting ComfyClient Bun example...");

	// Create the Bun WebSocket adapter
	const adapter = new BunWebSocketAdapter();

	// Create the ComfyClient with environment-based configuration
	const client = new ComfyClient({
		adapter,
		auth: {
			jwt: "your-jwt-token-here", // Replace with actual JWT if needed
		},
		logging: {
			level: env.CLIENT_LOG_LEVEL,
			prefix: "[ComfyClient-Example]",
		},
		onConnectionChange: (state) => {
			console.log("🔌 Connection state changed:", state);
		},
		onError: (err) => {
			console.error("❌ Client error:", err);
		},
		onMessage: (msg) => {
			console.log("📨 Received message:", msg);
		},
		reconnect: {
			backoffMultiplier: env.CLIENT_RECONNECT_BACKOFF_MULTIPLIER,
			enabled: env.CLIENT_RECONNECT_ENABLED,
			initialDelay: env.CLIENT_RECONNECT_INITIAL_DELAY,
			maxDelay: env.CLIENT_RECONNECT_MAX_DELAY,
			maxRetries: env.CLIENT_RECONNECT_MAX_RETRIES,
		},
		timeout: {
			connect: env.CLIENT_TIMEOUT_CONNECT,
			message: env.CLIENT_TIMEOUT_MESSAGE,
			operation: env.CLIENT_TIMEOUT_OPERATION,
		},
		url: env.PROXY_URL, // Connect to proxy, not ComfyUI directly
	});

	try {
		// Connect to ComfyUI
		console.log("🔗 Connecting to ComfyUI...");
		const connectResult = await client.connect();

		if (!connectResult.success) {
			console.error("❌ Connection failed:", connectResult.error);
			return;
		}

		console.log("✅ Connected successfully!");

		// Submit a prompt
		console.log("📤 Submitting prompt...");
		const submitResult = await client.submitPrompt(examplePrompt, {
			extraData: {
				example: true,
				timestamp: Date.now(),
			},
			promptId: `example-${Date.now()}`,
		});

		if (!submitResult.success) {
			console.error("❌ Prompt submission failed:", submitResult.error);
			return;
		}

		console.log("✅ Prompt submitted successfully:", submitResult.data);

		// Wait for prompt acceptance
		console.log("⏳ Waiting for prompt acceptance...");
		const acceptResult = await client.waitForEvent("prompt_accepted");

		if (!acceptResult.success) {
			console.error("❌ Failed to get prompt acceptance:", acceptResult.error);
			return;
		}

		console.log("✅ Prompt accepted:", acceptResult.data);

		// Collect all events until completion
		console.log("📊 Collecting all events...");
		const collectResult = await client.collectAllEvents({
			timeout: 120000, // 2 minutes
			waitForCompletion: true,
		});

		if (!collectResult.success) {
			console.error("❌ Failed to collect events:", collectResult.error);
			return;
		}

		const { events, binaryData, completed, error } = collectResult.data;

		console.log(`📈 Collected ${events.length} events`);
		console.log(`🖼️  Received ${binaryData.length} binary data chunks`);
		console.log(`✅ Execution completed: ${completed}`);

		if (error) {
			console.error("❌ Execution error:", error);
		} else {
			console.log("🎉 Execution completed successfully!");
		}

		// Validate event sequence
		const validation = client.validateEventSequence(events);
		console.log("🔍 Event sequence validation:", validation);
	} catch (error) {
		console.error("💥 Unexpected error:", error);
	} finally {
		// Clean up
		console.log("🧹 Disconnecting...");
		client.disconnect();
		console.log("👋 Goodbye!");
	}
}

// Run the example
if (import.meta.main) {
	main().catch(console.error);
}
