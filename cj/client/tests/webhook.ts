#!/usr/bin/env bun

/**
 * Webhook functionality tests for @cj/comfy-client package
 * Tests webhook registration, delivery, and cleanup
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

async function testWebhookRegistration(): Promise<void> {
	console.log("🔍 Testing webhook registration...");

	const userId = generateUniqueUserId("webhook-reg");
	const jwt = generateTestJWT(userId);

	const adapter = new BunWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		auth: { jwt },
		logging: { level: "info" },
		url: PROXY_URL,
	});

	try {
		const connectResult = await client.connect();
		if (!connectResult.success) {
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		// Submit prompt with webhook
		const prompt = createTestPrompt();
		const webhookUrl = "https://httpbin.org/post"; // Public test endpoint
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `webhook-test-${Date.now()}`,
			webhookSecret: "test-secret-123",
			webhookUrl,
		});

		if (!submitResult.success) {
			throw new Error(`Prompt submission failed: ${submitResult.error}`);
		}

		console.log("✅ Webhook registered successfully");
	} finally {
		client.disconnect();
	}
}

async function testInvalidWebhookUrl(): Promise<void> {
	console.log("🔍 Testing invalid webhook URL rejection...");

	const userId = generateUniqueUserId("webhook-invalid");
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
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		// Try to submit with invalid webhook URL
		const prompt = createTestPrompt();
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `webhook-invalid-${Date.now()}`,
			webhookUrl: "not-a-valid-url",
		});

		if (submitResult.success) {
			throw new Error("Invalid webhook URL should have been rejected");
		}

		console.log("✅ Invalid webhook URL properly rejected");
	} finally {
		client.disconnect();
	}
}

async function testNonHttpWebhookUrl(): Promise<void> {
	console.log("🔍 Testing non-HTTP webhook URL rejection...");

	const userId = generateUniqueUserId("webhook-protocol");
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
			throw new Error(`Connection failed: ${connectResult.error}`);
		}

		// Try to submit with non-HTTP webhook URL
		const prompt = createTestPrompt();
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `webhook-protocol-${Date.now()}`,
			webhookUrl: "file:///etc/passwd", // Attempt to use file protocol
		});

		if (submitResult.success) {
			throw new Error("Non-HTTP webhook URL should have been rejected");
		}

		console.log("✅ Non-HTTP webhook URL properly rejected");
	} finally {
		client.disconnect();
	}
}

async function main() {
	console.log("🧪 Running Webhook Tests...\n");

	const tests = [
		runTest("Webhook Registration", testWebhookRegistration),
		runTest("Invalid Webhook URL", testInvalidWebhookUrl),
		runTest("Non-HTTP Webhook URL", testNonHttpWebhookUrl),
	];

	const results = await Promise.all(tests);
	printTestResults(results);
}

if (import.meta.main) {
	main().catch(console.error);
}
