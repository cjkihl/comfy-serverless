#!/usr/bin/env bun

/**
 * Unit tests for @cj/comfy-client using MockWebSocketAdapter
 * All real adapter tests are in @web/tests/
 */

import { expect, test } from "bun:test";
import { loadEnv } from "@cjkihl/with-env";
import { MockWebSocketAdapter } from "../adapters/mock.pub";
import { ComfyClient, type ComfyClientConfig } from "../index.pub";
import { createTestClient, createTestPrompt } from "./fixtures";
import {
	PerformanceTracker,
	printPerformanceMetrics,
} from "./performance-tracker";

await loadEnv();

test("ComfyClient Creation", () => {
	const adapter = new MockWebSocketAdapter();
	const config: ComfyClientConfig = {
		adapter,
		auth: { jwt: "test-jwt" },
		autoConnect: false,
		heartbeat: { enabled: false }, // Disable heartbeat for tests
		logging: { level: "debug" },
		url: "ws://localhost:8188/ws",
	};

	const client = new ComfyClient(config);

	expect(client).toBeDefined();
	expect(client.getConnectionState()).toBe("disconnected");
	expect(client.isConnected()).toBe(false);

	client.disconnect();
});

test("ComfyClient Connection", async () => {
	const tracker = new PerformanceTracker();
	const client = createTestClient();

	tracker.startPhase("Connection Open");
	const connectResult = await client.connect();
	tracker.endPhase("Connection Open");

	expect(connectResult.success).toBe(true);
	expect(client.isConnected()).toBe(true);
	expect(client.getConnectionState()).toBe("connected");

	client.disconnect();

	const metrics = tracker.getMetrics();
	printPerformanceMetrics(metrics);
});

test("ComfyClient Disconnection", async () => {
	const client = createTestClient();

	await client.connect();
	client.disconnect();

	expect(client.isConnected()).toBe(false);
	expect(client.getConnectionState()).toBe("disconnected");
});

test("ComfyClient Submit Prompt", async () => {
	const tracker = new PerformanceTracker();
	const client = createTestClient();

	tracker.startPhase("Connection Open");
	await client.connect();
	tracker.endPhase("Connection Open");

	const prompt = createTestPrompt();
	tracker.startPhase("Prompt Submission");
	const submitResult = await client.submitPrompt(prompt, {
		promptId: "test-prompt-123",
	});
	tracker.endPhase("Prompt Submission");

	expect(submitResult.success).toBe(true);

	client.disconnect();

	const metrics = tracker.getMetrics();
	printPerformanceMetrics(metrics);
});

test("ComfyClient Wait For Event", async () => {
	const client = createTestClient();

	const timeoutResult = await client.waitForEvent("nonexistent_event", 100);
	expect(timeoutResult.success).toBe(false);

	client.disconnect();
});

test("ComfyClient Collect Events", async () => {
	const client = createTestClient();

	const collectResult = await client.collectAllEvents({
		timeout: 1000,
		waitForCompletion: false,
	});

	expect(collectResult.success).toBe(false);
	if (!collectResult.success) {
		expect(collectResult.error.message).toContain("Not connected");
	}

	client.disconnect();
});

test("ComfyClient Ping", async () => {
	const client = createTestClient();

	const pingResult = await client.ping();

	expect(pingResult.success).toBe(false);
	if (!pingResult.success) {
		expect(pingResult.error.message).toContain("Not connected");
	}

	client.disconnect();
});

test("ComfyClient Validation", () => {
	const client = createTestClient();

	const events = [
		{ type: "prompt_accepted" },
		{ type: "status" },
		{ type: "executing" },
		{ type: "progress_state" },
		{ type: "executed" },
		{ type: "execution_success" },
	];

	const validation = client.validateEventSequence(events);
	expect(validation.valid).toBe(true);

	const incompleteEvents = [{ type: "prompt_accepted" }, { type: "status" }];
	const incompleteValidation = client.validateEventSequence(incompleteEvents);
	expect(incompleteValidation.valid).toBe(false);

	client.disconnect();
});

test("Error Handling", async () => {
	const mockAdapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter: mockAdapter,
		auth: { jwt: "test-jwt" },
		heartbeat: { enabled: false }, // Disable heartbeat for tests
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	const submitResult = await client.submitPrompt(createTestPrompt());
	expect(submitResult.success).toBe(false);

	client.disconnect();
});

test("Configuration Validation", () => {
	const client = createTestClient({ auth: { jwt: "test-jwt" } });
	expect(client).toBeDefined();

	const fullConfigClient = createTestClient({
		logging: { level: "info", prefix: "[TestClient]" },
		onConnectionChange: () => {},
		onError: () => {},
		onMessage: () => {},
		reconnect: {
			backoffMultiplier: 2,
			enabled: true,
			initialDelay: 1000,
			maxDelay: 30000,
			maxRetries: 5,
		},
		timeout: { connect: 10000, message: 30000, operation: 120000 },
	});

	expect(fullConfigClient).toBeDefined();

	client.disconnect();
	fullConfigClient.disconnect();
});
