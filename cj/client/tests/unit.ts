#!/usr/bin/env bun

/**
 * Unit tests for @cj/comfy-client package
 * Tests individual components without requiring external services
 */

import type { ConnectionState } from "../src/adapters/types";
import {
	BrowserWebSocketAdapter,
	BunWebSocketAdapter,
	ComfyClient,
	type ComfyClientConfig,
	type ComfyPrompt,
	env,
	err,
	ok,
	type Result,
} from "../src/index";
import { createTestPrompt, printTestResults, runTest } from "./utils";

// Helper to create a test JWT token
function createTestJwt(): string {
	// Simple JWT for testing (with NO_VERIFY=true on the proxy)
	// Header: {"alg":"HS256","typ":"JWT"}
	// Payload: {"sub":"test-user-123","iat":1737792000}
	const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
	const payload = btoa(
		JSON.stringify({ iat: 1737792000, sub: "test-user-123" }),
	);
	const signature = "test-signature";
	return `${header}.${payload}.${signature}`;
}

// Mock WebSocket for testing
class MockWebSocket {
	public readyState = 0; // CONNECTING
	public url: string;
	public onopen: (() => void) | null = null;
	public onclose: ((event: { code: number; reason: string }) => void) | null =
		null;
	public onmessage: ((event: { data: string | ArrayBuffer }) => void) | null =
		null;
	public onerror: ((error: Error) => void) | null = null;

	private openHandlers: Array<() => void> = [];
	private messageHandlers: Array<
		(event: { data: string | ArrayBuffer }) => void
	> = [];
	private closeHandlers: Array<
		(event: { code: number; reason: string }) => void
	> = [];
	private errorHandlers: Array<(error: Error) => void> = [];
	private _connected = false;

	constructor(url: string, _options?: unknown) {
		this.url = url;
		// Simulate connection after a short delay
		setTimeout(() => {
			this.readyState = 1; // OPEN
			this._connected = true;
			// Call both legacy handler and addEventListener handlers
			if (this.onopen) this.onopen();
			// Call addEventListener handlers
			this.openHandlers.forEach((handler) => handler());
		}, 10);
	}

	send(data: string | ArrayBuffer): void {
		if (!this._connected) {
			throw new Error("WebSocket is not connected");
		}
		// Mock successful send
		console.log(
			`[MockWebSocket] Sent: ${typeof data === "string" ? data.substring(0, 100) : "binary data"}`,
		);
	}

	close(): void {
		this.readyState = 3; // CLOSED
		this._connected = false;
		const closeEvent = { code: 1000, reason: "Normal closure" };
		if (this.onclose) this.onclose(closeEvent);
		// Call addEventListener handlers
		this.closeHandlers.forEach((handler) => handler(closeEvent));
	}

	addEventListener(event: string, handler: unknown): void {
		if (event === "open") {
			this.openHandlers.push(handler as () => void);
		} else if (event === "message") {
			this.messageHandlers.push(
				handler as (event: { data: string | ArrayBuffer }) => void,
			);
		} else if (event === "close") {
			this.closeHandlers.push(
				handler as (event: { code: number; reason: string }) => void,
			);
		} else if (event === "error") {
			this.errorHandlers.push(handler as (error: Error) => void);
		}
	}

	removeEventListener(_event: string, _handler: unknown): void {
		// Mock implementation
	}

	// Helper methods for testing
	simulateMessage(data: string | ArrayBuffer): void {
		this.messageHandlers.forEach((handler) => {
			handler({ data });
		});
	}

	simulateClose(code = 1000, reason = "Normal closure"): void {
		this.closeHandlers.forEach((handler) => {
			handler({ code, reason });
		});
	}

	simulateError(error: Error): void {
		this.errorHandlers.forEach((handler) => {
			handler(error);
		});
	}
}

// Save original WebSocket before mocking
const OriginalWebSocket = global.WebSocket;

// Mock Bun WebSocket globally
// Type assertion needed because global is not properly typed for WebSocket assignment
// We use 'as any' here because we're intentionally replacing the global WebSocket for testing
(global as any).WebSocket = MockWebSocket;

// Mock adapter for testing
class MockWebSocketAdapter {
	private connected = false;
	private closeHandler?: (code: number, reason: string) => void;
	private messageHandler?: (data: string | ArrayBuffer) => void;

	async connect(_url: string, _options: unknown): Promise<Result<void>> {
		return new Promise((resolve) => {
			setTimeout(() => {
				this.connected = true;
				resolve(ok(undefined));
			}, 10);
		});
	}

	send(data: string | ArrayBuffer): Result<void> {
		if (!this.connected) {
			return err(new Error("Not connected"));
		}

		// Simulate response for prompt submissions
		if (typeof data === "string") {
			try {
				const message = JSON.parse(data);
				if (message.type === "submit_prompt" && this.messageHandler) {
					// Simulate prompt_accepted event
					setTimeout(() => {
						const response = JSON.stringify({
							data: { prompt_id: message.data.prompt_id },
							type: "prompt_accepted",
						});
						this.messageHandler!(response);
					}, 10);
				}
			} catch {
				// Not JSON, ignore
			}
		}

		return ok(undefined);
	}

	close(): void {
		this.connected = false;
		if (this.closeHandler) {
			this.closeHandler(1000, "Normal closure");
		}
	}

	onMessage(handler: (data: string | ArrayBuffer) => void): void {
		this.messageHandler = handler;
	}

	onClose(handler: (code: number, reason: string) => void): void {
		this.closeHandler = handler;
	}

	onError(_handler: (error: Error) => void): void {
		// Mock implementation - handler stored but not used in tests
	}

	getReadyState(): ConnectionState {
		return this.connected ? "open" : "closed";
	}

	removeAllListeners(): void {
		this.closeHandler = undefined;
		this.messageHandler = undefined;
	}
}

async function testComfyClientCreation(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const config: ComfyClientConfig = {
		adapter,
		auth: { jwt: "test-jwt" },
		logging: { level: "debug" },
		url: "ws://localhost:8188/ws",
	};

	const client = new ComfyClient(config);

	if (!client) {
		throw new Error("Failed to create ComfyClient");
	}

	if (client.getConnectionState() !== "disconnected") {
		throw new Error("Client should start in disconnected state");
	}

	if (client.isConnected()) {
		throw new Error("Client should not be connected initially");
	}
}

async function testComfyClientConnection(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	const connectResult = await client.connect();

	if (!connectResult.success) {
		throw new Error(`Connection failed: ${connectResult.error}`);
	}

	if (!client.isConnected()) {
		throw new Error("Client should be connected after successful connection");
	}

	if (client.getConnectionState() !== "connected") {
		throw new Error("Connection state should be connected");
	}

	client.disconnect();
}

async function testComfyClientDisconnection(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	await client.connect();
	client.disconnect();

	if (client.isConnected()) {
		throw new Error("Client should not be connected after disconnection");
	}

	if (client.getConnectionState() !== "disconnected") {
		throw new Error("Connection state should be disconnected");
	}
}

async function testComfyClientSubmitPrompt(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	await client.connect();

	const prompt: ComfyPrompt = createTestPrompt();
	const submitResult = await client.submitPrompt(prompt, {
		promptId: "test-prompt-123",
	});

	if (!submitResult.success) {
		throw new Error(`Prompt submission failed: ${submitResult.error}`);
	}

	client.disconnect();
}

async function testComfyClientWaitForEvent(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	await client.connect();

	// Test timeout
	const timeoutResult = await client.waitForEvent("nonexistent_event", 100);
	if (timeoutResult.success) {
		throw new Error("Should have timed out waiting for nonexistent event");
	}

	client.disconnect();
}

async function testComfyClientCollectEvents(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	await client.connect();

	const collectResult = await client.collectAllEvents({
		timeout: 1000,
		waitForCompletion: false,
	});

	if (!collectResult.success) {
		throw new Error(`Event collection failed: ${collectResult.error}`);
	}

	client.disconnect();
}

async function testComfyClientPing(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	await client.connect();

	const pingResult = await client.ping();
	if (!pingResult.success) {
		throw new Error(`Ping failed: ${pingResult.error}`);
	}

	client.disconnect();
}

async function testComfyClientValidation(): Promise<void> {
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	const events = [
		{ type: "prompt_accepted" },
		{ type: "status" },
		{ type: "executing" },
		{ type: "progress_state" },
		{ type: "executed" },
		{ type: "execution_success" },
	];

	const validation = client.validateEventSequence(events);
	if (!validation.valid) {
		throw new Error(
			`Event sequence validation failed: missing ${validation.missingEvents.join(", ")}`,
		);
	}

	// Test with missing events
	const incompleteEvents = [{ type: "prompt_accepted" }, { type: "status" }];

	const incompleteValidation = client.validateEventSequence(incompleteEvents);
	if (incompleteValidation.valid) {
		throw new Error(
			"Should have failed validation for incomplete event sequence",
		);
	}
}

async function testBunWebSocketAdapter(): Promise<void> {
	// Restore real WebSocket for this test
	(global as any).WebSocket = OriginalWebSocket;

	const adapter = new BunWebSocketAdapter();

	// Use the proxy URL from environment instead of hardcoded port
	let proxyUrl = env.PROXY_URL || "ws://localhost:8190/ws";

	// Create a test JWT token (simple JWT for testing with NO_VERIFY=true)
	const testJwt = createTestJwt();

	// Add JWT as query parameter since Bun WebSocket doesn't support custom headers
	const separator = proxyUrl.includes("?") ? "&" : "?";
	proxyUrl = `${proxyUrl}${separator}token=${encodeURIComponent(testJwt)}`;

	const connectResult = await adapter.connect(proxyUrl, {
		timeout: 5000,
	});

	if (!connectResult.success) {
		throw new Error(`Bun adapter connection failed: ${connectResult.error}`);
	}

	if (adapter.getReadyState() !== "open") {
		throw new Error("Adapter should be in open state after connection");
	}

	const sendResult = adapter.send("test message");
	if (!sendResult.success) {
		throw new Error(`Send failed: ${sendResult.error}`);
	}

	adapter.close();

	// Restore mock WebSocket for other tests
	(global as any).WebSocket = MockWebSocket;
}

async function testBrowserWebSocketAdapter(): Promise<void> {
	// Restore real WebSocket for this test
	(global as any).WebSocket = OriginalWebSocket;

	const adapter = new BrowserWebSocketAdapter();

	// Use the proxy URL from environment instead of hardcoded port
	const proxyUrl = env.PROXY_URL || "ws://localhost:8190/ws";

	// Create a test JWT token (simple JWT for testing with NO_VERIFY=true)
	const testJwt = createTestJwt();

	const connectResult = await adapter.connect(proxyUrl, {
		headers: {
			Authorization: `Bearer ${testJwt}`,
		},
		timeout: 5000,
	});

	if (!connectResult.success) {
		throw new Error(
			`Browser adapter connection failed: ${connectResult.error}`,
		);
	}

	if (adapter.getReadyState() !== "open") {
		throw new Error("Adapter should be in open state after connection");
	}

	const sendResult = adapter.send("test message");
	if (!sendResult.success) {
		throw new Error(`Send failed: ${sendResult.error}`);
	}

	adapter.close();

	// Restore mock WebSocket for other tests
	(global as any).WebSocket = MockWebSocket;
}

async function testErrorHandling(): Promise<void> {
	// Test operations on disconnected client with mock adapter
	const mockAdapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter: mockAdapter,
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	});

	// Test operations on disconnected client
	const submitResult = await client.submitPrompt(createTestPrompt());
	if (submitResult.success) {
		throw new Error(
			"Should have failed to submit prompt on disconnected client",
		);
	}

	// Test connection to invalid URL with real adapter
	// Restore real WebSocket for this test
	(global as any).WebSocket = OriginalWebSocket;

	const realAdapter = new BunWebSocketAdapter();
	const clientWithRealAdapter = new ComfyClient({
		adapter: realAdapter,
		logging: { level: "silent" },
		url: "ws://invalid-url-that-does-not-exist:12345/ws",
	});

	const connectResult = await clientWithRealAdapter.connect();
	if (connectResult.success) {
		throw new Error("Should have failed to connect to invalid URL");
	}

	// Restore mock WebSocket for other tests
	(global as any).WebSocket = MockWebSocket;
}

async function testConfigurationValidation(): Promise<void> {
	// Test with minimal config
	const adapter = new MockWebSocketAdapter();
	const client = new ComfyClient({
		adapter,
		url: "ws://localhost:8188/ws",
	});

	if (!client) {
		throw new Error("Failed to create client with minimal config");
	}

	// Test with full config
	const fullConfigClient = new ComfyClient({
		adapter,
		auth: { jwt: "test-jwt" },
		logging: {
			level: "info",
			prefix: "[TestClient]",
		},
		onConnectionChange: (state) => console.log("State:", state),
		onError: (err) => console.error("Error:", err),
		onMessage: (msg) => console.log("Message:", msg),
		reconnect: {
			backoffMultiplier: 2,
			enabled: true,
			initialDelay: 1000,
			maxDelay: 30000,
			maxRetries: 5,
		},
		timeout: {
			connect: 10000,
			message: 30000,
			operation: 120000,
		},
		url: "ws://localhost:8188/ws",
	});

	if (!fullConfigClient) {
		throw new Error("Failed to create client with full config");
	}
}

async function main() {
	console.log("🧪 Running ComfyClient Unit Tests...\n");

	const tests = [
		runTest("ComfyClient Creation", testComfyClientCreation),
		runTest("ComfyClient Connection", testComfyClientConnection),
		runTest("ComfyClient Disconnection", testComfyClientDisconnection),
		runTest("ComfyClient Submit Prompt", testComfyClientSubmitPrompt),
		runTest("ComfyClient Wait For Event", testComfyClientWaitForEvent),
		runTest("ComfyClient Collect Events", testComfyClientCollectEvents),
		runTest("ComfyClient Ping", testComfyClientPing),
		runTest("ComfyClient Validation", testComfyClientValidation),
		runTest("Bun WebSocket Adapter", testBunWebSocketAdapter),
		runTest("Browser WebSocket Adapter", testBrowserWebSocketAdapter),
		runTest("Error Handling", testErrorHandling),
		runTest("Configuration Validation", testConfigurationValidation),
	];

	const results = await Promise.all(tests);
	printTestResults(results);
}

if (import.meta.main) {
	main().catch(console.error);
}
