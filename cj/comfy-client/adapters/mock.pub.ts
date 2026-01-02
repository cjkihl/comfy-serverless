import { Logger } from "../errors";
import type { Result } from "../types";
import { err, ok } from "../types";
import type { ConnectionOptions, ConnectionState } from "./base";
import { WebSocketAdapter } from "./base";

/**
 * Mock adapter for testing WebSocket functionality
 * Simulates realistic ComfyUI behavior with full event sequences and mock image data
 */
export class MockWebSocketAdapter extends WebSocketAdapter {
	private connected = false;
	private executionTimers: NodeJS.Timeout[] = [];

	// Mock 1x1 PNG image in base64 (lightweight for testing)
	private readonly MOCK_IMAGE_BASE64 =
		"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

	constructor(url?: string, options?: ConnectionOptions, logger?: Logger) {
		super(
			url || "",
			options,
			logger || new Logger("debug", "[MockWebSocketAdapter]"),
		);
	}

	protected async _connect(): Promise<Result<void>> {
		return new Promise((resolve) => {
			this.logger.debug(`Mock: Connecting to ${this.url}`);
			setTimeout(() => {
				this.connected = true;
				this.logConnectSuccess();
				resolve(ok(undefined));
			}, 10);
		});
	}

	protected _send(data: string | ArrayBuffer): Result<void> {
		if (!this.connected) {
			const error = new Error("Not connected");
			this.logger.error("Mock: Send failed - not connected");
			return err(error);
		}

		// Simulate response for prompt submissions
		if (typeof data === "string") {
			try {
				const message = JSON.parse(data);
				this.logger.debug(
					`Mock: Received message type: ${message.type || "unknown"}`,
				);

				if (
					message.type === "submit_prompt" &&
					this.messageHandler &&
					this.connected
				) {
					this.simulateExecution(message.data.prompt_id);
				} else {
					this.logger.debug("Mock: No handler for this message type");
				}
			} catch (_error) {
				this.logger.debug(`Mock: Non-JSON message: ${data.substring(0, 100)}`);
			}
		}

		return ok(undefined);
	}

	/**
	 * Simulate full ComfyUI event sequence with realistic delays and mock image data
	 */
	private simulateExecution(promptId: string): void {
		if (!this.connected) {
			this.logger.debug("Mock: Skipping execution simulation - not connected");
			return;
		}
		this.logger.debug(`Mock: Simulating execution for prompt: ${promptId}`);

		// 1. prompt_accepted (immediate)
		const timer1 = setTimeout(() => {
			if (this.messageHandler) {
				const response = JSON.stringify({
					data: { prompt_id: promptId },
					type: "prompt_accepted",
				});
				this.logger.info("Mock: Sending prompt_accepted event");
				this.messageHandler(response);
			}
		}, 10);
		this.executionTimers.push(timer1);

		// 2. status event (50ms)
		const timer2 = setTimeout(() => {
			if (this.messageHandler) {
				const response = JSON.stringify({
					data: { execution_start: Date.now() / 1000, running: [] },
					type: "status",
				});
				this.logger.debug("Mock: Sending status event");
				this.messageHandler(response);
			}
		}, 50);
		this.executionTimers.push(timer2);

		// 3. executing start (100ms)
		const timer3 = setTimeout(() => {
			if (this.messageHandler) {
				const response = JSON.stringify({
					data: { node: "2" },
					type: "executing",
				});
				this.logger.debug("Mock: Sending executing event (start)");
				this.messageHandler(response);
			}
		}, 100);
		this.executionTimers.push(timer3);

		// 4. progress_state updates (150ms, 200ms, 250ms)
		const progressValues = [0.25, 0.5, 0.75];
		progressValues.forEach((value, index) => {
			const timer = setTimeout(
				() => {
					if (this.messageHandler) {
						const response = JSON.stringify({
							data: { progress_state: value },
							type: "progress_state",
						});
						this.logger.debug(
							`Mock: Sending progress_state event (${value * 100}%)`,
						);
						this.messageHandler(response);
					}
				},
				150 + index * 50,
			);
			this.executionTimers.push(timer);
		});

		// 5. executed with mock base64 image (300ms)
		const timer5 = setTimeout(() => {
			if (this.messageHandler) {
				const response = JSON.stringify({
					data: {
						node: "12",
						output: {
							result: [
								{
									// Not comfy ui returns image as pure base64, not data url
									image: this.MOCK_IMAGE_BASE64,
								},
							],
						},
					},
					type: "executed",
				});
				this.logger.info("Mock: Sending executed event with mock image");
				this.messageHandler(response);
			}
		}, 300);
		this.executionTimers.push(timer5);

		// 6. executing complete (320ms)
		const timer6 = setTimeout(() => {
			if (this.messageHandler) {
				const response = JSON.stringify({
					data: { node: null },
					type: "executing",
				});
				this.logger.debug("Mock: Sending executing event (complete)");
				this.messageHandler(response);
			}
		}, 320);
		this.executionTimers.push(timer6);

		// 7. execution_success (350ms)
		const timer7 = setTimeout(() => {
			if (this.messageHandler) {
				const response = JSON.stringify({
					type: "execution_success",
				});
				this.logger.info("Mock: Sending execution_success event");
				this.messageHandler(response);
			}
		}, 350);
		this.executionTimers.push(timer7);
	}

	protected _close(): void {
		this.logger.debug("Mock: Closing WebSocket connection");
		// Clear any pending timers
		for (const timer of this.executionTimers) {
			clearTimeout(timer);
		}
		this.executionTimers = [];

		this.connected = false;
	}

	getReadyState(): ConnectionState {
		return this.connected ? "open" : "closed";
	}
}
