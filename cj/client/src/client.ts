import type { WebSocketAdapter } from "./adapters/types";
import { env } from "./env";
import {
	type ComfyClientError,
	ComfyConnectionError,
	ComfyReconnectError,
	ComfyTimeoutError,
	Logger,
} from "./errors";
import type {
	CollectOptions,
	ComfyMessage,
	ComfyPrompt,
	ComfyWsMessage,
	ConnectionState,
	EventCollection,
	LogConfig,
	ReconnectConfig,
	Result,
	SubmitOptions,
	TimeoutConfig,
} from "./types";
import { err, ok } from "./types";

export type ComfyClientConfig = {
	url: string;
	adapter: WebSocketAdapter;
	auth?: { jwt?: string };
	reconnect?: Partial<ReconnectConfig>;
	timeout?: Partial<TimeoutConfig>;
	logging?: Partial<LogConfig>;
	onMessage?: (msg: ComfyMessage) => void;
	onError?: (err: ComfyClientError) => void;
	onConnectionChange?: (state: ConnectionState) => void;
};

type RequiredComfyClientConfig = Required<
	Omit<ComfyClientConfig, "onMessage" | "onError" | "onConnectionChange">
> & {
	onMessage?: (msg: ComfyMessage) => void;
	onError?: (err: ComfyClientError) => void;
	onConnectionChange?: (state: ConnectionState) => void;
};

export class ComfyClient {
	private config: RequiredComfyClientConfig;
	private logger: Logger;
	private connectionState: ConnectionState = "disconnected";
	private reconnectAttempts = 0;
	private reconnectTimeoutId?: NodeJS.Timeout;
	private pendingOperations: Array<() => void> = [];
	private messageHandlers = new Map<string, Array<(data: unknown) => void>>();

	constructor(config: ComfyClientConfig) {
		this.config = {
			auth: config.auth || {},
			logging: {
				level: env.CLIENT_LOG_LEVEL,
				prefix: env.CLIENT_LOG_PREFIX,
				...config.logging,
			},
			onConnectionChange: config.onConnectionChange,
			onError: config.onError,
			onMessage: config.onMessage,
			reconnect: {
				backoffMultiplier: env.CLIENT_RECONNECT_BACKOFF_MULTIPLIER,
				enabled: env.CLIENT_RECONNECT_ENABLED,
				initialDelay: env.CLIENT_RECONNECT_INITIAL_DELAY,
				maxDelay: env.CLIENT_RECONNECT_MAX_DELAY,
				maxRetries: env.CLIENT_RECONNECT_MAX_RETRIES,
				...config.reconnect,
			},
			timeout: {
				connect: env.CLIENT_TIMEOUT_CONNECT,
				message: env.CLIENT_TIMEOUT_MESSAGE,
				operation: env.CLIENT_TIMEOUT_OPERATION,
				...config.timeout,
			},
			...config,
		};

		this.logger = new Logger(
			this.config.logging.level,
			this.config.logging.prefix,
		);
		this.setupAdapterHandlers();
	}

	private setupAdapterHandlers(): void {
		this.config.adapter.onMessage((data) => {
			try {
				let message: ComfyMessage;
				if (typeof data === "string") {
					message = JSON.parse(data) as ComfyMessage;
				} else if (data instanceof ArrayBuffer) {
					message = { data, type: "binary" };
				} else {
					message = data as ComfyMessage;
				}

				this.logger.debug("Received message:", message);

				if (this.config.onMessage) {
					this.config.onMessage(message);
				}

				// Handle typed messages
				if (
					typeof message === "object" &&
					message !== null &&
					"type" in message
				) {
					const typedMessage = message as ComfyWsMessage;
					const handlers = this.messageHandlers.get(typedMessage.type);
					if (handlers) {
						handlers.forEach((handler) => {
							handler(typedMessage.data);
						});
					}
				}
			} catch (error) {
				this.logger.error("Failed to parse message:", error);
			}
		});

		this.config.adapter.onClose((code, reason) => {
			this.logger.warn(`WebSocket closed: ${code} - ${reason}`);
			this.setConnectionState("disconnected");

			if (
				this.config.reconnect?.enabled &&
				this.reconnectAttempts < (this.config.reconnect?.maxRetries ?? 5)
			) {
				this.scheduleReconnect();
			} else if (this.config.reconnect.enabled) {
				this.logger.error("Max reconnection attempts reached");
				if (this.config.onError) {
					this.config.onError(
						new ComfyReconnectError(
							"Max reconnection attempts reached",
							this.reconnectAttempts,
						),
					);
				}
			}
		});

		this.config.adapter.onError((error) => {
			this.logger.error("WebSocket error:", error);
			if (this.config.onError) {
				this.config.onError(error);
			}
		});
	}

	private setConnectionState(state: ConnectionState): void {
		if (this.connectionState !== state) {
			this.connectionState = state;
			this.logger.debug(`Connection state changed to: ${state}`);
			if (this.config.onConnectionChange) {
				this.config.onConnectionChange(state);
			}
		}
	}

	private scheduleReconnect(): void {
		if (this.reconnectTimeoutId) {
			clearTimeout(this.reconnectTimeoutId);
		}

		const delay = Math.min(
			(this.config.reconnect?.initialDelay ?? 1000) *
				(this.config.reconnect?.backoffMultiplier ?? 2) **
					this.reconnectAttempts,
			this.config.reconnect?.maxDelay ?? 30000,
		);

		this.logger.info(
			`Scheduling reconnect attempt ${this.reconnectAttempts + 1} in ${delay}ms`,
		);
		this.setConnectionState("reconnecting");

		this.reconnectTimeoutId = setTimeout(() => {
			this.connect();
		}, delay);
	}

	async connect(): Promise<Result<void>> {
		if (
			this.connectionState === "connected" ||
			this.connectionState === "connecting"
		) {
			return ok(undefined);
		}

		this.setConnectionState("connecting");
		this.logger.info(`Connecting to ${this.config.url}`);

		// For Bun WebSocket, add JWT as query parameter since headers aren't supported
		let url = this.config.url;
		if (this.config.auth?.jwt) {
			const separator = url.includes("?") ? "&" : "?";
			url = `${url}${separator}token=${encodeURIComponent(this.config.auth.jwt)}`;
		}

		const connectionOptions = {
			headers: undefined, // Bun WebSocket doesn't support custom headers
			timeout: this.config.timeout.connect,
		};

		const result = await this.config.adapter.connect(url, connectionOptions);

		if (result.success) {
			this.setConnectionState("connected");
			this.reconnectAttempts = 0;
			this.logger.info("Successfully connected");

			// Execute any pending operations
			const operations = [...this.pendingOperations];
			this.pendingOperations = [];
			operations.forEach((op) => {
				op();
			});

			return ok(undefined);
		}
		this.setConnectionState("disconnected");
		this.reconnectAttempts++;
		this.logger.error("Connection failed:", result.error);
		return result;
	}

	disconnect(): void {
		this.logger.info("Disconnecting...");
		this.config.reconnect.enabled = false;

		if (this.reconnectTimeoutId) {
			clearTimeout(this.reconnectTimeoutId);
			this.reconnectTimeoutId = undefined;
		}

		this.config.adapter.close();
		this.setConnectionState("disconnected");
		this.pendingOperations = [];

		// Clear all message handlers to prevent memory leaks
		this.messageHandlers.clear();
	}

	getConnectionState(): ConnectionState {
		return this.connectionState;
	}

	isConnected(): boolean {
		return this.connectionState === "connected";
	}

	private async waitForEventInternal(
		eventType: string,
		timeout?: number,
	): Promise<Result<unknown>> {
		return new Promise((resolve) => {
			const timeoutMs = timeout ?? this.config.timeout.message;
			let resolved = false;

			const timeoutId = setTimeout(() => {
				if (!resolved) {
					resolved = true;
					this.logger.error(`Timeout waiting for event: ${eventType}`);
					resolve(
						err(
							new ComfyTimeoutError(
								`Timeout waiting for event: ${eventType}`,
								timeoutMs,
							),
						),
					);
				}
			}, timeoutMs);

			const handler = (data: unknown) => {
				if (!resolved) {
					resolved = true;
					clearTimeout(timeoutId);

					// Immediately remove handler to prevent memory leak
					const handlers = this.messageHandlers.get(eventType);
					if (handlers) {
						const index = handlers.indexOf(handler);
						if (index > -1) {
							handlers.splice(index, 1);
						}
					}

					resolve(ok(data));
				}
			};

			// Add handler
			if (!this.messageHandlers.has(eventType)) {
				this.messageHandlers.set(eventType, []);
			}
			this.messageHandlers.get(eventType)!.push(handler);

			// Cleanup function for timeout case
			const cleanup = () => {
				if (!resolved) {
					resolved = true;
					clearTimeout(timeoutId);
					const handlers = this.messageHandlers.get(eventType);
					if (handlers) {
						const index = handlers.indexOf(handler);
						if (index > -1) {
							handlers.splice(index, 1);
						}
					}
				}
			};

			// Auto-cleanup after timeout (backup safety net)
			setTimeout(cleanup, timeoutMs! + 100);
		});
	}

	async waitForEvent(
		eventType: string,
		timeout?: number,
	): Promise<Result<unknown>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		return this.waitForEventInternal(eventType, timeout);
	}

	async collectAllEvents(
		options?: CollectOptions,
	): Promise<Result<EventCollection>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		const timeout = options?.timeout || this.config.timeout.operation;
		const waitForCompletion = options?.waitForCompletion !== false;

		return new Promise((resolve) => {
			const events: unknown[] = [];
			const binaryData: ArrayBuffer[] = [];
			let completed = false;
			let error: string | undefined;

			const timeoutId = setTimeout(() => {
				this.logger.error("Timeout collecting events");
				resolve(ok({ binaryData, completed: false, error: "Timeout", events }));
			}, timeout);

			const messageHandler = (data: string | ArrayBuffer) => {
				if (typeof data === "string") {
					try {
						const message = JSON.parse(data);
						events.push(message);

						if (waitForCompletion) {
							if (message.type === "execution_success") {
								completed = true;
								setTimeout(() => {
									clearTimeout(timeoutId);
									resolve(ok({ binaryData, completed, error, events }));
								}, 1000);
							} else if (
								message.type === "execution_error" ||
								message.type === "error"
							) {
								error = message.data?.message || "Unknown error";
								clearTimeout(timeoutId);
								resolve(ok({ binaryData, completed, error, events }));
							}
						}
					} catch {
						// Ignore non-JSON messages
					}
				} else if (data instanceof ArrayBuffer) {
					binaryData.push(data);
				}
			};

			// Set message handler
			this.config.adapter.onMessage(messageHandler);

			// Cleanup
			setTimeout(() => {
				clearTimeout(timeoutId);
				if (!completed && !error) {
					resolve(ok({ binaryData, completed: false, events }));
				}
			}, timeout);
		});
	}

	async submitPrompt(
		prompt: ComfyPrompt,
		options?: SubmitOptions,
	): Promise<Result<unknown>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		const promptId = options?.promptId || `prompt-${Date.now()}`;

		const message = {
			data: {
				extra_data: options?.extraData || {},
				prompt,
				prompt_id: promptId,
				...(options?.webhookUrl
					? {
							webhook_secret: options.webhookSecret,
							webhook_url: options.webhookUrl,
						}
					: {}),
			},
			type: "submit_prompt",
		};

		this.logger.info(`Submitting prompt: ${promptId}`);
		this.logger.debug(`Message type: ${message.type}`);

		const sendResult = this.config.adapter.send(JSON.stringify(message));
		if (!sendResult.success) {
			this.logger.error(`Failed to send message: ${sendResult.error}`);
			return sendResult;
		}
		this.logger.debug("Message sent, waiting for prompt_accepted event");

		// Wait for prompt acceptance
		const acceptResult = await this.waitForEvent(
			"prompt_accepted",
			this.config.timeout.message,
		);
		if (!acceptResult.success) {
			return err(
				new ComfyConnectionError(
					`Failed to get prompt acceptance: ${acceptResult.error}`,
				),
			);
		}

		this.logger.info(`Prompt accepted: ${promptId}`);
		return ok(acceptResult.data);
	}

	async ping(): Promise<Result<void>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		const message = { type: "ping" };
		const sendResult = this.config.adapter.send(JSON.stringify(message));

		if (!sendResult.success) {
			return sendResult;
		}

		this.logger.debug("Ping sent");
		return ok(undefined);
	}

	// Helper method to validate event sequence (useful for testing)
	validateEventSequence(events: unknown[]): {
		valid: boolean;
		missingEvents: string[];
		extraEvents: string[];
	} {
		const expectedEvents = [
			"prompt_accepted",
			"status",
			"executing",
			"progress_state",
			"executed",
			"execution_success",
		];

		const receivedEventTypes = events.map((e) => (e as { type: string }).type);
		const missingEvents = expectedEvents.filter(
			(eventType) => !receivedEventTypes.includes(eventType),
		);
		const extraEvents = receivedEventTypes.filter(
			(eventType) => !expectedEvents.includes(eventType),
		);

		return {
			extraEvents,
			missingEvents,
			valid: missingEvents.length === 0,
		};
	}
}
