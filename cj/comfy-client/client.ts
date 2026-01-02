/// <reference lib="dom" />
import type { WebSocketAdapter } from "./adapters/base";
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
	HeartbeatConfig,
	LogConfig,
	ReconnectConfig,
	Result,
	SubmitOptions,
	TimeoutConfig,
} from "./types";
import { err, ok } from "./types";

// Note: Using standard DOM types from lib.dom.d.ts

// Runtime environment check to safely gate browser-only features
const isBrowser = (): boolean =>
	typeof window !== "undefined" && typeof document !== "undefined";

/**
 * Configuration for ComfyClient
 *
 * @example
 * ```typescript
 * const client = new ComfyClient({
 *   url: 'ws://localhost:8190/ws',
 *   adapter: new UniversalWebSocketAdapter(),
 *   auth: { jwt: 'your-jwt-token' },
 *   autoConnect: true,
 *   logging: { level: 'info' }
 * });
 * ```
 */
export type ComfyClientConfig = {
	/** WebSocket URL to connect to (e.g., 'ws://localhost:8190/ws') */
	url: string;
	/** WebSocket adapter implementation (UniversalWebSocketAdapter for Bun/Browser) */
	adapter: WebSocketAdapter;
	/** Authentication configuration */
	auth: { jwt: string };
	/** Automatically connect on client creation (default: true) */
	autoConnect?: boolean;
	/** Heartbeat/ping configuration */
	heartbeat?: Partial<HeartbeatConfig>;
	/** Reconnection configuration */
	reconnect?: Partial<ReconnectConfig>;
	/** Timeout configuration for various operations */
	timeout?: Partial<TimeoutConfig>;
	/** Logging configuration */
	logging?: Partial<LogConfig>;
	/** Callback for received messages */
	onMessage?: (msg: ComfyMessage) => void;
	/** Callback for errors */
	onError?: (err: ComfyClientError) => void;
	/** Callback for connection state changes */
	onConnectionChange?: (state: ConnectionState) => void;
};

type RequiredComfyClientConfig = Required<
	Omit<
		ComfyClientConfig,
		"onMessage" | "onError" | "onConnectionChange" | "autoConnect" | "heartbeat"
	>
> & {
	autoConnect?: boolean;
	heartbeat?: Partial<HeartbeatConfig>;
	onMessage?: (msg: ComfyMessage) => void;
	onError?: (err: ComfyClientError) => void;
	onConnectionChange?: (state: ConnectionState) => void;
};

/**
 * ComfyClient - Universal WebSocket client for ComfyUI
 *
 * Provides a type-safe, production-ready client for interacting with ComfyUI via WebSocket.
 * Supports both Bun and Browser environments through adapter pattern.
 *
 * @example
 * ```typescript
 * import { ComfyClient } from '@cj/comfy-client';
 * import { UniversalWebSocketAdapter } from '@cj/comfy-client/adapters/universal';
 *
 * const adapter = new UniversalWebSocketAdapter();
 * const client = new ComfyClient({
 *   url: 'ws://localhost:8190/ws',
 *   adapter,
 *   auth: { jwt: 'your-jwt-token' },
 *   autoConnect: true
 * });
 *
 * const result = await client.submitPrompt({
 *   "1": {
 *     "inputs": { "text": "a beautiful landscape" },
 *     "class_type": "CLIPTextEncode"
 *   }
 * });
 * ```
 */
export class ComfyClient {
	private config: RequiredComfyClientConfig;
	private logger: Logger;
	private connectionState: ConnectionState = "disconnected";
	private reconnectAttempts = 0;
	private reconnectTimeoutId?: NodeJS.Timeout;
	private reconnectGiveUp = false;
	private isReconnecting = false;
	/**
	 * Tracks whether the user explicitly called disconnect().
	 * This flag prevents automatic reconnection after user-initiated disconnections.
	 *
	 * We use a dedicated flag instead of mutating config.reconnect.enabled because:
	 * - Config objects should remain immutable to avoid side effects
	 * - Once config.reconnect.enabled is set to false, it can't be easily recovered
	 * - A manual connect() call should override this flag (reset it)
	 *
	 * When true: No automatic reconnection should occur
	 * When false: Allow reconnection if enabled in config (default behavior)
	 */
	private explicitDisconnect = false;
	private pendingOperations: Array<() => void> = [];
	private messageHandlers = new Map<string, Array<(data: unknown) => void>>();
	private heartbeatIntervalId?: NodeJS.Timeout;
	private visibilityHandler?: () => void;

	/**
	 * Creates a new ComfyClient instance
	 *
	 * @param config - Client configuration
	 * @throws {Error} If configuration is invalid
	 */
	constructor(config: ComfyClientConfig) {
		this.config = {
			autoConnect: config.autoConnect !== undefined ? config.autoConnect : true,
			heartbeat: {
				enabled: true,
				interval: 30000,
				...config.heartbeat,
			},
			logging: {
				level: "info",
				prefix: "[ComfyClient]",
				...config.logging,
			},
			onConnectionChange: config.onConnectionChange,
			onError: config.onError,
			onMessage: config.onMessage,
			reconnect: {
				backoffMultiplier: 2,
				enabled: true,
				initialDelay: 1000,
				maxDelay: 30000,
				maxRetries: 5,
				...config.reconnect,
			},
			timeout: {
				connect: 10000,
				message: 30000,
				operation: 120000,
				...config.timeout,
			},
			...config,
		};

		this.logger = new Logger(
			this.config.logging.level,
			this.config.logging.prefix,
		);
		this.setupAdapterHandlers();
		this.setupVisibilityTracking();

		// Auto-connect if enabled
		if (this.config.autoConnect) {
			this.connect().catch((err) => {
				this.logger.error("Auto-connect failed:", err);
			});
		}

		// Start heartbeat if enabled
		this.startHeartbeat();
	}

	private setupAdapterHandlers(): void {
		// Initialize adapter with base URL and options
		this.config.adapter.setUrl(this.config.url);
		this.config.adapter.setOptions({ timeout: this.config.timeout.connect });

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
					this.logger.debug(
						`📥 Message handler: type=${typedMessage.type}, has handlers=${this.messageHandlers.has(typedMessage.type)}`,
					);
					const handlers = this.messageHandlers.get(typedMessage.type);
					if (handlers) {
						this.logger.debug(
							`✅ Found ${handlers.length} handler(s) for type ${typedMessage.type}, calling them...`,
						);
						handlers.forEach((handler) => {
							handler(typedMessage.data);
						});
					} else {
						this.logger.debug(
							`⚠️ No handlers registered for message type ${typedMessage.type}`,
						);
					}
				}
			} catch (error) {
				this.logger.error("Failed to parse message:", error);
			}
		});

		this.config.adapter.onClose((code, reason) => {
			this.logger.warn(`WebSocket closed: ${code} - ${reason}`);
			this.logger.debug(
				`[RECONNECT DEBUG] onClose - reconnectAttempts: ${this.reconnectAttempts}, isReconnecting: ${this.isReconnecting}, reconnectGiveUp: ${this.reconnectGiveUp}, maxRetries: ${this.config.reconnect?.maxRetries ?? 5}`,
			);
			this.setConnectionState("disconnected");

			// Only attempt reconnection if enabled, not already reconnecting, haven't given up, and within retry limit
			// CRITICAL: Also check !this.explicitDisconnect to prevent reconnection after user called disconnect()
			if (
				this.config.reconnect?.enabled &&
				!this.isReconnecting &&
				!this.reconnectGiveUp &&
				!this.explicitDisconnect &&
				this.reconnectAttempts < (this.config.reconnect?.maxRetries ?? 5)
			) {
				this.logger.debug(
					"[RECONNECT DEBUG] Conditions met for reconnection, scheduling...",
				);
				this.scheduleReconnect();
			} else if (
				this.config.reconnect.enabled &&
				!this.explicitDisconnect &&
				this.reconnectAttempts >= (this.config.reconnect?.maxRetries ?? 5)
			) {
				// Max retries reached - give up and notify
				this.reconnectGiveUp = true;
				this.isReconnecting = false;
				this.logger.error("Max reconnection attempts reached");
				this.logger.debug("[RECONNECT DEBUG] Max retries reached, giving up");
				if (this.config.onError) {
					this.config.onError(
						new ComfyReconnectError(
							"Max reconnection attempts reached",
							this.reconnectAttempts,
						),
					);
				}
			} else {
				this.logger.debug(
					`[RECONNECT DEBUG] Not reconnecting - enabled: ${this.config.reconnect?.enabled}, isReconnecting: ${this.isReconnecting}, reconnectGiveUp: ${this.reconnectGiveUp}, explicitDisconnect: ${this.explicitDisconnect}, attempts: ${this.reconnectAttempts}/${this.config.reconnect?.maxRetries ?? 5}`,
				);
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
		this.logger.debug(
			`[RECONNECT DEBUG] scheduleReconnect called - reconnectGiveUp: ${this.reconnectGiveUp}, isReconnecting: ${this.isReconnecting}, explicitDisconnect: ${this.explicitDisconnect}, attempts: ${this.reconnectAttempts}`,
		);

		// Don't schedule if we've given up, already reconnecting, or user explicitly disconnected
		if (
			this.reconnectGiveUp ||
			this.isReconnecting ||
			this.explicitDisconnect
		) {
			this.logger.debug(
				`[RECONNECT DEBUG] Skipping scheduleReconnect - reconnectGiveUp: ${this.reconnectGiveUp}, isReconnecting: ${this.isReconnecting}, explicitDisconnect: ${this.explicitDisconnect}`,
			);
			return;
		}

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
		this.logger.debug("[RECONNECT DEBUG] Setting isReconnecting = true");
		this.setConnectionState("reconnecting");
		this.isReconnecting = true;

		this.reconnectTimeoutId = setTimeout(() => {
			this.logger.debug(
				"[RECONNECT DEBUG] Timeout fired, incrementing attempt counter and calling connect()",
			);
			// Increment attempts BEFORE trying to connect, so if connection succeeds but immediately fails,
			// we don't lose track of the attempt
			this.reconnectAttempts++;
			this.logger.debug(
				`[RECONNECT DEBUG] Incremented reconnectAttempts to: ${this.reconnectAttempts}`,
			);

			this.connect().finally(() => {
				this.logger.debug(
					"[RECONNECT DEBUG] connect() completed, setting isReconnecting = false",
				);
				this.isReconnecting = false;
			});
		}, delay);
	}

	/**
	 * Establishes WebSocket connection to the server
	 *
	 * If autoConnect is enabled, this is called automatically on construction.
	 * Can be called manually to reconnect after disconnect().
	 *
	 * @returns Result indicating success or failure
	 * @example
	 * ```typescript
	 * const result = await client.connect();
	 * if (!result.success) {
	 *   console.error('Connection failed:', result.error);
	 * }
	 * ```
	 */
	async connect(): Promise<Result<void>> {
		// If user explicitly disconnected, calling connect() should override that
		// and reset the explicitDisconnect flag to allow reconnection in the future
		if (this.explicitDisconnect) {
			this.logger.debug(
				"[DISCONNECT DEBUG] connect() called after explicit disconnect, resetting explicitDisconnect flag",
			);
			this.explicitDisconnect = false;
		}

		if (
			this.connectionState === "connected" ||
			this.connectionState === "connecting"
		) {
			return ok(undefined);
		}

		const connectStartTime = performance.now();
		this.setConnectionState("connecting");
		this.logger.info(`Connecting to ${this.config.url}`);

		// Add JWT as query parameter (WebSockets don't support custom headers)
		let url = this.config.url;
		if (this.config.auth.jwt) {
			const separator = url.includes("?") ? "&" : "?";
			url = `${url}${separator}token=${encodeURIComponent(this.config.auth.jwt)}`;
		}

		const connectionOptions: { timeout?: number } = {
			timeout: this.config.timeout.connect,
		};

		// Update adapter with full URL and options before connecting
		this.config.adapter.setUrl(url);
		this.config.adapter.setOptions(connectionOptions);

		const result = await this.config.adapter.connect();
		const connectDuration = performance.now() - connectStartTime;

		if (result.success) {
			this.logger.info("Successfully connected", {
				durationMs: Math.round(connectDuration),
				url: this.config.url,
			});
			this.logger.debug(
				`[RECONNECT DEBUG] Connection succeeded! Before state change - reconnectAttempts: ${this.reconnectAttempts}, isReconnecting: ${this.isReconnecting}`,
			);
			this.setConnectionState("connected");

			// Only reset reconnection state if this was NOT part of a reconnection flow
			// This prevents resetting the counter if the server immediately closes the connection
			if (!this.isReconnecting) {
				// Manual connection - reset everything
				this.reconnectAttempts = 0;
				this.reconnectGiveUp = false;
				this.logger.debug(
					"[RECONNECT DEBUG] Manual connection - reset reconnection state",
				);
			} else {
				// This was a reconnection attempt that succeeded - don't reset counter yet
				// We'll reset it only after the connection stays stable for a reasonable time
				this.reconnectGiveUp = false; // Allow future reconnections since this one worked initially
				this.logger.debug(
					`[RECONNECT DEBUG] Reconnection succeeded - keeping attempt counter at: ${this.reconnectAttempts} for now`,
				);

				// Schedule a delayed reset of reconnection attempts after connection is stable
				// Use a shorter delay to avoid too much lag
				setTimeout(() => {
					if (this.connectionState === "connected") {
						this.logger.debug(
							"[RECONNECT DEBUG] Connection stable for 2s, resetting reconnection state",
						);
						this.reconnectAttempts = 0;
					}
				}, 2000); // Reset after 2 seconds of stable connection
			}

			this.isReconnecting = false; // Always clear this flag
			this.logger.debug(
				`[RECONNECT DEBUG] After state change - reconnectAttempts: ${this.reconnectAttempts}, isReconnecting: ${this.isReconnecting}`,
			);

			// Execute any pending operations
			const operations = [...this.pendingOperations];
			this.pendingOperations = [];
			operations.forEach((op) => {
				op();
			});

			return ok(undefined);
		}

		this.setConnectionState("disconnected");
		// Note: reconnectAttempts is now incremented in scheduleReconnect BEFORE connection attempt
		this.logger.debug(
			`[RECONNECT DEBUG] Connection failed, attempts currently at: ${this.reconnectAttempts}`,
		);
		this.logger.error("Connection failed", {
			durationMs: Math.round(connectDuration),
			error: result.error,
			reconnectAttempts: this.reconnectAttempts,
			url: this.config.url,
		});
		return result;
	}

	/**
	 * Closes WebSocket connection and stops automatic reconnection
	 *
	 * After calling disconnect(), the client will not automatically reconnect.
	 * Call connect() manually to reconnect.
	 *
	 * @example
	 * ```typescript
	 * client.disconnect();
	 * // Later...
	 * await client.connect();
	 * ```
	 */
	disconnect(): void {
		this.logger.info("Disconnecting...");

		// IMPORTANT: Use explicitDisconnect flag instead of mutating config.reconnect.enabled
		// This preserves config immutability and allows proper state management.
		// When set to true, prevents automatic reconnection attempts after this disconnect.
		this.logger.debug(
			"[DISCONNECT DEBUG] Setting explicitDisconnect=true to prevent automatic reconnection",
		);
		this.explicitDisconnect = true;

		if (this.reconnectTimeoutId) {
			clearTimeout(this.reconnectTimeoutId);
			this.reconnectTimeoutId = undefined;
		}

		// Reset reconnection state flags to allow fresh connection attempts
		this.reconnectAttempts = 0;
		this.reconnectGiveUp = false;
		this.isReconnecting = false;

		// Clear heartbeat interval
		if (this.heartbeatIntervalId) {
			clearInterval(this.heartbeatIntervalId);
			this.heartbeatIntervalId = undefined;
		}

		// Remove visibility tracking
		if (this.visibilityHandler && isBrowser()) {
			document.removeEventListener("visibilitychange", this.visibilityHandler);
			this.visibilityHandler = undefined;
		}

		this.config.adapter.close();
		this.setConnectionState("disconnected");
		this.pendingOperations = [];

		// Clear all message handlers to prevent memory leaks
		this.messageHandlers.clear();
	}

	/**
	 * Gets the current connection state
	 *
	 * @returns Current connection state: "disconnected" | "connecting" | "connected" | "reconnecting"
	 */
	getConnectionState(): ConnectionState {
		return this.connectionState;
	}

	/**
	 * Checks if the client is currently connected
	 *
	 * @returns true if connected, false otherwise
	 */
	isConnected(): boolean {
		return this.connectionState === "connected";
	}

	/**
	 * Gets the logger instance for advanced logging operations
	 *
	 * Allows access to logger features like correlation IDs and runtime log level changes.
	 *
	 * @returns The logger instance
	 * @example
	 * ```typescript
	 * const logger = client.getLogger();
	 * logger.setCorrelationId("request-123");
	 * logger.setLevel("debug");
	 * ```
	 */
	getLogger(): Logger {
		return this.logger;
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

	/**
	 * Waits for a specific event type to be received
	 *
	 * @param eventType - The event type to wait for (e.g., "prompt_accepted", "executed")
	 * @param timeout - Optional timeout in milliseconds (defaults to config.timeout.message)
	 * @returns Result containing the event data, or error if timeout
	 * @example
	 * ```typescript
	 * const result = await client.waitForEvent("executed", 30000);
	 * if (result.success) {
	 *   console.log("Event received:", result.data);
	 * }
	 * ```
	 */
	async waitForEvent(
		eventType: string,
		timeout?: number,
	): Promise<Result<unknown>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		return this.waitForEventInternal(eventType, timeout);
	}

	// Wait for the first occurrence of any of the provided event types
	private async waitForAnyEvent(
		eventTypes: string[],
		timeout?: number,
	): Promise<Result<{ type: string; data: unknown }>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		return new Promise((resolve) => {
			const timeoutMs = timeout ?? this.config.timeout.message;
			let resolved = false;

			const timeoutId = setTimeout(() => {
				if (!resolved) {
					resolved = true;
					this.logger.error(
						`Timeout waiting for any of events: ${eventTypes.join(", ")}`,
					);
					resolve(
						err(
							new ComfyTimeoutError(
								`Timeout waiting for any of events: ${eventTypes.join(", ")}`,
								timeoutMs,
							),
						),
					);
				}
			}, timeoutMs);

			const handlers = new Map<string, (data: unknown) => void>();
			eventTypes.forEach((type) => {
				const handler = (data: unknown) => {
					if (!resolved) {
						resolved = true;
						clearTimeout(timeoutId);
						// remove all handlers
						eventTypes.forEach((t) => {
							const list = this.messageHandlers.get(t);
							if (list) {
								const idx = list.indexOf(handlers.get(t)!);
								if (idx > -1) list.splice(idx, 1);
							}
						});
						resolve(ok({ data, type }));
					}
				};
				handlers.set(type, handler);
				if (!this.messageHandlers.has(type)) {
					this.messageHandlers.set(type, []);
				}
				this.messageHandlers.get(type)!.push(handler);
			});
		});
	}

	/**
	 * Collects all events until completion or timeout
	 *
	 * Useful for capturing the full execution flow of a prompt.
	 * Returns all events and binary data received during execution.
	 *
	 * @param options - Collection options (timeout, waitForCompletion)
	 * @returns Result containing event collection with events, binaryData, and completion status
	 * @example
	 * ```typescript
	 * const result = await client.collectAllEvents({
	 *   timeout: 120000,
	 *   waitForCompletion: true
	 * });
	 * if (result.success) {
	 *   console.log(`Received ${result.data.events.length} events`);
	 *   console.log(`Completed: ${result.data.completed}`);
	 * }
	 * ```
	 */
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
								clearTimeout(timeoutId);
								resolve(ok({ binaryData, completed, error, events }));
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

	/**
	 * Submits a prompt/workflow to ComfyUI for execution
	 *
	 * The prompt is sent via WebSocket and the method waits for acceptance.
	 * Returns when the prompt is accepted and queued for execution.
	 *
	 * @param prompt - ComfyUI workflow/prompt object
	 * @param options - Optional submission options (promptId, extraData, webhookUrl, etc.)
	 * @returns Result containing prompt acceptance data, or error if submission failed
	 * @example
	 * ```typescript
	 * const prompt = {
	 *   "1": {
	 *     "inputs": { "text": "a beautiful landscape" },
	 *     "class_type": "CLIPTextEncode"
	 *   }
	 * };
	 *
	 * const result = await client.submitPrompt(prompt, {
	 *   promptId: "my-prompt-123",
	 *   extraData: { userId: "user123" }
	 * });
	 *
	 * if (result.success) {
	 *   console.log("Prompt accepted:", result.data);
	 * }
	 * ```
	 */
	async submitPrompt(
		prompt: ComfyPrompt,
		options?: SubmitOptions,
	): Promise<Result<unknown>> {
		if (!this.isConnected()) {
			return err(new ComfyConnectionError("Not connected"));
		}

		const promptId = options?.promptId || `prompt-${Date.now()}`;
		const startTime = performance.now();

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
		this.logger.debug(
			`Full message: ${JSON.stringify(message).substring(0, 300)}`,
		);

		const sendStartTime = performance.now();
		const sendResult = this.config.adapter.send(JSON.stringify(message));
		const sendDuration = performance.now() - sendStartTime;

		if (!sendResult.success) {
			this.logger.error(`Failed to send message: ${sendResult.error}`, {
				promptId,
				sendDurationMs: Math.round(sendDuration),
			});
			return sendResult;
		}
		this.logger.debug(
			`Message sent in ${Math.round(sendDuration)}ms for prompt ${promptId}`,
		);
		this.logger.info(
			`✅ Message sent for prompt ${promptId}, now waiting for prompt_accepted event (timeout: ${this.config.timeout.message}ms)`,
		);

		// Wait for prompt acceptance or equivalent start signals seen in some deployments
		// Accept any of these as successful acknowledgement to improve compatibility:
		// - prompt_accepted (preferred)
		// - execution_start / executing
		// - status (some proxies emit a status update immediately upon queueing)
		const acceptStartTime = performance.now();
		const acceptResult = await this.waitForAnyEvent(
			["prompt_accepted", "execution_start", "executing", "status"],
			this.config.timeout.message,
		);
		const acceptDuration = performance.now() - acceptStartTime;
		const totalDuration = performance.now() - startTime;

		if (!acceptResult.success) {
			this.logger.error(
				`❌ Failed to get prompt acceptance for ${promptId}: ${acceptResult.error}`,
				{
					acceptDurationMs: Math.round(acceptDuration),
					promptId,
					totalDurationMs: Math.round(totalDuration),
				},
			);
			return err(
				new ComfyConnectionError(
					`Failed to get prompt acceptance: ${acceptResult.error}`,
				),
			);
		}

		this.logger.info(
			`✅ Prompt accepted: ${promptId} (via ${acceptResult.data.type})`,
			{
				acceptDurationMs: Math.round(acceptDuration),
				promptId,
				totalDurationMs: Math.round(totalDuration),
				...acceptResult.data,
			},
		);
		return ok(acceptResult.data);
	}

	/**
	 * Sends a ping message to the server to keep connection alive
	 *
	 * Heartbeat is handled automatically if enabled in config.
	 * This method can be called manually for custom ping logic.
	 *
	 * @returns Result indicating success or failure
	 */
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

	/**
	 * Validates that an event sequence contains all expected events
	 *
	 * Useful for testing and debugging. Checks if all expected ComfyUI events
	 * are present in the sequence.
	 *
	 * @param events - Array of events to validate
	 * @returns Validation result with missing and extra events
	 * @example
	 * ```typescript
	 * const validation = client.validateEventSequence(events);
	 * if (!validation.valid) {
	 *   console.log("Missing events:", validation.missingEvents);
	 * }
	 * ```
	 */
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

	private setupVisibilityTracking(): void {
		if (!isBrowser()) return;
		this.visibilityHandler = () => {
			if (document.visibilityState === "visible") {
				this.handleTabFocused();
			}
		};
		document.addEventListener("visibilitychange", this.visibilityHandler);
	}

	private handleTabFocused(): void {
		if (
			!this.isConnected() &&
			this.config.autoConnect &&
			!this.reconnectGiveUp
		) {
			this.logger.info("Tab focused, auto-reconnecting...");
			// Reset reconnection state for fresh attempt via tab focus
			this.reconnectAttempts = 0;
			this.reconnectGiveUp = false;
			this.isReconnecting = false;
			this.connect().catch((err) => {
				this.logger.error("Auto-reconnect failed:", err);
			});
		}
	}

	private startHeartbeat(): void {
		if (this.config.heartbeat?.enabled) {
			this.heartbeatIntervalId = setInterval(() => {
				if (this.isConnected()) {
					this.ping().catch(() => {
						this.logger.warn("Heartbeat ping failed");
					});
				}
			}, this.config.heartbeat.interval);
		}
	}
}
