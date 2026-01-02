import { ComfyConnectionError, Logger } from "../errors";
import type { Result } from "../types";
import { err } from "../types";

/**
 * WebSocket connection state enumeration
 * @readonly
 */
export type ConnectionState = "connecting" | "open" | "closing" | "closed";

/**
 * Configuration options for WebSocket connection
 */
export interface ConnectionOptions {
	/** Connection timeout in milliseconds (default: 10000) */
	timeout?: number;
}

/**
 * Abstract base class for WebSocket adapters.
 * Provides a unified interface for WebSocket connections across different platforms (Browser, Bun, etc.).
 * Implements the Template Method pattern where subclasses provide platform-specific implementations.
 *
 * @abstract
 * @example
 * ```typescript
 * class MyWebSocketAdapter extends WebSocketAdapter {
 *   protected async _connect(): Promise<Result<void>> {
 *     // Platform-specific connection logic
 *   }
 *   protected _send(data: string | ArrayBuffer): Result<void> {
 *     // Platform-specific send logic
 *   }
 *   protected _close(): void {
 *     // Platform-specific cleanup
 *   }
 *   getReadyState(): ConnectionState {
 *     // Platform-specific state mapping
 *   }
 * }
 * ```
 */
export abstract class WebSocketAdapter {
	protected url: string;
	protected options: ConnectionOptions;
	protected logger: Logger;
	protected messageHandler?: (data: string | ArrayBuffer) => void;
	protected closeHandler?: (code: number, reason: string) => void;
	protected _errorHandler?: (error: Error) => void;

	/**
	 * Create a new WebSocket adapter instance
	 * @param url - WebSocket URL to connect to
	 * @param options - Optional connection configuration
	 * @param logger - Optional logger instance
	 */
	constructor(url: string, options?: ConnectionOptions, logger?: Logger) {
		this.url = url;
		this.options = options || {};
		this.logger = logger || new Logger("info", "[WebSocketAdapter]");
	}

	/**
	 * Update the connection URL.
	 * @param url - New WebSocket URL
	 */
	setUrl(url: string): void {
		this.url = url;
	}

	/**
	 * Get the current connection URL.
	 * @returns Current connection URL
	 */
	getUrl(): string {
		return this.url;
	}

	/**
	 * Update the connection options.
	 * @param options - New connection configuration options
	 */
	setOptions(options: ConnectionOptions): void {
		this.options = options;
	}

	/**
	 * Establish a WebSocket connection.
	 * Template method that handles logging and delegates to platform-specific _connect().
	 *
	 * @returns Promise resolving to a Result indicating connection success or failure
	 */
	async connect(): Promise<Result<void>> {
		this.logConnectStart();
		return this._connect();
	}

	/**
	 * Send data over the WebSocket connection.
	 * Template method that validates connection state, logs, and delegates to platform-specific _send().
	 *
	 * @param data - String or binary data to send
	 * @returns Result indicating send success or failure
	 * @throws ComfyConnectionError if not in open state
	 */
	send(data: string | ArrayBuffer): Result<void> {
		if (this.getReadyState() !== "open") {
			return err(new ComfyConnectionError("WebSocket not in open state"));
		}
		this.logSend(data);
		const result = this._send(data);
		if (!result.success) {
			// Type assertion needed due to TS control flow analysis limitation
			this.logSendError((result as { success: false; error: Error }).error);
			return result;
		}
		this.logSendSuccess();
		return result;
	}

	/**
	 * Close the WebSocket connection and clean up all handlers.
	 * Template method that handles logging, delegates to platform-specific _close(), and removes listeners.
	 */
	close(): void {
		this.logClose();
		this._close();
		this.removeAllListeners();
	}

	/**
	 * Register a handler for incoming WebSocket messages
	 * @param handler - Function to call when messages are received
	 */
	onMessage(handler: (data: string | ArrayBuffer) => void): void {
		this.messageHandler = handler;
	}

	/**
	 * Register a handler for WebSocket close events
	 * @param handler - Function to call when connection closes (receives code and reason)
	 */
	onClose(handler: (code: number, reason: string) => void): void {
		this.closeHandler = handler;
	}

	/**
	 * Register a handler for WebSocket errors
	 * @param handler - Function to call when errors occur
	 */
	onError(handler: (error: Error) => void): void {
		this._errorHandler = handler;
	}

	/**
	 * Get the current connection state of the WebSocket.
	 * Subclasses must implement this to map platform-specific readyState values.
	 *
	 * @returns Current connection state
	 */
	abstract getReadyState(): ConnectionState;

	/**
	 * Remove all registered event handlers to prevent memory leaks.
	 * Called automatically by close() but can be called manually if needed.
	 */
	removeAllListeners(): void {
		this.messageHandler = undefined;
		this.closeHandler = undefined;
		this._errorHandler = undefined;
		this.logger.debug("All listeners removed");
	}

	// ============================================================================
	// Protected Logging Helpers (for subclasses)
	// ============================================================================

	/** Log connection attempt start */
	protected logConnectStart(): void {
		this.logger.debug(`Connecting to ${this.url}`);
	}

	/** Log successful connection establishment */
	protected logConnectSuccess(): void {
		this.logger.info("WebSocket connection established");
	}

	/**
	 * Log connection error
	 * @param error - Error that occurred during connection
	 */
	protected logConnectError(error: unknown): void {
		this.logger.error("WebSocket connection error:", error);
	}

	/**
	 * Log message being sent (truncates long strings for readability)
	 * @param data - Data being sent
	 */
	protected logSend(data: string | ArrayBuffer): void {
		if (typeof data === "string") {
			const preview = data.substring(0, 200);
			this.logger.debug(
				`Sending message: ${preview}${data.length > 200 ? "..." : ""}`,
			);
		} else {
			this.logger.debug("Sending binary message");
		}
	}

	/** Log successful message send */
	protected logSendSuccess(): void {
		this.logger.debug("Message sent successfully");
	}

	/**
	 * Log failed message send
	 * @param error - Error that occurred
	 */
	protected logSendError(error: Error): void {
		this.logger.error("Failed to send message:", error);
	}

	/** Log connection close attempt */
	protected logClose(): void {
		this.logger.debug("Closing WebSocket connection");
	}

	/**
	 * Log connection close event
	 * @param code - WebSocket close code
	 * @param reason - Close reason string
	 */
	protected logCloseEvent(code: number, reason: string): void {
		this.logger.debug(`WebSocket closed: code=${code}, reason=${reason}`);
	}

	/**
	 * Log WebSocket error event
	 * @param error - Error that occurred
	 */
	protected logErrorEvent(error: Error): void {
		this.logger.error("WebSocket error event:", error);
	}

	/** Log message reception */
	protected logMessageReceived(): void {
		this.logger.debug("Message received");
	}

	// ============================================================================
	// Abstract Methods (must be implemented by subclasses)
	// ============================================================================

	/**
	 * Establish the WebSocket connection (implementation-specific).
	 * Subclasses must implement this to create and configure the WebSocket for their platform.
	 *
	 * **Implementation requirements:**
	 * - Create WebSocket instance with correct API for platform
	 * - Set up event listeners for open, error, close, and message events
	 * - Handle connection timeout
	 * - Return err(ComfyTimeoutError) on timeout
	 * - Return err(ComfyConnectionError) on failure
	 * - Ensure listeners are cleaned up on timeout or error
	 * - Return ok(undefined) on successful connection
	 *
	 * @returns Promise resolving to a Result indicating success or failure
	 */
	protected abstract _connect(): Promise<Result<void>>;

	/**
	 * Send data over the WebSocket connection (implementation-specific).
	 * Subclasses must implement this to handle platform-specific WebSocket.send() API.
	 * The base class already validates readyState === "open" before calling this.
	 *
	 * **Implementation requirements:**
	 * - Call the WebSocket instance's send method
	 * - Return err(ComfyConnectionError) if send fails
	 * - Return ok(undefined) on successful send
	 *
	 * @param data - Data to send (string or ArrayBuffer)
	 * @returns Result indicating send success or failure
	 */
	protected abstract _send(data: string | ArrayBuffer): Result<void>;

	/**
	 * Close the WebSocket connection and perform cleanup (implementation-specific).
	 * Subclasses must implement this to clean up platform-specific resources.
	 *
	 * **Implementation requirements:**
	 * - Call WebSocket close() method
	 * - Remove all event listeners to prevent memory leaks
	 * - Set internal WebSocket reference to null
	 * - Handle any exceptions during close
	 *
	 * Note: removeAllListeners() is called automatically by the base class close() method.
	 */
	protected abstract _close(): void;
}
