import type { Result } from "../types";

export type ConnectionState = "connecting" | "open" | "closing" | "closed";

export interface ConnectionOptions {
	headers?: Record<string, string>;
	protocols?: string[];
	timeout?: number;
}

export interface WebSocketAdapter {
	/**
	 * Establish a WebSocket connection
	 */
	connect(url: string, options: ConnectionOptions): Promise<Result<void>>;

	/**
	 * Send data over the WebSocket connection
	 */
	send(data: string | ArrayBuffer): Result<void>;

	/**
	 * Close the WebSocket connection
	 */
	close(): void;

	/**
	 * Register a message handler
	 */
	onMessage(handler: (data: string | ArrayBuffer) => void): void;

	/**
	 * Register a close handler
	 */
	onClose(handler: (code: number, reason: string) => void): void;

	/**
	 * Register an error handler
	 */
	onError(handler: (error: Error) => void): void;

	/**
	 * Get the current ready state of the WebSocket
	 */
	getReadyState(): ConnectionState;

	/**
	 * Remove all event listeners
	 */
	removeAllListeners(): void;
}
