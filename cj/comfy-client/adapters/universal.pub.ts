import { ComfyConnectionError, ComfyTimeoutError, Logger } from "../errors";
import type { Result } from "../types";
import { err, ok } from "../types";
import type { ConnectionOptions, ConnectionState } from "./base";
import { WebSocketAdapter } from "./base";

/**
 * Universal WebSocket adapter for Browser and Bun runtimes.
 * Uses the global WebSocket and event listeners with tracked cleanup.
 */
export class UniversalWebSocketAdapter extends WebSocketAdapter {
	private ws: WebSocket | null = null;
	/** Track active connection timeout to prevent race conditions */
	private connectionTimeoutId: NodeJS.Timeout | null = null;
	/** Track event listeners for cleanup to prevent memory leaks */
	private eventListeners: Array<{
		event: string;
		handler: (...args: unknown[]) => void;
	}> = [];

	constructor(url?: string, options?: ConnectionOptions, logger?: Logger) {
		super(
			url || "",
			options,
			logger || new Logger("info", "[UniversalWebSocketAdapter]"),
		);
	}

	protected async _connect(): Promise<Result<void>> {
		return new Promise((resolve) => {
			try {
				this.logger.debug(
					`Connecting to ${this.url} with options:`,
					this.options,
				);

				// Do not pass protocols; neither server endpoint negotiates subprotocols
				this.ws = new WebSocket(this.url);

				let settled = false;
				const timeout = this.options.timeout || 10000;

				// Connection timeout
				this.connectionTimeoutId = setTimeout(() => {
					if (!settled && this.ws) {
						settled = true;
						this.logger.error(`Connection timeout after ${timeout}ms`);
						try {
							this.ws.close();
						} catch (closeError) {
							this.logger.error("Error closing on timeout:", closeError);
						}
						resolve(
							err(
								new ComfyTimeoutError(
									`Connection timeout after ${timeout}ms`,
									timeout,
								),
							),
						);
					}
				}, timeout);

				// Helper to settle once and clear timeout
				const settle = (result: Result<void>) => {
					if (!settled && this.connectionTimeoutId) {
						settled = true;
						clearTimeout(this.connectionTimeoutId);
						this.connectionTimeoutId = null;
						resolve(result);
					}
				};

				// open
				this.addListener("open", () => {
					this.logConnectSuccess();
					settle(ok(undefined));
				});

				// error during connection
				this.addListener("error", (error) => {
					this.logConnectError(error);
					if (this._errorHandler) {
						this._errorHandler(
							new ComfyConnectionError("WebSocket connection failed"),
						);
					}
					settle(err(new ComfyConnectionError("WebSocket connection failed")));
				});

				// close (always clear timeout)
				this.addListener("close", (event) => {
					if (this.connectionTimeoutId) {
						clearTimeout(this.connectionTimeoutId);
						this.connectionTimeoutId = null;
					}
					if (event instanceof CloseEvent) {
						this.logCloseEvent(event.code, event.reason);
						if (this.closeHandler) {
							this.closeHandler(event.code, event.reason);
						}
					} else {
						this.logger.error("Close event is not a CloseEvent:", event);
					}
				});

				// message
				// biome-ignore lint/suspicious/noExplicitAny: event param comes from runtime
				this.addListener("message", (event: any) => {
					this.logMessageReceived();
					if (this.messageHandler) {
						if (event instanceof MessageEvent) {
							this.messageHandler(event.data);
						} else {
							this.logger.error("Message event is not a MessageEvent:", event);
						}
					}
				});
			} catch (error) {
				this.logger.error("Failed to create WebSocket:", error);
				if (this.connectionTimeoutId) {
					clearTimeout(this.connectionTimeoutId);
					this.connectionTimeoutId = null;
				}
				resolve(
					err(new ComfyConnectionError(`Failed to create WebSocket: ${error}`)),
				);
			}
		});
	}

	protected _send(data: string | ArrayBuffer): Result<void> {
		if (!this.ws) {
			return err(new ComfyConnectionError("WebSocket not connected"));
		}
		try {
			this.ws.send(data);
			return ok(undefined);
		} catch (error) {
			return err(new ComfyConnectionError(`Failed to send message: ${error}`));
		}
	}

	protected _close(): void {
		if (this.connectionTimeoutId) {
			clearTimeout(this.connectionTimeoutId);
			this.connectionTimeoutId = null;
		}

		if (this.ws) {
			// Remove all event listeners to prevent leaks
			for (const { event, handler } of this.eventListeners) {
				try {
					this.ws.removeEventListener(event, handler);
				} catch (error) {
					this.logger.error(`Error removing listener for ${event}:`, error);
				}
			}
			this.eventListeners = [];

			try {
				this.ws.close();
			} catch (error) {
				this.logger.error("Error during WebSocket close:", error);
			}
			this.ws = null;
		}
	}

	getReadyState(): ConnectionState {
		if (!this.ws) return "closed";
		switch (this.ws.readyState) {
			case 0:
				return "connecting";
			case 1:
				return "open";
			case 2:
				return "closing";
			case 3:
				return "closed";
			default:
				return "closed";
		}
	}

	private addListener(
		event: string,
		handler: (...args: unknown[]) => void,
	): void {
		if (this.ws) {
			this.ws.addEventListener(event, handler as EventListener);
			this.eventListeners.push({ event, handler });
		}
	}
}
