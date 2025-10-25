import { ComfyConnectionError, ComfyTimeoutError, Logger } from "../errors";
import type { Result } from "../types";
import { err, ok } from "../types";
import type {
	ConnectionOptions,
	ConnectionState,
	WebSocketAdapter,
} from "./types";

export class BrowserWebSocketAdapter implements WebSocketAdapter {
	private ws: WebSocket | null = null;
	private messageHandler?: (data: string | ArrayBuffer) => void;
	private closeHandler?: (code: number, reason: string) => void;
	private _errorHandler?: (error: Error) => void;
	private logger: Logger;

	constructor(logger?: Logger) {
		this.logger = logger || new Logger("info", "[BrowserWebSocketAdapter]");
	}

	async connect(
		url: string,
		options: ConnectionOptions,
	): Promise<Result<void>> {
		return new Promise((resolve) => {
			try {
				// Browser WebSocket doesn't support custom headers directly
				// We'll need to pass auth via query params or other means
				let finalUrl = url;

				if (options.headers?.Authorization) {
					// Extract JWT from Authorization header and add as query param
					const authHeader = options.headers.Authorization;
					if (authHeader.startsWith("Bearer ")) {
						const token = authHeader.substring(7);
						const separator = url.includes("?") ? "&" : "?";
						finalUrl = `${url}${separator}token=${encodeURIComponent(token)}`;
					}
				}

				this.logger.debug(`Connecting to ${finalUrl}`);

				this.ws = new WebSocket(finalUrl, options.protocols);

				let settled = false;
				const timeout = options.timeout || 10000;

				const timeoutId = setTimeout(() => {
					if (!settled && this.ws) {
						settled = true;
						this.logger.error(`Connection timeout after ${timeout}ms`);
						try {
							this.ws.close();
						} catch {}
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

				this.ws.onopen = () => {
					if (!settled) {
						settled = true;
						clearTimeout(timeoutId);
						this.logger.info("WebSocket connection established");
						resolve(ok(undefined));
					}
				};

				this.ws.onerror = (error) => {
					if (!settled) {
						settled = true;
						clearTimeout(timeoutId);
						this.logger.error("WebSocket connection error:", error);
						if (this._errorHandler) {
							this._errorHandler(
								new ComfyConnectionError("WebSocket connection failed"),
							);
						}
						resolve(
							err(new ComfyConnectionError("WebSocket connection failed")),
						);
					}
				};

				this.ws.onclose = (event) => {
					clearTimeout(timeoutId);
					if (this.closeHandler) {
						this.closeHandler(event.code, event.reason);
					}
				};

				this.ws.onmessage = (event) => {
					if (this.messageHandler) {
						this.messageHandler(event.data);
					}
				};
			} catch (error) {
				this.logger.error("Failed to create WebSocket:", error);
				resolve(
					err(new ComfyConnectionError(`Failed to create WebSocket: ${error}`)),
				);
			}
		});
	}

	send(data: string | ArrayBuffer): Result<void> {
		if (!this.ws) {
			return err(new ComfyConnectionError("WebSocket not connected"));
		}

		if (this.getReadyState() !== "open") {
			return err(new ComfyConnectionError("WebSocket not in open state"));
		}

		try {
			this.ws.send(data);
			this.logger.debug("Message sent successfully");
			return ok(undefined);
		} catch (error) {
			this.logger.error("Failed to send message:", error);
			return err(new ComfyConnectionError(`Failed to send message: ${error}`));
		}
	}

	close(): void {
		if (this.ws) {
			this.logger.debug("Closing WebSocket connection");
			this.ws.close();
			this.ws = null;
		}
	}

	onMessage(handler: (data: string | ArrayBuffer) => void): void {
		this.messageHandler = handler;
	}

	onClose(handler: (code: number, reason: string) => void): void {
		this.closeHandler = handler;
	}

	onError(handler: (error: Error) => void): void {
		this._errorHandler = handler;
	}

	getReadyState(): ConnectionState {
		if (!this.ws) return "closed";

		switch (this.ws.readyState) {
			case WebSocket.CONNECTING:
				return "connecting";
			case WebSocket.OPEN:
				return "open";
			case WebSocket.CLOSING:
				return "closing";
			case WebSocket.CLOSED:
				return "closed";
			default:
				return "closed";
		}
	}

	removeAllListeners(): void {
		this.messageHandler = undefined;
		this.closeHandler = undefined;
		this._errorHandler = undefined;
	}
}
