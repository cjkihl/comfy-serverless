import type { WebSocket } from "bun";
import { ComfyConnectionError, ComfyTimeoutError, Logger } from "../errors";
import type { Result } from "../types";
import { err, ok } from "../types";
import type {
	ConnectionOptions,
	ConnectionState,
	WebSocketAdapter,
} from "./types";

export class BunWebSocketAdapter implements WebSocketAdapter {
	private ws: WebSocket | null = null;
	private messageHandler?: (data: string | ArrayBuffer) => void;
	private closeHandler?: (code: number, reason: string) => void;
	private _errorHandler?: (error: Error) => void;
	private logger: Logger;

	constructor(logger?: Logger) {
		this.logger = logger || new Logger("debug", "[BunWebSocketAdapter]");
	}

	async connect(
		url: string,
		options: ConnectionOptions,
	): Promise<Result<void>> {
		return new Promise((resolve) => {
			try {
				this.logger.debug(`Connecting to ${url} with options:`, options);

				const wsOptions: {
					headers?: Record<string, string>;
					protocols?: string[];
				} = {};
				if (options.headers) {
					wsOptions.headers = options.headers;
				}
				if (options.protocols) {
					wsOptions.protocols = options.protocols;
				}

				this.ws = new WebSocket(url, wsOptions);

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

				this.ws!.addEventListener("open", () => {
					if (!settled) {
						settled = true;
						clearTimeout(timeoutId);
						this.logger.info("WebSocket connection established");
						resolve(ok(undefined));
					}
				});

				this.ws!.addEventListener("error", (error) => {
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
				});

				this.ws!.addEventListener("close", (event) => {
					clearTimeout(timeoutId);
					if (this.closeHandler) {
						this.closeHandler(event.code, event.reason);
					}
				});

				this.ws!.addEventListener("message", (event: MessageEvent) => {
					if (this.messageHandler) {
						this.messageHandler(event.data);
					}
				});
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
			const messageStr = typeof data === "string" ? data : "<binary>";
			this.logger.debug(`Sending message: ${messageStr.substring(0, 200)}`);
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
			case 0: // CONNECTING
				return "connecting";
			case 1: // OPEN
				return "open";
			case 2: // CLOSING
				return "closing";
			case 3: // CLOSED
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
