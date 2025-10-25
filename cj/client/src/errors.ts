export class ComfyConnectionError extends Error {
	constructor(
		message: string,
		public readonly code?: string,
	) {
		super(message);
		this.name = "ComfyConnectionError";
	}
}

export class ComfyTimeoutError extends Error {
	constructor(
		message: string,
		public readonly timeoutMs?: number,
	) {
		super(message);
		this.name = "ComfyTimeoutError";
	}
}

export class ComfyAuthError extends Error {
	constructor(
		message: string,
		public readonly authType?: string,
	) {
		super(message);
		this.name = "ComfyAuthError";
	}
}

export class ComfyPromptError extends Error {
	constructor(
		message: string,
		public readonly promptId?: string,
	) {
		super(message);
		this.name = "ComfyPromptError";
	}
}

export class ComfyReconnectError extends Error {
	constructor(
		message: string,
		public readonly attempt?: number,
	) {
		super(message);
		this.name = "ComfyReconnectError";
	}
}

// Union type for all ComfyUI client errors
export type ComfyClientError =
	| ComfyConnectionError
	| ComfyTimeoutError
	| ComfyAuthError
	| ComfyPromptError
	| ComfyReconnectError;

export type LogLevel = "debug" | "info" | "warn" | "error" | "silent";

export class Logger {
	private level: LogLevel;
	private prefix: string;

	constructor(level: LogLevel = "info", prefix = "[ComfyClient]") {
		this.level = level;
		this.prefix = prefix;
	}

	private shouldLog(level: LogLevel): boolean {
		const levels: LogLevel[] = ["debug", "info", "warn", "error", "silent"];
		const currentIndex = levels.indexOf(this.level);
		const messageIndex = levels.indexOf(level);
		return messageIndex >= currentIndex;
	}

	private formatMessage(
		level: LogLevel,
		message: string,
		..._args: unknown[]
	): string {
		const timestamp = new Date().toISOString();
		return `${this.prefix} [${timestamp}] [${level.toUpperCase()}] ${message}`;
	}

	debug(message: string, ...args: unknown[]): void {
		if (this.shouldLog("debug")) {
			console.debug(this.formatMessage("debug", message), ...args);
		}
	}

	info(message: string, ...args: unknown[]): void {
		if (this.shouldLog("info")) {
			console.info(this.formatMessage("info", message), ...args);
		}
	}

	warn(message: string, ...args: unknown[]): void {
		if (this.shouldLog("warn")) {
			console.warn(this.formatMessage("warn", message), ...args);
		}
	}

	error(message: string, ...args: unknown[]): void {
		if (this.shouldLog("error")) {
			console.error(this.formatMessage("error", message), ...args);
		}
	}

	setLevel(level: LogLevel): void {
		this.level = level;
	}

	setPrefix(prefix: string): void {
		this.prefix = prefix;
	}
}
