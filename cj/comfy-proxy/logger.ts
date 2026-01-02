import consola, { type ConsolaInstance } from "consola";

export type LogLevel = "debug" | "info" | "warn" | "error" | "silent";

const LEVEL_MAP: Record<LogLevel, number> = {
	debug: 0,
	error: 5,
	info: 3,
	silent: 7,
	warn: 4,
};

export class Logger {
	private consolaInstance: ConsolaInstance;
	private correlationId?: string;

	constructor(level: LogLevel = "info", prefix = "[Proxy]") {
		this.consolaInstance = consola.withTag(prefix);
		this.consolaInstance.level = LEVEL_MAP[level];
	}

	setCorrelationId(correlationId: string): void {
		this.correlationId = correlationId;
	}

	getCorrelationId(): string | undefined {
		return this.correlationId;
	}

	withCorrelationId(correlationId: string): Logger {
		const child = new Logger();
		child.consolaInstance = this.consolaInstance;
		child.correlationId = correlationId;
		return child;
	}

	private formatArgs(...args: unknown[]): unknown[] {
		if (!this.correlationId) return args;
		if (args.length > 0 && typeof args[0] === "object" && args[0] !== null) {
			return [
				{ ...args[0], correlationId: this.correlationId },
				...args.slice(1),
			];
		}
		return [{ correlationId: this.correlationId }, ...args];
	}

	debug(message: string, ...args: unknown[]): void {
		this.consolaInstance.debug(message, ...this.formatArgs(...args));
	}

	info(message: string, ...args: unknown[]): void {
		this.consolaInstance.info(message, ...this.formatArgs(...args));
	}

	warn(message: string, ...args: unknown[]): void {
		this.consolaInstance.warn(message, ...this.formatArgs(...args));
	}

	error(message: string, ...args: unknown[]): void {
		this.consolaInstance.error(message, ...this.formatArgs(...args));
	}

	setLevel(level: LogLevel): void {
		this.consolaInstance.level = LEVEL_MAP[level];
	}

	setPrefix(prefix: string): void {
		const currentLevel = this.consolaInstance.level;
		this.consolaInstance = consola.withTag(prefix);
		this.consolaInstance.level = currentLevel;
	}
}
