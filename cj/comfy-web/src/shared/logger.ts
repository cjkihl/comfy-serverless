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

	constructor(level: LogLevel = "info", prefix = "[Web]") {
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

export interface LogEntry {
	timestamp: string;
	message: string;
	type: "info" | "success" | "error" | "warn" | "debug";
}

export class UILogger {
	private logger: Logger;
	private entries: LogEntry[] = [];
	private listeners: Array<(entries: LogEntry[]) => void> = [];

	constructor(level: LogLevel = "info", prefix = "[Web]") {
		this.logger = new Logger(level, prefix);
	}

	private addUIEntry(message: string, type: LogEntry["type"]) {
		const timestamp = new Date().toISOString();
		const entry: LogEntry = { message, timestamp, type };
		this.entries.push(entry);
		this.notifyListeners();
		if (this.entries.length > 100) {
			this.entries = this.entries.slice(-100);
		}
	}

	private notifyListeners() {
		for (const listener of this.listeners) {
			listener([...this.entries]);
		}
	}

	info(message: string, ...args: unknown[]): void {
		this.logger.info(message, ...args);
		this.addUIEntry(message, "info");
	}

	success(message: string, ...args: unknown[]): void {
		this.logger.info(message, ...args);
		this.addUIEntry(message, "success");
	}

	error(message: string, ...args: unknown[]): void {
		this.logger.error(message, ...args);
		this.addUIEntry(message, "error");
	}

	warn(message: string, ...args: unknown[]): void {
		this.logger.warn(message, ...args);
		this.addUIEntry(message, "warn");
	}

	debug(message: string, ...args: unknown[]): void {
		this.logger.debug(message, ...args);
		this.addUIEntry(message, "debug");
	}

	setLevel(level: LogLevel): void {
		this.logger.setLevel(level);
	}

	setPrefix(prefix: string): void {
		this.logger.setPrefix(prefix);
	}

	clear() {
		this.entries = [];
		this.notifyListeners();
	}

	getEntries(): LogEntry[] {
		return [...this.entries];
	}

	subscribe(listener: (entries: LogEntry[]) => void) {
		this.listeners.push(listener);
		return () => {
			const index = this.listeners.indexOf(listener);
			if (index > -1) this.listeners.splice(index, 1);
		};
	}
}
