import type { Logger } from "./logger";
import type { QueuedPrompt } from "./types";

export interface CircuitBreakerConfig {
	threshold: number;
	timeoutMs: number;
}

class CircuitBreaker {
	private failures = 0;
	private lastFailureTime = 0;
	private state: "closed" | "open" | "half-open" = "closed";
	private config: CircuitBreakerConfig;
	private logger: Logger;

	constructor(config: CircuitBreakerConfig, logger: Logger) {
		this.config = config;
		this.logger = logger;
	}

	isOpen(): boolean {
		if (this.state === "open") {
			// Check if timeout has passed
			if (Date.now() - this.lastFailureTime > this.config.timeoutMs) {
				this.state = "half-open";
				return false;
			}
			return true;
		}
		return false;
	}

	recordSuccess(): void {
		this.failures = 0;
		this.state = "closed";
	}

	recordFailure(): void {
		this.failures++;
		this.lastFailureTime = Date.now();

		if (this.failures >= this.config.threshold) {
			this.state = "open";
			this.logger.warn(
				`Circuit breaker opened after ${this.failures} failures`,
			);
		}
	}

	getState(): string {
		return this.state;
	}
}

export class PromptQueue {
	private queues = new Map<string, QueuedPrompt[]>(); // userId -> prompts
	private circuitBreaker: CircuitBreaker;
	private maxQueuedPromptsPerUser: number;
	private logger: Logger;

	constructor(
		maxQueuedPromptsPerUser: number,
		circuitBreakerConfig: CircuitBreakerConfig,
		logger: Logger,
	) {
		this.maxQueuedPromptsPerUser = maxQueuedPromptsPerUser;
		this.logger = logger;
		this.circuitBreaker = new CircuitBreaker(circuitBreakerConfig, logger);
	}

	addPrompt(userId: string, prompt: QueuedPrompt): boolean {
		const userQueue = this.queues.get(userId) || [];

		if (userQueue.length >= this.maxQueuedPromptsPerUser) {
			this.logger.warn(`Queue full for user ${userId}, rejecting prompt`);
			return false;
		}

		userQueue.push(prompt);
		this.queues.set(userId, userQueue);
		this.logger.debug(
			`Queued prompt for user ${userId}, queue size: ${userQueue.length}`,
		);
		return true;
	}

	getNextPrompt(userId: string): QueuedPrompt | undefined {
		const userQueue = this.queues.get(userId) || [];
		return userQueue.shift();
	}

	getQueueSize(userId: string): number {
		return this.queues.get(userId)?.length || 0;
	}

	getAllQueueSizes(): Map<string, number> {
		const sizes = new Map<string, number>();
		for (const [userId, queue] of this.queues.entries()) {
			sizes.set(userId, queue.length);
		}
		return sizes;
	}

	clearQueue(userId: string): void {
		this.queues.delete(userId);
	}

	clearAllQueues(): void {
		this.queues.clear();
	}

	// Circuit breaker methods
	canProcess(): boolean {
		return !this.circuitBreaker.isOpen();
	}

	recordSuccess(): void {
		this.circuitBreaker.recordSuccess();
	}

	recordFailure(): void {
		this.circuitBreaker.recordFailure();
	}

	getCircuitBreakerState(): string {
		return this.circuitBreaker.getState();
	}
}

// Factory function to create a PromptQueue instance
export function createPromptQueue(
	maxQueuedPromptsPerUser: number,
	circuitBreakerConfig: CircuitBreakerConfig,
	logger: Logger,
): PromptQueue {
	return new PromptQueue(maxQueuedPromptsPerUser, circuitBreakerConfig, logger);
}
