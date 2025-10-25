import { env } from "./env";
import type { QueuedPrompt } from "./types";

const CIRCUIT_BREAKER_THRESHOLD = 5; // Failures before opening circuit
const CIRCUIT_BREAKER_TIMEOUT = 30000; // 30 seconds

export class CircuitBreaker {
	private failures = 0;
	private lastFailureTime = 0;
	private state: "closed" | "open" | "half-open" = "closed";

	isOpen(): boolean {
		if (this.state === "open") {
			// Check if timeout has passed
			if (Date.now() - this.lastFailureTime > CIRCUIT_BREAKER_TIMEOUT) {
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

		if (this.failures >= CIRCUIT_BREAKER_THRESHOLD) {
			this.state = "open";
			console.warn(`Circuit breaker opened after ${this.failures} failures`);
		}
	}

	getState(): string {
		return this.state;
	}
}

export class PromptQueue {
	private queues = new Map<string, QueuedPrompt[]>(); // userId -> prompts
	private circuitBreaker = new CircuitBreaker();

	addPrompt(userId: string, prompt: QueuedPrompt): boolean {
		const userQueue = this.queues.get(userId) || [];

		if (userQueue.length >= env.MAX_QUEUED_PROMPTS_PER_USER) {
			console.warn(`Queue full for user ${userId}, rejecting prompt`);
			return false;
		}

		userQueue.push(prompt);
		this.queues.set(userId, userQueue);
		console.log(
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

// Global instances
export const promptQueue = new PromptQueue();
export const circuitBreaker = new CircuitBreaker();
