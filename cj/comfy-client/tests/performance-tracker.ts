#!/usr/bin/env bun

/**
 * Performance tracker for measuring timing phases in test scenarios
 * Tracks multiple phases with high-resolution timestamps
 */

interface PerformanceMetrics {
	phases: Array<{ name: string; duration: number }>;
	startTime: number;
	endTime?: number;
	totalDuration?: number;
}

export class PerformanceTracker {
	private metrics: PerformanceMetrics;
	private currentPhase?: string;
	private phaseStartTime?: number;

	constructor() {
		this.metrics = {
			phases: [],
			startTime: performance.now(),
		};
	}

	/**
	 * Start tracking a named phase
	 */
	startPhase(phaseName: string): void {
		// End any current phase
		if (this.currentPhase && this.phaseStartTime !== undefined) {
			this.endPhase(this.currentPhase);
		}

		this.currentPhase = phaseName;
		this.phaseStartTime = performance.now();
	}

	/**
	 * End tracking a named phase
	 */
	endPhase(phaseName: string): void {
		if (!this.phaseStartTime || !this.currentPhase) {
			console.warn("No active phase to end");
			return;
		}

		const duration = performance.now() - this.phaseStartTime;

		this.metrics.phases.push({
			duration,
			name: phaseName,
		});

		this.currentPhase = undefined;
		this.phaseStartTime = undefined;
	}

	/**
	 * Mark the overall test as complete
	 */
	complete(): void {
		this.metrics.endTime = performance.now();
		this.metrics.totalDuration =
			(this.metrics.endTime - this.metrics.startTime) / 1000;
	}

	/**
	 * Get the current performance metrics
	 */
	getMetrics(): PerformanceMetrics {
		// If still in a phase, end it
		if (this.currentPhase && this.phaseStartTime !== undefined) {
			this.endPhase(this.currentPhase);
		}

		// Complete if not already completed
		if (!this.metrics.endTime) {
			this.complete();
		}

		return { ...this.metrics };
	}

	/**
	 * Reset the tracker to start a new measurement
	 */
	reset(): void {
		this.metrics = {
			phases: [],
			startTime: performance.now(),
		};
		this.currentPhase = undefined;
		this.phaseStartTime = undefined;
	}

	/**
	 * Get a specific phase duration by name
	 */
	getPhaseDuration(phaseName: string): number | undefined {
		return this.metrics.phases.find((p) => p.name === phaseName)?.duration;
	}

	/**
	 * Check if a phase was tracked
	 */
	hasPhase(phaseName: string): boolean {
		return this.metrics.phases.some((p) => p.name === phaseName);
	}
}

/**
 * Format duration in milliseconds to human readable format
 */
function formatDuration(ms: number): string {
	if (ms < 1000) {
		return `${ms.toFixed(0)}ms`;
	}
	if (ms < 60000) {
		return `${(ms / 1000).toFixed(1)}s`;
	}
	const minutes = Math.floor(ms / 60000);
	const seconds = ((ms % 60000) / 1000).toFixed(1);
	return `${minutes}m ${seconds}s`;
}

/**
 * Print performance metrics in a formatted table
 */
export function printPerformanceMetrics(metrics: PerformanceMetrics): void {
	if (!metrics.phases || metrics.phases.length === 0) {
		console.log("\n⚠️  No performance metrics to display");
		return;
	}

	console.log("\n⏱️  Performance Breakdown:");
	console.log("=".repeat(80));

	// Calculate total time for percentage calculations
	const totalMs = metrics.phases.reduce(
		(sum, phase) => sum + phase.duration,
		0,
	);

	// Find the longest phase name for formatting
	const maxNameLength = Math.max(
		...metrics.phases.map((p) => p.name.length),
		20,
	);

	// Print header
	console.log(
		`Phase Name${" ".repeat(maxNameLength - 10)} | Duration      | % of Total`,
	);
	console.log("=".repeat(80));

	// Print each phase
	for (const phase of metrics.phases) {
		const name = phase.name.padEnd(maxNameLength);
		const duration = formatDuration(phase.duration);
		const percentage =
			totalMs > 0 ? ((phase.duration / totalMs) * 100).toFixed(1) : "0.0";
		const paddedDuration = duration.padEnd(13);
		console.log(`${name} | ${paddedDuration} | ${percentage}%`);
	}

	console.log("=".repeat(80));

	// Print summary
	if (metrics.totalDuration !== undefined) {
		console.log(
			`\n📊 Total Execution Time: ${formatDuration(metrics.totalDuration * 1000)}`,
		);
	}

	// Identify bottlenecks
	const sortedPhases = [...metrics.phases].sort(
		(a, b) => b.duration - a.duration,
	);
	if (sortedPhases.length > 0 && totalMs > 100) {
		const topPhase = sortedPhases[0];
		if (topPhase) {
			const topPercentage =
				totalMs > 0 ? (topPhase.duration / totalMs) * 100 : 0;
			if (topPercentage > 50) {
				console.log(
					`\n🔥 Bottleneck detected: ${topPhase.name} takes ${topPercentage.toFixed(1)}% of total time`,
				);
			}
		}
	}
}
