/**
 * Shared test utilities that work in both Bun and Browser environments
 */

/**
 * Performance metrics interface (mirrored from @cj/comfy-client)
 */
export interface PerformanceMetrics {
	phases: Array<{ name: string; duration: number }>;
	startTime: number;
	endTime?: number;
	totalDuration?: number;
}

/**
 * Generate a unique test user ID with timestamp and random component
 * This ensures test isolation and prevents conflicts between test runs
 */
export function generateUniqueUserId(prefix = "test-user"): string {
	const timestamp = Date.now();
	const random = Math.random().toString(36).substring(2, 9);
	return `${prefix}-${timestamp}-${random}`;
}

/**
 * Measure the execution time of an async function
 */
export async function measureTime<T>(
	fn: () => Promise<T>,
): Promise<{ result: T; duration: number }> {
	const start = performance.now();
	const result = await fn();
	const duration = performance.now() - start;
	return { duration, result };
}

/**
 * Validate that all expected event types were received
 */
export function validateEventSequence(events: unknown[]): {
	valid: boolean;
	missingEvents: string[];
	extraEvents: string[];
} {
	const expectedEvents = [
		"prompt_accepted",
		"status",
		"executing",
		"progress_state",
		"executed",
		"execution_success",
	];

	const receivedEventTypes = events.map((e) => (e as { type: string }).type);
	const missingEvents = expectedEvents.filter(
		(eventType) => !receivedEventTypes.includes(eventType),
	);
	const extraEvents = receivedEventTypes.filter(
		(eventType) => !expectedEvents.includes(eventType),
	);

	return {
		extraEvents,
		missingEvents,
		valid: missingEvents.length === 0,
	};
}

/**
 * Format duration in milliseconds to human readable format
 */
export function formatDuration(ms: number): string {
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
 * Validate that the image base64 is valid
 */
export function isValidImageData(data: string | ArrayBuffer): boolean {
	if (typeof data === "string") {
		return data.startsWith("data:image/");
	}
	// For ArrayBuffer, consider any non-empty buffer as valid image data for test purposes
	return data.byteLength > 0;
}

/**
 * Test result interface
 */
export interface TestResult {
	name: string;
	success: boolean;
	duration: number;
	error?: string;
	details?: unknown;
}

/**
 * Run a test and return formatted result
 */
export async function runTest(
	name: string,
	testFn: () => Promise<void>,
): Promise<TestResult> {
	const start = performance.now();
	try {
		await testFn();
		const duration = performance.now() - start;
		return {
			duration,
			name,
			success: true,
		};
	} catch (error) {
		const duration = performance.now() - start;
		return {
			duration,
			error: error instanceof Error ? error.message : String(error),
			name,
			success: false,
		};
	}
}

/**
 * Print test results in a formatted way
 * Returns true if all tests passed, false otherwise
 */
export function printTestResults(results: TestResult[]): boolean {
	console.log("\n📊 Test Results:");
	console.log("================");

	let passed = 0;
	let failed = 0;

	for (const result of results) {
		const status = result.success ? "✅" : "❌";
		const duration = formatDuration(result.duration);
		console.log(`${status} ${result.name} (${duration})`);

		if (!result.success && result.error) {
			console.log(`   Error: ${result.error}`);
		}

		if (result.success) {
			passed++;
		} else {
			failed++;
		}
	}

	console.log("================");
	console.log(`✅ Passed: ${passed}`);
	console.log(`❌ Failed: ${failed}`);
	console.log(`📈 Total: ${results.length}`);

	// Return true if all tests passed, false otherwise
	// The caller handles process.exit() to ensure proper cleanup
	return failed === 0;
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
