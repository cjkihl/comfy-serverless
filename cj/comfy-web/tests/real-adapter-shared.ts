/**
 * Shared, environment-agnostic real adapter tests.
 * Both Bun and Browser runners consume these helpers by providing an Env.
 */

import type { ComfyClient, ComfyPrompt } from "@cj/comfy-client";
import { generateTestJWT } from "../src/shared/jwt";
import { getLightTestPrompt, getTestPrompt } from "../src/shared/test-prompt";
import {
	formatDuration,
	generateUniqueUserId,
	isValidImageData,
	printTestResults,
	runTest,
	validateEventSequence,
} from "./utils-test";

/**
 * Check if light prompt should be used based on environment variable
 * Works in both Node/Bun (process.env) and browser (import.meta.env) contexts
 */
function shouldUseLightPrompt(): boolean {
	if (typeof process !== "undefined" && process.env) {
		return process.env.VITE_LIGHT_PROMPT === "true";
	}
	if (typeof import.meta !== "undefined" && import.meta.env) {
		return import.meta.env.VITE_LIGHT_PROMPT === "true";
	}
	return false;
}

export interface Env {
	proxyUrl: string;
	log: (...args: unknown[]) => void;
	loadTestImageBase64: () => Promise<string>;
	saveImageArtifact: (
		name: string,
		data: ArrayBuffer | string,
		mime: string,
	) => Promise<void>;
	createComfyClient: (opts: unknown) => ComfyClient;
}

export async function testAdapterConnectivity(
	env: Env,
	name: string,
): Promise<void> {
	const jwt = generateTestJWT(
		`test-${name.toLowerCase().replace(/\s+/g, "-")}`,
	);
	const client = env.createComfyClient({
		auth: { jwt },
		autoConnect: false,
		url: env.proxyUrl,
	});
	const res = await client.connect();
	if (!res.success) throw new Error(`${name} connection failed: ${res.error}`);
	client.disconnect();
}

export async function checkPrerequisites(env: Env): Promise<void> {
	env.log("🔍 Checking prerequisites...");
	const httpUrl = env.proxyUrl
		.replace("ws://", "http://")
		.replace("wss://", "https://");

	const healthUrl = `${httpUrl.split("/ws")[0]}/health`;
	env.log(`📡 Checking proxy health at ${healthUrl}...`);
	const healthResp = await fetch(healthUrl);
	if (!healthResp.ok)
		throw new Error(
			`Proxy health check failed: ${healthResp.status} ${healthResp.statusText}`,
		);
	env.log(`✅ Proxy is healthy: ${JSON.stringify(await healthResp.json())}`);

	const readyUrl = `${httpUrl.split("/ws")[0]}/ready`;
	env.log(`📡 Checking proxy ready status at ${readyUrl}...`);
	const readyResp = await fetch(readyUrl);
	if (!readyResp.ok)
		throw new Error(
			`Proxy ready check failed: ${readyResp.status} ${readyResp.statusText}`,
		);
	env.log(`✅ Proxy is ready: ${JSON.stringify(await readyResp.json())}`);
}

export async function testSingleUserWorkflow(env: Env): Promise<void> {
	const userId = generateUniqueUserId("integration");
	const jwt = generateTestJWT(userId);

	const client = env.createComfyClient({
		auth: { jwt },
		autoConnect: false,
		logging: { level: "debug" },
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.proxyUrl,
	});

	try {
		const connectResult = await client.connect();
		if (!connectResult.success)
			throw new Error(`Connection failed: ${connectResult.error}`);

		const imageBase64 = await env.loadTestImageBase64();
		const prompt: ComfyPrompt = shouldUseLightPrompt()
			? getLightTestPrompt(imageBase64)
			: getTestPrompt(imageBase64);

		const submitResult = await client.submitPrompt(prompt, {
			extraData: { test: "integration", timestamp: Date.now(), userId },
			promptId: `integration-${Date.now()}`,
		});
		if (!submitResult.success)
			throw new Error(`Prompt submission failed: ${submitResult.error}`);

		const collectResult = await client.collectAllEvents({
			timeout: 120000,
			waitForCompletion: true,
		});
		if (!collectResult.success)
			throw new Error(`Event collection failed: ${collectResult.error}`);

		const { events, binaryData, completed, error } = collectResult.data;
		if (error) throw new Error(`Execution error: ${error}`);
		if (!completed) throw new Error("Execution did not complete");

		const validation = validateEventSequence(events);
		if (!validation.valid)
			env.log(
				`⚠️  Event sequence validation failed: missing ${validation.missingEvents.join(", ")}`,
			);

		let validImages = 0;
		for (const data of binaryData as ArrayBuffer[])
			if (isValidImageData(data)) validImages++;
		if (validImages === 0) env.log("⚠️  No valid images found");

		await saveTestImages(
			env,
			events,
			binaryData as ArrayBuffer[],
			"integration",
		);
	} finally {
		client.disconnect();
	}
}

export async function testConcurrentUsers(env: Env): Promise<void> {
	const userCount = 3;
	await Promise.all(
		Array.from({ length: userCount }, (_, i) => runConcurrentUser(env, i)),
	);
}

async function runConcurrentUser(env: Env, userIndex: number): Promise<void> {
	const userId = generateUniqueUserId(`concurrent-${userIndex}`);
	const jwt = generateTestJWT(userId);

	const client = env.createComfyClient({
		auth: { jwt },
		autoConnect: false,
		logging: { level: "debug" },
		reconnect: { enabled: true, maxRetries: 2 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.proxyUrl,
	});

	try {
		const connectResult = await client.connect();
		if (!connectResult.success)
			throw new Error(
				`User ${userIndex} connection failed: ${connectResult.error}`,
			);

		const imageBase64 = await env.loadTestImageBase64();
		const prompt = shouldUseLightPrompt()
			? getLightTestPrompt(imageBase64)
			: getTestPrompt(imageBase64);
		const submitResult = await client.submitPrompt(prompt, {
			promptId: `concurrent-${userIndex}-${Date.now()}`,
		});
		if (!submitResult.success)
			throw new Error(
				`User ${userIndex} prompt submission failed: ${submitResult.error}`,
			);
	} finally {
		client.disconnect();
		await new Promise((r) => setTimeout(r, 100));
	}
}

export async function testPerformanceBenchmark(env: Env): Promise<void> {
	const iterations = 3;
	const results: number[] = [];
	for (let i = 0; i < iterations; i++) {
		const userId = generateUniqueUserId(`perf-${i}`);
		const jwt = generateTestJWT(userId);

		const client = env.createComfyClient({
			auth: { jwt },
			autoConnect: false,
			logging: { level: "silent" },
			timeout: { connect: 10000, message: 30000, operation: 120000 },
			url: env.proxyUrl,
		});

		const start = performance.now();
		try {
			const connectResult = await client.connect();
			if (!connectResult.success)
				throw new Error(`Connection failed: ${connectResult.error}`);

			const imageBase64 = await env.loadTestImageBase64();
			const prompt = shouldUseLightPrompt()
				? getLightTestPrompt(imageBase64)
				: getTestPrompt(imageBase64);
			const submitResult = await client.submitPrompt(prompt, {
				promptId: `perf-${i}-${Date.now()}`,
			});
			if (!submitResult.success)
				throw new Error(`Prompt submission failed: ${submitResult.error}`);

			const collectResult = await client.collectAllEvents({
				timeout: 120000,
				waitForCompletion: true,
			});
			if (!collectResult.success)
				throw new Error(`Event collection failed: ${collectResult.error}`);

			const duration = performance.now() - start;
			results.push(duration);
			env.log(`📊 Iteration ${i + 1}: ${formatDuration(duration)}`);
		} finally {
			client.disconnect();
			await new Promise((r) => setTimeout(r, 200));
		}
	}
	if (results.length > 0) {
		const avg = results.reduce((a, b) => a + b, 0) / results.length;
		env.log(`📈 Performance Results: Avg=${formatDuration(avg)}`);
	} else {
		env.log("📈 Performance Results: No data collected");
	}
}

export async function testMultipleConnectionsSameUser(env: Env): Promise<void> {
	const userId = "multi-connection-user";
	const jwt = generateTestJWT(userId);

	const clients: Array<{ disconnect: () => void }> = [];
	const MAX_CONNECTIONS = 3;
	try {
		for (let i = 0; i < MAX_CONNECTIONS; i++) {
			const client = env.createComfyClient({
				auth: { jwt },
				autoConnect: false,
				logging: { level: "silent" },
				url: env.proxyUrl,
			});
			const connectResult = await client.connect();
			if (!connectResult.success)
				throw new Error(
					`Client ${i + 1} connection failed: ${connectResult.error}`,
				);
			clients.push(client);
		}
	} finally {
		for (const c of clients) {
			c.disconnect();
		}
	}
}

export async function testSinglePromptExecution(env: Env): Promise<void> {
	const userId = generateUniqueUserId("single-prompt-execution");
	const jwt = generateTestJWT(userId);

	const client = env.createComfyClient({
		auth: { jwt },
		autoConnect: false,
		logging: { level: "debug" },
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.proxyUrl,
	});

	try {
		const connectResult = await client.connect();
		if (!connectResult.success)
			throw new Error(`Connection failed: ${connectResult.error}`);

		const imageBase64 = await env.loadTestImageBase64();
		const prompt = shouldUseLightPrompt()
			? getLightTestPrompt(imageBase64)
			: getTestPrompt(imageBase64);
		const submitResult = await client.submitPrompt(prompt, {
			extraData: {
				test: "single-prompt-execution",
				timestamp: Date.now(),
				userId,
			},
			promptId: `single-prompt-execution-${Date.now()}`,
		});
		if (!submitResult.success)
			throw new Error(`Prompt submission failed: ${submitResult.error}`);

		const collectResult = await client.collectAllEvents({
			timeout: 120000,
			waitForCompletion: true,
		});
		if (!collectResult.success)
			throw new Error(`Event collection failed: ${collectResult.error}`);

		const { events, binaryData, completed, error } = collectResult.data;
		if (error) throw new Error(`Execution error: ${error}`);
		if (!completed) throw new Error("Execution did not complete");

		const validation = validateEventSequence(events);
		if (!validation.valid)
			env.log(
				`⚠️  Event sequence validation failed: missing ${validation.missingEvents.join(", ")}`,
			);

		let validImages = 0;
		for (const data of binaryData as ArrayBuffer[])
			if (isValidImageData(data)) validImages++;
		if (validImages === 0) env.log("⚠️  No valid images found");

		await saveTestImages(
			env,
			events,
			binaryData as ArrayBuffer[],
			"single-prompt-execution",
		);
	} finally {
		client.disconnect();
	}
}

export async function testPromptWithPerformanceMeasure(
	env: Env,
): Promise<void> {
	const userId = generateUniqueUserId("prompt-with-performance-measure");
	const jwt = generateTestJWT(userId);

	const client = env.createComfyClient({
		auth: { jwt },
		autoConnect: false,
		logging: { level: "debug" },
		reconnect: { enabled: true, maxRetries: 3 },
		timeout: { connect: 10000, message: 30000, operation: 120000 },
		url: env.proxyUrl,
	});

	try {
		const connectResult = await client.connect();
		if (!connectResult.success)
			throw new Error(`Connection failed: ${connectResult.error}`);

		const imageBase64 = await env.loadTestImageBase64();
		const prompt = shouldUseLightPrompt()
			? getLightTestPrompt(imageBase64)
			: getTestPrompt(imageBase64);
		const submitResult = await client.submitPrompt(prompt, {
			extraData: {
				test: "prompt-with-performance-measure",
				timestamp: Date.now(),
				userId,
			},
			promptId: `prompt-with-performance-measure-${Date.now()}`,
		});
		if (!submitResult.success)
			throw new Error(`Prompt submission failed: ${submitResult.error}`);

		const collectResult = await client.collectAllEvents({
			timeout: 120000,
			waitForCompletion: true,
		});
		if (!collectResult.success)
			throw new Error(`Event collection failed: ${collectResult.error}`);

		const { events, binaryData, completed, error } = collectResult.data;
		if (error) throw new Error(`Execution error: ${error}`);
		if (!completed) throw new Error("Execution did not complete");

		const validation = validateEventSequence(events);
		if (!validation.valid)
			env.log(
				`⚠️  Event sequence validation failed: missing ${validation.missingEvents.join(", ")}`,
			);

		let validImages = 0;
		for (const data of binaryData as ArrayBuffer[])
			if (isValidImageData(data)) validImages++;
		if (validImages === 0) env.log("⚠️  No valid images found");

		await saveTestImages(
			env,
			events,
			binaryData as ArrayBuffer[],
			"prompt-with-performance-measure",
		);
	} finally {
		client.disconnect();
	}
}

async function saveTestImages(
	env: Env,
	events: unknown[],
	binaryData: ArrayBuffer[],
	testName: string,
): Promise<void> {
	const timestamp = Date.now();
	let imageIndex = 1;

	for (const data of binaryData) {
		if (isValidImageData(data)) {
			await env.saveImageArtifact(
				`${testName}-${timestamp}-${imageIndex}`,
				data,
				"webp",
			);
			imageIndex++;
		}
	}

	for (const event of events) {
		if (typeof event === "object" && event !== null && "type" in event) {
			const typedEvent = event as { type: string; data?: unknown };
			if (typedEvent.type === "executed" && typedEvent.data) {
				const data = typedEvent.data as { output?: { result?: unknown[] } };
				if (data.output?.result && Array.isArray(data.output.result)) {
					for (const item of data.output.result) {
						if (typeof item === "object" && item !== null && "image" in item) {
							const imageData = (item as { image?: string }).image;
							if (typeof imageData === "string") {
								let base64Data = imageData;
								let mimeType = "webp";
								if (imageData.startsWith("data:")) {
									const match = imageData.match(
										/^data:image\/(\w+);base64,(.+)$/,
									);
									if (match?.[1] && match[2]) {
										mimeType = match[1]!;
										base64Data = match[2]!;
									}
								}
								await env.saveImageArtifact(
									`${testName}-${timestamp}-${imageIndex}`,
									base64Data,
									mimeType,
								);
								imageIndex++;
							}
						}
					}
				}
			}
		}
	}
}

export async function runSuite(env: Env): Promise<{ resultsPassed: boolean }> {
	await checkPrerequisites(env);
	const tests = [
		runTest("Adapter Connectivity", () =>
			testAdapterConnectivity(env, "Adapter"),
		),
		runTest("Single User Workflow", () => testSingleUserWorkflow(env)),
		runTest("Concurrent Users", () => testConcurrentUsers(env)),
		runTest("Performance Benchmark", () => testPerformanceBenchmark(env)),
		runTest("Multiple Connections Same User", () =>
			testMultipleConnectionsSameUser(env),
		),
		runTest("Single Prompt Execution", () => testSinglePromptExecution(env)),
		runTest("Prompt with Performance Measure", () =>
			testPromptWithPerformanceMeasure(env),
		),
	];

	const results = await Promise.all(tests);
	const allPassed = printTestResults(results);
	return { resultsPassed: allPassed };
}
