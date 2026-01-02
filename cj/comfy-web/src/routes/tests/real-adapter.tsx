import { ComfyClient } from "@cj/comfy-client";
import { UniversalWebSocketAdapter } from "@cj/comfy-client/adapters/universal";
import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { generateUniqueUserId } from "../../../tests/utils-test";
import { generateTestJWT } from "../../shared/jwt";
import { Logger } from "../../shared/logger";
import { getLightTestPrompt, getTestPrompt } from "../../shared/test-prompt";
import { loadImageAsBase64, TEST_IMAGE_URL } from "../../shared/utils-web";

const logger = new Logger("info", "[browser-test]");

export const Route = createFileRoute("/tests/real-adapter")({
	component: RealAdapterTestPage,
});

function RealAdapterTestPage() {
	const [status, setStatus] = useState<string>("starting");
	const [result, setResult] = useState<"pending" | "passed" | "failed">(
		"pending",
	);
	const [generatedImage, setGeneratedImage] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;

		const run = async () => {
			setStatus("running");
			const startTime = performance.now();

			const defaultUrl = "ws://localhost:8190/ws";
			const search = new URLSearchParams(window.location.search);
			const proxyUrl = search.get("url") || defaultUrl;
			const jwt = search.get("jwt") || "";

			try {
				// 1. Load image from URL and convert to base64
				logger.info("Loading test image from URL...");
				const imageBase64 = await loadImageAsBase64(TEST_IMAGE_URL);
				logger.info("✅ Image loaded and converted to base64");

				// 2. Create client and connect
				const userId = generateUniqueUserId("full-test-browser");
				const clientJWT = jwt || generateTestJWT(userId);

				const client = new ComfyClient({
					adapter: new UniversalWebSocketAdapter(proxyUrl),
					auth: { jwt: clientJWT },
					autoConnect: false,
					logging: { level: "info" },
					timeout: { connect: 10000, message: 30000, operation: 60000 }, // 60 seconds max for browser
					url: proxyUrl,
				});

				logger.info("📡 Connecting to proxy...");
				const connectResult = await client.connect();
				if (!connectResult.success) {
					throw new Error(`Connection failed: ${connectResult.error}`);
				}
				logger.info("✅ Connected successfully");

				// 3. Get test prompt with loaded image
				logger.info("📤 Preparing test prompt...");
				const prompt =
					import.meta.env.VITE_LIGHT_PROMPT === "true"
						? getLightTestPrompt(imageBase64)
						: getTestPrompt(imageBase64);
				logger.info(
					`✅ Prompt prepared with ${Object.keys(prompt).length} nodes`,
				);

				// 4. Submit prompt
				logger.info("📤 Submitting prompt...");
				const submitResult = await client.submitPrompt(prompt, {
					promptId: `full-test-browser-${Date.now()}`,
				});
				if (!submitResult.success) {
					throw new Error(`Prompt submission failed: ${submitResult.error}`);
				}
				logger.info("✅ Prompt submitted and accepted");

				// 5. Collect events and wait for base64 result
				logger.info("📥 Collecting events (waiting for base64 result)...");
				const collectResult = await client.collectAllEvents({
					timeout: 60000, // 60 seconds max (browser may be slower)
					waitForCompletion: true,
				});

				if (!collectResult.success) {
					throw new Error(`Event collection failed: ${collectResult.error}`);
				}

				const { events, completed, error } = collectResult.data;

				// Debug: log received events
				logger.info(`📊 Received ${events.length} events`);
				const eventTypes = new Set<string>();
				for (const event of events) {
					if (typeof event === "object" && event !== null && "type" in event) {
						const eventType = (event as { type: string }).type;
						eventTypes.add(eventType);
						if (eventTypes.size <= 15) {
							logger.debug(`  - Event type: ${eventType}`);
						}
					}
				}
				if (events.length > 15) {
					logger.debug(`  ... and ${events.length - 15} more events`);
				}

				if (error) {
					logger.error(`❌ Execution error: ${error}`);
					logger.error(`Events received: ${Array.from(eventTypes).join(", ")}`);
					throw new Error(`Execution error: ${error}`);
				}
				if (!completed) {
					logger.warn(
						"⚠️  Execution did not complete, but continuing to check for image...",
					);
					logger.debug(`Events received: ${Array.from(eventTypes).join(", ")}`);
					// Continue anyway to see if we got the image
				}

				// 6. Extract base64 image from SaveImageBase64 node output
				let base64Image: string | null = null;
				for (const event of events) {
					if (
						typeof event === "object" &&
						event !== null &&
						"type" in event &&
						event.type === "executed"
					) {
						const typedEvent = event as { type: string; data?: unknown };
						const data = typedEvent.data as
							| {
									output?: { result?: Array<{ image?: string }> };
							  }
							| undefined;
						if (data?.output?.result && Array.isArray(data.output.result)) {
							for (const item of data.output.result) {
								if (item.image && typeof item.image === "string") {
									base64Image = item.image;
									break;
								}
							}
						}
						if (base64Image) break;
					}
				}

				if (!base64Image) {
					throw new Error("No base64 image found in SaveImageBase64 output");
				}

				logger.info("✅ Base64 image received");

				// Normalize base64 image for display (ensure it has data: prefix)
				let imageDataUrl = base64Image;
				if (!base64Image.startsWith("data:")) {
					imageDataUrl = `data:image/webp;base64,${base64Image}`;
				}

				if (cancelled) return;
				setGeneratedImage(imageDataUrl);

				// 7. Assert completion time < 60 seconds (actual execution may vary, browser can be slower)
				const duration = performance.now() - startTime;
				logger.info(
					`⏱️  Total execution time: ${(duration / 1000).toFixed(2)}s`,
				);
				if (duration >= 60000) {
					throw new Error(
						`Test took ${(duration / 1000).toFixed(2)}s, expected < 60s`,
					);
				}
				if (duration < 10000) {
					logger.info("✅ Test completed successfully in less than 10 seconds");
				} else if (duration < 30000) {
					logger.warn(
						`⚠️  Test completed in ${(duration / 1000).toFixed(2)}s (target was < 10s, but within 30s limit)`,
					);
				} else {
					logger.warn(
						`⚠️  Test completed in ${(duration / 1000).toFixed(2)}s (slower than expected, but within 60s limit)`,
					);
				}

				if (cancelled) return;
				setResult("passed");
				setStatus("finished");
				window.postMessage({ passed: true, type: "suite-finished" }, "*");

				// Re-broadcast a few times to catch late listeners
				let count = 0;
				const resend = setInterval(() => {
					count++;
					window.postMessage({ passed: true, type: "suite-finished" }, "*");
					if (count >= 50) clearInterval(resend);
				}, 100);
				logger.debug("suite-finished", "true");

				client.disconnect();
			} catch (err) {
				if (cancelled) return;
				const duration = performance.now() - startTime;
				const errorMsg = err instanceof Error ? err.message : String(err);
				logger.error(
					`❌ Test failed after ${(duration / 1000).toFixed(2)}s:`,
					errorMsg,
				);
				setResult("failed");
				setStatus(errorMsg);
				window.postMessage(
					{ error: errorMsg, passed: false, type: "suite-finished" },
					"*",
				);
				let count = 0;
				const resend = setInterval(() => {
					count++;
					window.postMessage(
						{ error: errorMsg, passed: false, type: "suite-finished" },
						"*",
					);
					if (count >= 50) clearInterval(resend);
				}, 100);
				logger.debug("suite-finished", "false");
			}
		};

		run();
		return () => {
			cancelled = true;
		};
	}, []);

	return (
		<div style={{ padding: 16 }}>
			<h1>Real Adapter Browser Test</h1>
			<p>Status: {status}</p>
			<p>Result: {result}</p>
			{generatedImage && (
				<div style={{ marginTop: 16 }}>
					<h2>Generated Image</h2>
					<img
						alt="Generated result"
						src={generatedImage}
						style={{
							border: "1px solid #ccc",
							borderRadius: 4,
							height: "auto",
							maxWidth: "100%",
						}}
					/>
				</div>
			)}
		</div>
	);
}
