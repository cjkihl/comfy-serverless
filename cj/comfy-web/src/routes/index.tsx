import type { ComfyPrompt } from "@cj/comfy-client";
import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { generateUniqueUserId } from "../../tests/utils-test";
import { ConnectionForm } from "../components/ConnectionForm";
import { useImageGallery } from "../components/ImageGallery";
import { LogViewer } from "../components/LogViewer";
import { PromptTester } from "../components/PromptTester";
import { StatusDisplay } from "../components/StatusDisplay";
import type { ConnectionConfig } from "../hooks/useComfyClient";
import { useComfyClient } from "../hooks/useComfyClient";
import { generateJWTToken } from "../shared/jwt-server";
import { UILogger } from "../shared/logger";

class PerformanceTracker {
	private phases: Array<{
		name: string;
		startTime: number;
		duration?: number;
	}> = [];

	startPhase(name: string) {
		this.phases.push({ name, startTime: Date.now() });
	}

	endPhase(name: string) {
		const phase = this.phases.find(
			(p) => p.name === name && p.duration === undefined,
		);
		if (phase) {
			phase.duration = Date.now() - phase.startTime;
		}
	}

	getMetrics() {
		return {
			phases: this.phases.filter((p) => p.duration !== undefined),
		};
	}
}

export const Route = createFileRoute("/")({
	component: ManualTestingPage,
});

// Initialize shared logger instance
const logger = new UILogger();

function ManualTestingPage() {
	const [config, setConfig] = useState<ConnectionConfig>({
		autoConnect: false,
		jwtToken: "", // Will be generated on mount
		url: "ws://localhost:8190/ws",
	});

	const trackerRef = useRef<PerformanceTracker | null>(null);

	// Generate JWT token on component mount
	useEffect(() => {
		const generateJWT = async () => {
			try {
				const userId = generateUniqueUserId("manual-test");
				logger.info(`🔑 Generating JWT for user: ${userId}`);
				const token = await generateJWTToken({ data: { userId } });
				setConfig((prev) => ({ ...prev, jwtToken: token }));
				logger.success("✅ JWT token generated successfully");
			} catch (error) {
				const errorMessage =
					error instanceof Error ? error.message : String(error);
				logger.error(`❌ Failed to generate JWT: ${errorMessage}`);
			}
		};

		generateJWT();
	}, []);

	// Initialize Image Gallery hook
	const { ImageGalleryComponent, displayImage } = useImageGallery();

	// Initialize ComfyClient hook
	const client = useComfyClient({
		autoConnect: false,
		logger,
		onImage: displayImage, // Pass image handler to ComfyClient
	});

	const handleConfigChange = (newConfig: ConnectionConfig) => {
		setConfig(newConfig);
	};

	const handleConnect = async () => {
		trackerRef.current = new PerformanceTracker();
		trackerRef.current.startPhase("Connection Open");

		await client.connect(config);

		if (trackerRef.current) {
			trackerRef.current.endPhase("Connection Open");
		}
	};

	const handleSubmitPrompt = async (prompt: ComfyPrompt) => {
		try {
			trackerRef.current?.startPhase("Prompt Submission");

			await client.submitPrompt(prompt);

			trackerRef.current?.endPhase("Prompt Submission");

			// Display performance metrics
			if (trackerRef.current) {
				const metrics = trackerRef.current.getMetrics();
				logger.info("⏱️ Performance Metrics:");
				metrics.phases.forEach((phase: { name: string; duration?: number }) => {
					const duration = ((phase.duration ?? 0) / 1000).toFixed(2);
					logger.info(`  - ${phase.name}: ${duration}s`);
				});
			}
		} catch (error) {
			const errorMessage =
				error instanceof Error ? error.message : String(error);
			logger.error(`❌ Error: ${errorMessage}`);
		}
	};

	// Initialize
	useEffect(() => {
		logger.success("🚀 ComfyClient Manual Test loaded");
		logger.info("💡 Enter your WebSocket URL and click Connect to get started");
	}, []);

	return (
		<div className="min-h-screen bg-linear-to-br from-indigo-500 to-purple-600 p-5">
			<div className="max-w-6xl mx-auto bg-white rounded-xl shadow-2xl overflow-hidden">
				<header className="bg-linear-to-r from-indigo-500 to-purple-600 text-white p-8">
					<h1 className="text-3xl font-bold mb-3">
						🎨 ComfyClient Manual Test
					</h1>
					<StatusDisplay status={client.status} />
				</header>

				<div className="p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
					<div className="space-y-8">
						<ConnectionForm
							config={config}
							isConnected={client.isConnected}
							isConnecting={client.status === "connecting"}
							onConfigChange={handleConfigChange}
							onConnect={handleConnect}
							onDisconnect={client.disconnect}
						/>

						<PromptTester
							canSubmit={client.isConnected}
							onPing={client.ping}
							onSubmitPrompt={handleSubmitPrompt}
						/>
					</div>

					<div>
						<LogViewer logger={logger} />
					</div>

					<div className="lg:col-span-2">
						<ImageGalleryComponent />
					</div>
				</div>
			</div>
		</div>
	);
}
