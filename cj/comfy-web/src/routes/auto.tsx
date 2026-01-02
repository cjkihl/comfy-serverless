import { createFileRoute } from "@tanstack/react-router";
import type React from "react";
import { useEffect, useState } from "react";
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

export const Route = createFileRoute("/auto")({
	component: AutoTestingPage,
});

// Initialize shared logger instance
const logger = new UILogger();

function AutoTestingPage(): React.ReactElement {
	const [config, setConfig] = useState<ConnectionConfig>({
		autoConnect: true,
		jwtToken: "", // Will be generated on mount
		url: "ws://localhost:8190/ws", // Default ComfyUI WebSocket URL
	});

	// Generate JWT token on component mount
	useEffect(() => {
		const generateJWT = async () => {
			try {
				const userId = generateUniqueUserId("auto-test");
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

	// Initialize ComfyClient hook with autoConnect enabled and config
	const client = useComfyClient({
		autoConnect: true,
		config, // Pass config so ComfyClient can autoConnect
		logger,
		onImage: displayImage, // Pass image handler to ComfyClient
	});

	const handleConfigChange = (newConfig: ConnectionConfig) => {
		setConfig(newConfig);
	};

	const handleManualRetry = () => {
		logger.info("🔄 Manual retry initiated - resetting connection...");
		// Connect directly - the hook will handle cleanup and reset retry state
		client.connect(config);
	};

	// Auto-connect is handled by ComfyClient's built-in autoConnect feature
	// No manual connection management needed here

	return (
		<div className="min-h-screen bg-linear-to-br from-indigo-500 to-purple-600 p-5">
			<div className="max-w-6xl mx-auto bg-white rounded-xl shadow-2xl overflow-hidden">
				<header
					className="bg-linear-to-r from-indigo-500 to-purple-600 text-white p-8"
					data-status={client.status}
					id="status-container"
				>
					<h1 className="text-3xl font-bold mb-3">
						🔄 ComfyClient AutoConnect Test
					</h1>
					<StatusDisplay status={client.status} />
				</header>

				<div className="p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
					<div className="space-y-8">
						<ConnectionForm
							autoConnect={true}
							config={config}
							isConnected={client.isConnected}
							isConnecting={client.status === "connecting"}
							onConfigChange={handleConfigChange}
							onConnect={() => client.connect(config)}
							onDisconnect={client.disconnect}
						/>

						{/* Manual retry section for when autoConnect exhausts retries */}
						{client.status === "disconnected" &&
							!client.isConnected &&
							client.retryState.attempts >= client.retryState.maxAttempts &&
							!client.retryState.isRetrying && (
								<div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
									<h3 className="text-lg font-semibold text-amber-800 mb-2">
										🔄 Connection Failed
									</h3>
									<p className="text-amber-700 mb-3">
										AutoConnect has exhausted all{" "}
										{client.retryState.maxAttempts} retry attempts. You can
										manually retry the connection or modify the settings above.
									</p>
									<button
										className="px-4 py-2 bg-amber-500 text-white font-semibold rounded-lg transition-all hover:bg-amber-600"
										id="retry-connection"
										onClick={handleManualRetry}
										type="button"
									>
										🔄 Retry Connection
									</button>
								</div>
							)}

						{/* Show retry progress when autoConnect is retrying */}
						{client.retryState.isRetrying &&
							client.status === "disconnected" && (
								<div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
									<h3 className="text-lg font-semibold text-blue-800 mb-2">
										🔄 Auto-Retry in Progress
									</h3>
									<p className="text-blue-700">
										Attempt {client.retryState.attempts}/
										{client.retryState.maxAttempts} - Next retry coming up...
									</p>
								</div>
							)}

						<PromptTester
							canSubmit={client.isConnected}
							onPing={client.ping}
							onSubmitPrompt={client.submitPrompt}
						/>
					</div>

					<div>
						<LogViewer logger={logger} />
					</div>

					<div className="lg:col-span-2" id="image-gallery-container">
						<ImageGalleryComponent />
					</div>
				</div>
			</div>
		</div>
	);
}
