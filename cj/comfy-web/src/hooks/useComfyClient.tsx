import type { ComfyPrompt, ConnectionState } from "@cj/comfy-client";
import { ComfyClient } from "@cj/comfy-client";
import { UniversalWebSocketAdapter } from "@cj/comfy-client/adapters/universal";
import { extractImagesFromMessage } from "@cj/comfy-client/utils";
import { useCallback, useEffect, useRef, useState } from "react";
import type { UILogger } from "../shared/logger";

export interface ConnectionConfig {
	url: string;
	jwtToken?: string;
	autoConnect?: boolean;
}

interface UseComfyClientOptions {
	logger: UILogger;
	autoConnect?: boolean;
	config?: ConnectionConfig; // Config for autoConnect mode
	onImage?: (imageData: string) => void; // Callback for handling image data
}

interface RetryState {
	attempts: number;
	maxAttempts: number;
	isRetrying: boolean;
}

interface UseComfyClientReturn {
	client: ComfyClient | null;
	status: ConnectionState;
	isConnected: boolean;
	connect: (config: ConnectionConfig) => Promise<void>;
	disconnect: () => void;
	submitPrompt: (prompt: ComfyPrompt) => Promise<void>;
	ping: () => Promise<void>;
	retryState: RetryState;
}

export const useComfyClient = ({
	logger,
	autoConnect = false,
	config,
	onImage,
}: UseComfyClientOptions): UseComfyClientReturn => {
	const [client, setClient] = useState<ComfyClient | null>(null);
	const [status, setStatus] = useState<ConnectionState>("disconnected");
	const [retryState, setRetryState] = useState<RetryState>({
		attempts: 0,
		isRetrying: false,
		maxAttempts: 3,
	});
	const clientRef = useRef<ComfyClient | null>(null);
	const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

	const isConnected = status === "connected";

	// Helper function to create and connect a ComfyClient
	const createAndConnectClient = useCallback(
		async (config: ConnectionConfig, isAutoRetry = false) => {
			try {
				if (!isAutoRetry) {
					logger.info("🔗 Connecting to ComfyUI...");
				} else {
					logger.info(
						`🔄 Auto-retry attempt ${retryState.attempts + 1}/${retryState.maxAttempts}...`,
					);
				}
				setStatus("connecting");

				const adapter = new UniversalWebSocketAdapter(config.url);

				const newClient = new ComfyClient({
					adapter,
					auth: { jwt: config.jwtToken || "" },
					autoConnect: false, // Disable ComfyClient's autoConnect - we manage retries
					logging: { level: "debug" },
					onConnectionChange: (state) => {
						logger.info(`🔌 Connection state: ${state}`);
						if (state === "connected") {
							setStatus("connected");
							// Reset retry state on successful connection
							setRetryState((prev) => ({
								...prev,
								attempts: 0,
								isRetrying: false,
							}));
							logger.success("✅ Connected successfully!");
						} else if (state === "disconnected") {
							setStatus("disconnected");
						} else if (state === "reconnecting") {
							setStatus("reconnecting");
						}
					},
					onError: (err) => {
						logger.error(`❌ Connection error: ${err.message}`);
					},
					onMessage: (msg) => {
						// Extract images from message using client utility
						const images = extractImagesFromMessage(msg);
						if (images.length > 0 && onImage) {
							for (const imageData of images) {
								onImage(imageData);
								logger.success("🖼️ Image received and displayed in gallery");
							}
						}

						// Log message (truncate large messages)
						if (
							typeof msg === "object" &&
							msg !== null &&
							"type" in msg &&
							JSON.stringify(msg).length > 10000
						) {
							logger.info(
								`📨 Message: ${(msg as { type?: string }).type} (large data - ${JSON.stringify(msg).length} chars)`,
							);
						} else {
							logger.info(`📨 Message: ${JSON.stringify(msg)}`);
						}
					},
					reconnect: { enabled: false }, // Disable ComfyClient's reconnect - we handle it
					url: config.url,
				});

				const result = await newClient.connect();
				if (!result.success) {
					throw new Error(result.error.message);
				}

				setClient(newClient);
				clientRef.current = newClient;
				return true;
			} catch (error) {
				const errorMsg =
					error instanceof Error ? error.message : "Unknown error";
				if (!isAutoRetry) {
					logger.error(`❌ Connection failed: ${errorMsg}`);
				} else {
					logger.error(`❌ Auto-retry failed: ${errorMsg}`);
				}
				setStatus("disconnected");
				return false;
			}
		},
		[logger, retryState.attempts, retryState.maxAttempts, onImage],
	);

	const connect = useCallback(
		async (config: ConnectionConfig) => {
			// Reset retry state for manual connections
			setRetryState((prev) => ({ ...prev, attempts: 0, isRetrying: false }));
			await createAndConnectClient(config, false);
		},
		[createAndConnectClient],
	);

	const disconnect = useCallback(() => {
		// Clear any pending retry timeouts
		if (retryTimeoutRef.current) {
			clearTimeout(retryTimeoutRef.current);
			retryTimeoutRef.current = null;
		}

		// Reset retry state
		setRetryState((prev) => ({ ...prev, attempts: 0, isRetrying: false }));

		if (clientRef.current) {
			// Only log disconnect messages if actually connected
			if (isConnected) {
				logger.info("🧹 Disconnecting...");
			} else {
				logger.info("🧹 Cleaning up client instance...");
			}

			clientRef.current.disconnect();
			setClient(null);
			clientRef.current = null;
			setStatus("disconnected");

			// Only log success message if we were actually connected
			if (isConnected) {
				logger.success("👋 Disconnected");
			} else {
				logger.info("🧹 Client cleanup completed");
			}
		}
	}, [isConnected, logger]);

	const submitPrompt = useCallback(
		async (prompt: ComfyPrompt) => {
			if (!clientRef.current || !isConnected) {
				logger.error("❌ Not connected");
				return;
			}

			try {
				logger.info("📤 Submitting prompt...");

				const result = await clientRef.current.submitPrompt(prompt, {
					promptId: `browser-${Date.now()}`,
				});

				if (!result.success) {
					throw new Error(result.error.message);
				}

				logger.success("✅ Prompt submitted and accepted!");
				logger.info(
					// biome-ignore lint/suspicious/noExplicitAny: Unknown API response structure
					`📋 Prompt ID: ${(result.data as any)?.prompt_id || "unknown"}`,
				);
			} catch (error) {
				logger.error(
					`❌ Prompt submission failed: ${error instanceof Error ? error.message : "Unknown error"}`,
				);
			}
		},
		[isConnected, logger],
	);

	const ping = useCallback(async () => {
		if (!clientRef.current || !isConnected) {
			logger.error("❌ Not connected");
			return;
		}

		try {
			logger.info("🏓 Sending ping...");
			const result = await clientRef.current.ping();

			if (!result.success) {
				throw new Error(result.error.message);
			}

			logger.success("✅ Ping sent successfully!");
		} catch (error) {
			logger.error(
				`❌ Ping failed: ${error instanceof Error ? error.message : "Unknown error"}`,
			);
		}
	}, [isConnected, logger]);

	// Auto-connect logic with proper retry handling
	useEffect(() => {
		if (
			autoConnect &&
			config &&
			config.url &&
			!client &&
			!retryState.isRetrying
		) {
			logger.info("🔄 AutoConnect enabled - starting initial connection");
			setRetryState((prev) => ({ ...prev, attempts: 0, isRetrying: true }));
			createAndConnectClient(config, false);
		} else if (autoConnect && !config) {
			logger.info("🔄 AutoConnect enabled but no config provided");
		}
	}, [
		autoConnect,
		config,
		client,
		retryState.isRetrying,
		createAndConnectClient,
		logger,
	]);

	// Handle auto-retry on connection failure
	useEffect(() => {
		if (
			autoConnect &&
			config &&
			config.url &&
			status === "disconnected" &&
			!isConnected &&
			retryState.isRetrying &&
			retryState.attempts < retryState.maxAttempts
		) {
			const delay = Math.min(1000 * 2 ** retryState.attempts, 10000); // Exponential backoff, max 10s
			logger.info(
				`🔄 Scheduling retry ${retryState.attempts + 1}/${retryState.maxAttempts} in ${delay}ms`,
			);

			retryTimeoutRef.current = setTimeout(async () => {
				setRetryState((prev) => ({ ...prev, attempts: prev.attempts + 1 }));
				const success = await createAndConnectClient(config, true);

				if (!success && retryState.attempts + 1 >= retryState.maxAttempts) {
					logger.error(
						"🔄 AutoConnect exhausted all retry attempts - stopped trying",
					);
					logger.info("💡 Use the manual retry button to try again");
					setRetryState((prev) => ({ ...prev, isRetrying: false }));
				}
			}, delay);
		}

		return () => {
			if (retryTimeoutRef.current) {
				clearTimeout(retryTimeoutRef.current);
				retryTimeoutRef.current = null;
			}
		};
	}, [
		autoConnect,
		config,
		status,
		isConnected,
		retryState,
		createAndConnectClient,
		logger,
	]);

	// Tab focus handler removed - let ComfyClient handle its own reconnection logic
	// The infinite loop was caused by creating new ComfyClient instances on every tab focus

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			if (retryTimeoutRef.current) {
				clearTimeout(retryTimeoutRef.current);
			}
			if (clientRef.current) {
				clientRef.current.disconnect();
			}
		};
	}, []);

	return {
		client,
		connect,
		disconnect,
		isConnected,
		ping,
		retryState,
		status,
		submitPrompt,
	};
};
