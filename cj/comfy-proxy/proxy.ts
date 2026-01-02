import { verifyAuthHeader } from "@cj/comfy-auth";
import type { WebSocket, WebSocketHandler } from "bun";
import { serve } from "bun";
import {
	type CircuitBreakerConfig,
	createPromptQueue,
	type PromptQueue,
} from "./errorHandling";
import { createErrorJSON } from "./errorHelpers";
import { Logger } from "./logger";
import {
	clearSession,
	evictLeastActiveIfNeeded,
	getAllSessions,
	getSession,
	getSessionCount,
	getSessionsForUser,
	initializeSessionManager,
	type SessionManagerConfig,
	updateLastActive,
	upsertSession,
} from "./sessionManager";
import type {
	ComfyNode,
	ComfyWsMessage,
	ExtendedServerWebSocket,
	MetricsResponse,
	ProxyWsInbound,
	ProxyWsOutbound,
	SubmitPromptBody,
} from "./types";
import { ErrorCode } from "./types";

/**
 * Configuration for the ComfyUI Proxy Server
 *
 * All configuration values can be set via environment variables or CLI arguments.
 * See README.md for detailed documentation of each option.
 */
export interface ProxyConfig {
	CLEANUP_INTERVAL_MS: number;
	CIRCUIT_BREAKER_THRESHOLD: number;
	CIRCUIT_BREAKER_TIMEOUT_MS: number;
	CONNECTION_TIMEOUT_MS: number;
	HEALTH_CHECK_INTERVAL_MS: number;
	HEALTH_CHECK_TIMEOUT_MS: number;
	HTTP_REQUEST_TIMEOUT_MS: number;
	INITIAL_RETRY_DELAY_MS: number;
	PROXY_COMFY_JWT_SECRET: string;
	LOG_LEVEL: "debug" | "info" | "warn" | "error" | "silent";
	MAX_CONNECTIONS_PER_USER: number;
	MAX_PROMPT_RETRIES: number;
	MAX_QUEUED_PROMPTS_PER_USER: number;
	MAX_RETRY_DELAY_MS: number;
	METRICS_SECRET: string;
	PROXY_COMFY_URL: string;
	PROXY_COMFY_RECONNECT_ENABLED: boolean;
	PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS: number;
	PROXY_COMFY_RECONNECT_MAX_DELAY_MS: number;
	PROXY_COMFY_RECONNECT_MAX_RETRIES: number;
	PROXY_PORT: number;
	SESSION_IDLE_EVICTION_MS: number;
	SESSION_READY_TIMEOUT_MS: number;
	SESSION_TIMEOUT_MS: number;
}

// Type guard for WebSocket readyState
function isWebSocketOpen(ws: WebSocket | undefined): boolean {
	if (!ws) return false;
	const readyState = (ws as WebSocket & { readyState?: number }).readyState;
	return readyState === 1; // OPEN
}

// Health check function for ComfyUI server
export async function checkComfyHealth(
	comfyUrl: string,
	timeoutMs: number,
): Promise<boolean> {
	try {
		const healthUrl = `${comfyUrl}/system_stats`;

		// Use Promise.race to implement timeout
		const timeoutPromise = new Promise<never>((_, reject) => {
			setTimeout(() => reject(new Error("Health check timeout")), timeoutMs);
		});

		const fetchPromise = fetch(healthUrl, {
			method: "GET",
		});

		const response = await Promise.race([fetchPromise, timeoutPromise]);

		const isHealthy = response.ok;
		if (!isHealthy) {
			// Logging will be done by caller
			return false;
		}

		// Validate response body is valid JSON
		try {
			const contentType = response.headers.get("content-type");
			if (contentType && !contentType.includes("application/json")) {
				// Logging will be done by caller
			}
			const body = await response.json();
			// Basic validation: ensure response is an object (system_stats returns an object)
			// Reject arrays, null, and non-objects
			if (typeof body !== "object" || body === null || Array.isArray(body)) {
				return false;
			}
		} catch (_error) {
			return false;
		}

		return true;
	} catch (_error) {
		return false;
	}
}

// Periodic health check function
let healthCheckInterval: Timer | null = null;

export function startPeriodicHealthCheck(
	comfyUrl: string,
	intervalMs: number,
	timeoutMs: number,
	logger: Logger,
): void {
	// Clear any existing interval
	if (healthCheckInterval) {
		clearInterval(healthCheckInterval);
	}

	// Start periodic health checks
	healthCheckInterval = setInterval(async () => {
		const isHealthy = await checkComfyHealth(comfyUrl, timeoutMs);
		if (!isHealthy) {
			logger.warn("ComfyUI server appears to be unhealthy");
		}
	}, intervalMs);

	logger.info(`Started periodic ComfyUI health checks (every ${intervalMs}ms)`);
}

export function stopPeriodicHealthCheck(logger?: Logger): void {
	if (healthCheckInterval) {
		clearInterval(healthCheckInterval);
		healthCheckInterval = null;
		if (logger) {
			logger.info("Stopped periodic ComfyUI health checks");
		}
	}
}

function corsHeaders(): Record<string, string> {
	return {
		"access-control-allow-headers": "*",
		"access-control-allow-methods": "GET,POST,OPTIONS",
		"access-control-allow-origin": "*",
	};
}

function withCorsHeaders(
	base: Record<string, string> = {},
): Record<string, string> {
	return { ...corsHeaders(), ...base };
}

function badRequest(message: string, code?: number): Response {
	return new Response(JSON.stringify({ error: message }), {
		headers: withCorsHeaders({ "content-type": "application/json" }),
		status: code ?? 400,
	});
}

async function waitForComfySessionReady(
	sessionId: string,
	timeoutMs: number,
): Promise<void> {
	const session = getSession(sessionId);

	// If already ready, return immediately
	const comfyOpen = isWebSocketOpen(session?.comfyWs);
	if (session?.sid && comfyOpen) {
		return;
	}

	// Wait for readyPromise or timeout
	if (!session?.readyPromise) {
		// Fallback to polling if promise not initialized
		const start = Date.now();
		while (Date.now() - start < timeoutMs) {
			const s = getSession(sessionId);
			// Check if session still exists and is ready
			if (!s) {
				throw new Error(
					"Session was deleted while waiting for ComfyUI session",
				);
			}
			if (s.sid && isWebSocketOpen(s.comfyWs)) {
				return;
			}
			await new Promise((r) => setTimeout(r, 10));
		}
		throw new Error("Timeout establishing ComfyUI session");
	}

	await Promise.race([
		session.readyPromise,
		new Promise<never>((_, reject) =>
			setTimeout(
				() => reject(new Error("Timeout establishing ComfyUI session")),
				timeoutMs,
			),
		),
	]);
}

async function handleSubmitForUser(
	sessionId: string,
	body: SubmitPromptBody,
	config: ProxyConfig,
	promptQueue: PromptQueue,
	logger: Logger,
) {
	const session = getSession(sessionId);
	if (!session?.comfyWs) throw new Error("WS session not ready");
	// Ensure ComfyUI session is fully ready (sid available)
	if (!session.sid) {
		await waitForComfySessionReady(sessionId, config.SESSION_READY_TIMEOUT_MS);
	}

	const userId = session.userId;

	// Check circuit breaker
	if (!promptQueue.canProcess()) {
		// Try to queue the prompt instead
		const queuedPrompt = {
			extra_data: body.extra_data,
			partial_execution_targets: body.partial_execution_targets,
			prompt: body.prompt,
			prompt_id: body.prompt_id,
			retryCount: 0,
			timestamp: Date.now(),
			userId,
		};

		if (promptQueue.addPrompt(userId, queuedPrompt)) {
			throw new Error(
				"ComfyUI is temporarily unavailable, prompt queued for later processing",
			);
		}
		throw new Error(
			"ComfyUI is unavailable and queue is full, please try again later",
		);
	}

	// POST to ComfyUI HTTP endpoint (not WebSocket)
	// Sanitize prompt payload (e.g., ensure base64 fields are valid)
	const sanitizeBase64 = (value: unknown): string => {
		if (typeof value !== "string") return "";
		let v = value.trim();
		const m = v.match(/^data:image\/\w+;base64,(.+)$/);
		if (m?.[1]) v = m[1]!;
		// remove whitespace/newlines
		v = v.replace(/\s+/g, "");
		// fix padding to multiple of 4
		const pad = v.length % 4;
		if (pad === 2) v += "==";
		else if (pad === 3) v += "=";
		else if (pad === 1) {
			logger.warn(
				"⚠️ [SANITIZE] Invalid base64 padding (pad=1), rejecting corrupted data",
			);
			return "";
		}
		return v;
	};

	const sanitizePrompt = (prompt: unknown): unknown => {
		try {
			if (prompt && typeof prompt === "object") {
				const clone: Record<string, unknown> = JSON.parse(
					JSON.stringify(prompt),
				);
				for (const [key, node] of Object.entries(clone)) {
					if (
						node &&
						typeof node === "object" &&
						(node as ComfyNode).class_type === "LoadImageBase64"
					) {
						const inputs = (node as ComfyNode).inputs || {};
						if (typeof inputs.image_base64 === "string") {
							const beforeLen = inputs.image_base64.length;
							inputs.image_base64 = sanitizeBase64(inputs.image_base64);
							const afterLen = inputs.image_base64.length;
							logger.debug(
								`🧼 [SANITIZE] Node ${key} image_base64 length ${beforeLen} → ${afterLen}`,
							);
						}
					}
				}
				return clone;
			}
		} catch (e) {
			logger.warn("⚠️ [SANITIZE] Failed to sanitize prompt:", e);
		}
		return prompt;
	};

	const requestBody = {
		client_id: session.clientId,
		extra_data: body.extra_data,
		partial_execution_targets: body.partial_execution_targets,
		prompt: sanitizePrompt(body.prompt),
		prompt_id: body.prompt_id,
	};
	const promptUrl = `${config.PROXY_COMFY_URL}/prompt`;
	logger.debug(
		`POST ${promptUrl} (user=${userId}, client_id=${session.clientId}, prompt_id=${body.prompt_id ?? "(auto)"})`,
	);
	logger.debug(
		`Request body size: ${JSON.stringify(requestBody).length} chars`,
	);
	const startedAt = Date.now();

	let response: Response;
	try {
		logger.debug("Initiating fetch request...");
		response = await fetch(promptUrl, {
			body: JSON.stringify(requestBody),
			headers: { "Content-Type": "application/json" },
			method: "POST",
			signal: AbortSignal.timeout(config.HTTP_REQUEST_TIMEOUT_MS),
		});
		logger.debug("Fetch completed, got response object");
	} catch (error) {
		promptQueue.recordFailure();
		const errorMsg = error instanceof Error ? error.message : "Unknown error";
		logger.error(`Failed to connect to ComfyUI: ${errorMsg}`);
		logger.error(`ComfyUI URL: ${promptUrl}`);
		logger.error(
			`ComfyUI server may be unreachable at ${config.PROXY_COMFY_URL}`,
		);
		throw new Error(
			`Failed to connect to ComfyUI: ${errorMsg}. Please check that ComfyUI is running on ${config.PROXY_COMFY_URL}`,
		);
	}

	const elapsedMs = Date.now() - startedAt;
	logger.info(
		`Response ${response.status} ${response.statusText} in ${elapsedMs}ms (user=${userId})`,
		{
			durationMs: elapsedMs,
			promptId: body.prompt_id,
			requestSize: JSON.stringify(requestBody).length,
			statusCode: response.status,
			userId,
		},
	);
	logger.debug(
		"Response headers:",
		Object.fromEntries(response.headers.entries()),
	);

	if (!response.ok) {
		let errorData: unknown;
		try {
			const responseText = await response.text();
			logger.debug(`Response body text (${responseText.length} chars)`);

			// Try to parse as JSON
			try {
				errorData = JSON.parse(responseText);
				logger.debug("Parsed error data:", errorData);
			} catch (parseError) {
				logger.debug("Could not parse response as JSON:", parseError);
				errorData = { error: `Raw response: ${responseText}` };
			}
		} catch (textError) {
			logger.error("Could not read response text:", textError);
			errorData = { error: "Could not read error response" };
		}

		promptQueue.recordFailure();
		logger.error(
			`ComfyUI error response: ${response.status} ${response.statusText}`,
		);
		logger.error(`ComfyUI URL: ${promptUrl}`);
		throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
	}

	// Success - record it and process any queued prompts
	promptQueue.recordSuccess();
	let json: unknown;
	try {
		json = await response.json();
		logger.debug(`/prompt OK (user=${userId})`);
	} catch (e) {
		logger.warn("/prompt OK but JSON parse failed:", e);
		json = {};
	}

	// Process queued prompts for this user
	processQueuedPrompts(userId, config, promptQueue, logger);

	return json;
}

async function processQueuedPrompts(
	userId: string,
	config: ProxyConfig,
	promptQueue: PromptQueue,
	logger: Logger,
) {
	while (promptQueue.getQueueSize(userId) > 0 && promptQueue.canProcess()) {
		const queuedPrompt = promptQueue.getNextPrompt(userId);
		if (!queuedPrompt) break;

		// Check retry count and apply exponential backoff
		const retryCount = queuedPrompt.retryCount || 0;
		if (retryCount > 0) {
			// Calculate exponential backoff delay: min(INITIAL * 2^(retryCount-1), MAX)
			const delayMs = Math.min(
				config.INITIAL_RETRY_DELAY_MS * 2 ** (retryCount - 1),
				config.MAX_RETRY_DELAY_MS,
			);
			// Only wait if we haven't already waited long enough
			const timeSinceQueued = Date.now() - queuedPrompt.timestamp;
			if (timeSinceQueued < delayMs) {
				// Re-queue and wait
				promptQueue.addPrompt(userId, queuedPrompt);
				break;
			}
		}

		// Check if we've exceeded max retries
		if (retryCount >= config.MAX_PROMPT_RETRIES) {
			logger.error(
				`Dropping queued prompt for user ${userId} after ${retryCount} retries (prompt_id: ${queuedPrompt.prompt_id || "unknown"})`,
			);
			// Don't re-queue, just drop it
			continue;
		}

		// Find an active session for this user
		const userSessions = getSessionsForUser(userId);
		if (userSessions.length === 0) {
			logger.warn(
				`No active session for user ${userId}, cannot process queued prompt`,
			);
			// Re-queue with incremented retry count
			promptQueue.addPrompt(userId, {
				...queuedPrompt,
				retryCount: retryCount + 1,
			});
			break;
		}

		// Use the most recent session
		const sessionId = userSessions[0]?.sessionId;
		if (!sessionId) {
			logger.warn(
				`No valid session ID for user ${userId}, cannot process queued prompt`,
			);
			// Re-queue with incremented retry count
			promptQueue.addPrompt(userId, {
				...queuedPrompt,
				retryCount: retryCount + 1,
			});
			break;
		}

		try {
			await handleSubmitForUser(
				sessionId,
				queuedPrompt,
				config,
				promptQueue,
				logger,
			);
			logger.debug(
				`Processed queued prompt for user ${userId} (retry ${retryCount})`,
			);
		} catch (error) {
			logger.error(
				`Failed to process queued prompt for user ${userId} (retry ${retryCount}):`,
				error,
			);
			// Re-queue with incremented retry count
			promptQueue.addPrompt(userId, {
				...queuedPrompt,
				retryCount: retryCount + 1,
			});
			break;
		}
	}
}

type WsData = { userId: string; sessionId?: string };
// Queue to store messages that arrive before sessionId is set
const messageQueue = new Map<string, string[]>();

/**
 * Sets up ComfyUI WebSocket connection handlers (message, error, close)
 * This is extracted to be reusable for both initial connection and reconnection
 */
function setupComfyConnectionHandlers(
	comfyConnection: WebSocket,
	sessionId: string,
	userId: string,
	clientWs: ExtendedServerWebSocket<WsData>,
	config: ProxyConfig,
	logger: Logger,
): void {
	// Set up message forwarding from ComfyUI to client
	comfyConnection.onmessage = (evt: MessageEvent) => {
		logger.debug(
			`Message from ComfyUI for user ${userId}:`,
			typeof evt.data === "string"
				? (evt.data as string).substring(0, 200)
				: "binary data",
		);
		if (typeof evt.data === "string") {
			try {
				const msg = JSON.parse(evt.data as string) as ComfyWsMessage;
				if (msg?.type) {
					logger.debug(`Parsed type=${msg.type} for session ${sessionId}`);
					// Log full message for critical types
					if (
						["executed", "execution_success", "execution_error"].includes(
							msg.type,
						)
					) {
						logger.debug(`Full ${msg.type} message:`, JSON.stringify(msg));
					}
				}
				if (
					msg.type === "status" &&
					(msg.data as { status?: unknown; sid?: string })?.sid
				) {
					const sid = (msg.data as { sid?: string }).sid;
					if (sid) {
						logger.debug(`Updating session ${sessionId} with sid: ${sid}`);
						const session = getSession(sessionId);
						if (session?.markReady) {
							session.markReady();
						}
						upsertSession(userId, { sessionId, sid });
					}
				}

				// Additionally, normalize errors for clients by emitting a simple error envelope
				if (msg.type === "execution_error") {
					try {
						const data = (msg.data || {}) as Record<string, unknown>;
						const norm = {
							exception_message: (data as { exception_message?: string })
								.exception_message,
							exception_type: (data as { exception_type?: string })
								.exception_type,
							node_id: (data as { node_id?: string }).node_id,
							prompt_id: (data as { prompt_id?: string }).prompt_id,
							traceback: (data as { traceback?: string[] }).traceback,
						};
						// Log full execution error details server-side
						logger.error("EXECUTION ERROR", {
							exceptionMessage: norm.exception_message,
							exceptionType: norm.exception_type,
							nodeId: norm.node_id,
							promptId: norm.prompt_id,
							traceback: norm.traceback,
							userId,
						});
						if (clientWs.readyState === 1) {
							clientWs.send(
								createErrorJSON(
									norm.exception_message || "Execution error",
									ErrorCode.UNKNOWN_ERROR,
									{
										promptId: norm.prompt_id,
										userId,
									},
								),
							);
						}
					} catch {}
				}
				// Relay to client
				logger.debug(
					`Relaying message type=${msg.type} to client for session ${sessionId}`,
				);
				if (clientWs.readyState === 1) {
					clientWs.send(evt.data as string);
				}
			} catch (error) {
				logger.error("Failed to parse message, sending raw data:", error);
				if (clientWs.readyState === 1) {
					clientWs.send(evt.data as string);
				}
			}
		} else if (evt.data instanceof ArrayBuffer) {
			logger.debug(
				`Binary data (${evt.data.byteLength} bytes) for user ${userId}`,
			);
			if (clientWs.readyState === 1) {
				clientWs.send(evt.data);
			}
		}
	};

	comfyConnection.onerror = (error) => {
		const errorMsg = `ComfyUI connection failed. Please check that ComfyUI is running on ${config.PROXY_COMFY_URL}`;
		logger.error(`ComfyUI connection error for user ${userId}:`, error);

		// Only send error if client is still connected
		if (clientWs.readyState === 1) {
			clientWs.send(
				createErrorJSON(errorMsg, ErrorCode.UNKNOWN_ERROR, {
					retryable: true,
					userId,
				}),
			);
		}
	};

	comfyConnection.onclose = () => {
		logger.warn(
			`ComfyUI connection closed for user: ${userId}, session: ${sessionId}`,
		);

		// Check if client connection is still open
		if (clientWs.readyState !== 1) {
			logger.debug(
				`Client connection already closed for session ${sessionId}, clearing session`,
			);
			clearSession(sessionId);
			return;
		}

		// Check if reconnection is enabled
		if (!config.PROXY_COMFY_RECONNECT_ENABLED) {
			logger.debug(
				`Reconnection disabled, closing client connection for session ${sessionId}`,
			);
			clientWs.send(
				createErrorJSON(
					"ComfyUI connection closed unexpectedly",
					ErrorCode.UNKNOWN_ERROR,
					{
						retryable: true,
						userId,
					},
				),
			);
			clientWs.close();
			clearSession(sessionId);
			return;
		}

		// Attempt reconnection
		const session = getSession(sessionId);
		if (!session) {
			logger.debug(
				`Session ${sessionId} already cleared, not attempting reconnection`,
			);
			return;
		}

		// Check if already reconnecting
		if (session.isReconnecting) {
			logger.debug(
				`Already reconnecting for session ${sessionId}, ignoring close event`,
			);
			return;
		}

		// Check if max retries exceeded
		const reconnectAttempts = session.reconnectAttempts || 0;
		if (reconnectAttempts >= config.PROXY_COMFY_RECONNECT_MAX_RETRIES) {
			logger.error(
				`Max reconnection attempts (${config.PROXY_COMFY_RECONNECT_MAX_RETRIES}) exceeded for session ${sessionId}`,
			);
			clientWs.send(
				createErrorJSON(
					"ComfyUI connection closed and reconnection failed after maximum retries",
					ErrorCode.UNKNOWN_ERROR,
					{
						retryable: false,
						userId,
					},
				),
			);
			clientWs.close();
			clearSession(sessionId);
			return;
		}

		// Schedule reconnection with exponential backoff
		reconnectComfyConnection(sessionId, config, logger);
	};
}

/**
 * Attempts to reconnect the ComfyUI WebSocket connection for a session
 */
function reconnectComfyConnection(
	sessionId: string,
	config: ProxyConfig,
	logger: Logger,
): void {
	const session = getSession(sessionId);
	if (!session) {
		logger.debug(`Session ${sessionId} not found, cannot reconnect`);
		return;
	}

	const userId = session.userId;
	const clientWs = session.clientWs;
	if (!clientWs) {
		logger.debug(
			`No client WebSocket for session ${sessionId}, cannot reconnect`,
		);
		clearSession(sessionId);
		return;
	}

	// Check if client connection is still open
	if (clientWs.readyState !== 1) {
		logger.debug(
			`Client connection closed for session ${sessionId}, clearing session`,
		);
		clearSession(sessionId);
		return;
	}

	// Check if already reconnecting
	if (session.isReconnecting) {
		logger.debug(`Already reconnecting for session ${sessionId}`);
		return;
	}

	const reconnectAttempts = (session.reconnectAttempts || 0) + 1;

	// Calculate exponential backoff delay
	const delayMs = Math.min(
		config.PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS *
			2 ** (reconnectAttempts - 1),
		config.PROXY_COMFY_RECONNECT_MAX_DELAY_MS,
	);

	logger.info(
		`Scheduling reconnection attempt ${reconnectAttempts}/${config.PROXY_COMFY_RECONNECT_MAX_RETRIES} for session ${sessionId} in ${delayMs}ms`,
	);

	// Mark as reconnecting
	upsertSession(userId, {
		isReconnecting: true,
		reconnectAttempts,
		sessionId,
	});

	// Schedule reconnection
	const timeoutId = setTimeout(async () => {
		const currentSession = getSession(sessionId);
		if (!currentSession) {
			logger.debug(`Session ${sessionId} cleared during reconnection delay`);
			return;
		}

		// Check if client is still connected
		if (!currentSession.clientWs || currentSession.clientWs.readyState !== 1) {
			logger.debug(
				`Client disconnected during reconnection delay for session ${sessionId}`,
			);
			clearSession(sessionId);
			return;
		}

		logger.info(
			`Attempting reconnection ${reconnectAttempts}/${config.PROXY_COMFY_RECONNECT_MAX_RETRIES} for session ${sessionId}`,
		);

		const protocol = config.PROXY_COMFY_URL.startsWith("https://")
			? "wss://"
			: "ws://";
		const comfyWsUrl = `${config.PROXY_COMFY_URL.replace(/^https?:\/\//, protocol)}/ws?clientId=${currentSession.clientId}`;

		try {
			const comfyConnection = new WebSocket(comfyWsUrl);

			// Create new ready promise
			let markReady: (() => void) | undefined;
			const readyPromise = new Promise<void>((resolve) => {
				markReady = resolve;
			});

			// Set up handlers
			setupComfyConnectionHandlers(
				comfyConnection,
				sessionId,
				userId,
				currentSession.clientWs as ExtendedServerWebSocket<WsData>,
				config,
				logger,
			);

			// Wait for connection to open
			await new Promise<void>((resolve, reject) => {
				const timeout = setTimeout(() => {
					reject(new Error("Reconnection timeout"));
				}, config.CONNECTION_TIMEOUT_MS);

				const onOpenHandler = () => {
					clearTimeout(timeout);
					comfyConnection.onerror = null;
					resolve();
				};

				const onErrorHandler = () => {
					clearTimeout(timeout);
					comfyConnection.onopen = null;
					reject(new Error("Reconnection failed"));
				};

				comfyConnection.onopen = onOpenHandler;
				comfyConnection.onerror = onErrorHandler;
			});

			logger.info(`Reconnection successful for session ${sessionId}`);

			// Update session with new connection
			upsertSession(userId, {
				comfyWs: comfyConnection,
				connectionState: "connected",
				isReconnecting: false,
				markReady,
				readyPromise,
				reconnectAttempts: 0, // Reset reconnect attempts after successful reconnection
				reconnectTimeoutId: undefined,
				sessionId,
			});

			// Wait for sid to be received (or timeout)
			try {
				await Promise.race([
					readyPromise,
					new Promise<never>((_, reject) =>
						setTimeout(
							() =>
								reject(new Error("Timeout waiting for sid after reconnection")),
							config.SESSION_READY_TIMEOUT_MS,
						),
					),
				]);
				logger.info(`Session ${sessionId} fully restored after reconnection`);
			} catch (error) {
				logger.warn(
					`Session ${sessionId} reconnected but sid not received:`,
					error,
				);
			}

			// Notify client of successful reconnection
			if (currentSession.clientWs.readyState === 1) {
				currentSession.clientWs.send(
					JSON.stringify({
						data: { message: "ComfyUI connection restored" },
						type: "reconnected",
					} satisfies ProxyWsOutbound),
				);
			}
		} catch (error) {
			logger.error(
				`Reconnection attempt ${reconnectAttempts} failed for session ${sessionId}:`,
				error,
			);

			// Clear reconnecting flag and schedule next attempt or give up
			const updatedSession = getSession(sessionId);
			if (!updatedSession) {
				return; // Session was cleared
			}

			if (reconnectAttempts >= config.PROXY_COMFY_RECONNECT_MAX_RETRIES) {
				logger.error(
					`Max reconnection attempts exceeded for session ${sessionId}, closing client connection`,
				);
				if (
					updatedSession.clientWs &&
					updatedSession.clientWs.readyState === 1
				) {
					updatedSession.clientWs.send(
						createErrorJSON(
							"ComfyUI reconnection failed after maximum retries",
							ErrorCode.UNKNOWN_ERROR,
							{
								retryable: false,
								userId,
							},
						),
					);
					updatedSession.clientWs.close();
				}
				clearSession(sessionId);
			} else {
				// Schedule next reconnection attempt
				upsertSession(userId, {
					isReconnecting: false,
					reconnectTimeoutId: undefined,
					sessionId,
				});
				reconnectComfyConnection(sessionId, config, logger);
			}
		}
	}, delayMs);

	// Store timeout ID for cancellation
	upsertSession(userId, {
		reconnectTimeoutId: timeoutId,
		sessionId,
	});
}

function createWsHandler(
	config: ProxyConfig,
	promptQueue: PromptQueue,
	logger: Logger,
): WebSocketHandler<WsData> {
	const sessionManagerConfig: SessionManagerConfig = {
		CLEANUP_INTERVAL_MS: config.CLEANUP_INTERVAL_MS,
		MAX_CONNECTIONS_PER_USER: config.MAX_CONNECTIONS_PER_USER,
		SESSION_IDLE_EVICTION_MS: config.SESSION_IDLE_EVICTION_MS,
		SESSION_TIMEOUT_MS: config.SESSION_TIMEOUT_MS,
	};

	const handler: WebSocketHandler<WsData> = {
		close(ws) {
			const extendedWs = ws as ExtendedServerWebSocket<WsData>;
			const sessionId = extendedWs.sessionId;
			const userId = ws.data?.userId || extendedWs.userId;

			// Clear message queue for this user to prevent memory leaks
			if (userId) {
				messageQueue.delete(userId);
			}

			if (!sessionId) return;

			logger.debug(`Closing session: ${sessionId}`);

			// Cancel any pending reconnection attempts
			const session = getSession(sessionId);
			if (session?.reconnectTimeoutId) {
				logger.debug(
					`Cancelling reconnection attempt for session ${sessionId}`,
				);
				clearTimeout(session.reconnectTimeoutId);
			}

			// Close the ComfyUI connection for this specific session
			if (session?.comfyWs) {
				try {
					session.comfyWs.close();
				} catch (error) {
					logger.error(
						`Error closing ComfyUI connection for session ${sessionId}:`,
						error,
					);
				}
			}

			clearSession(sessionId);
		},
		async message(ws, message) {
			const extendedWs = ws as ExtendedServerWebSocket<WsData>;
			// Try to get userId from ws.data (set during upgrade)
			const userId = ws.data?.userId || extendedWs.userId;
			const sessionId = extendedWs.sessionId;

			// Get correlation ID from session if available
			const session = sessionId ? getSession(sessionId) : undefined;
			const correlationId = session?.correlationId || crypto.randomUUID();
			const messageLogger = logger.withCorrelationId(correlationId);

			messageLogger.debug(
				`Message handler - sessionId=${sessionId} userId=${userId}`,
			);
			if (!userId) {
				messageLogger.warn(
					"No userId found in WebSocket, cannot process message",
				);
				return;
			}

			// If we don't have a sessionId, we're still in the connecting phase
			// Queue the message to be processed once sessionId is set
			if (!sessionId) {
				messageLogger.debug(
					`No sessionId yet for user ${userId}, connection still being established. Queueing message...`,
				);

				// Queue the message for this userId
				if (!messageQueue.has(userId)) {
					messageQueue.set(userId, []);
				}
				const queue = messageQueue.get(userId)!;

				// Convert message to string if it's not already
				const messageStr =
					typeof message === "string" ? message : JSON.stringify(message);
				queue.push(messageStr);
				messageLogger.debug(
					`Queued message for user ${userId} (${queue.length} messages in queue). Will process once session is ready.`,
				);
				return;
			}

			// Update last active timestamp
			updateLastActive(sessionId);

			try {
				if (typeof message === "string") {
					messageLogger.debug(
						`Received message from user ${userId}: ${message.substring(0, 200)}`,
					);
					let parsed: ProxyWsInbound;
					try {
						parsed = JSON.parse(message);
					} catch {
						messageLogger.warn(`Failed to parse message from user ${userId}`);
						return;
					}
					messageLogger.debug(
						`Parsed message type=${parsed.type} from user ${userId}`,
					);
					if (parsed.type === "submit_prompt") {
						messageLogger.debug(`Processing submit_prompt from user ${userId}`);

						// Wait for session to be ready
						const session = getSession(sessionId);
						if (!session?.comfyWs) {
							messageLogger.debug(
								`Waiting for ComfyUI session to be ready for session ${sessionId}`,
							);
							try {
								await waitForComfySessionReady(
									sessionId,
									config.SESSION_READY_TIMEOUT_MS,
								);
							} catch (error) {
								messageLogger.error(
									`Timeout waiting for ComfyUI session for session ${sessionId}:`,
									error,
								);
								ws.send(
									createErrorJSON(
										"ComfyUI session not ready",
										ErrorCode.SESSION_NOT_READY,
										{ retryable: true, userId },
									),
								);
								return;
							}
						}

						// Ensure prompt_id is set
						const incoming = parsed.data as SubmitPromptBody;
						const finalPromptId =
							incoming.prompt_id ||
							`proxy-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
						incoming.prompt_id = finalPromptId;

						try {
							messageLogger.debug(
								`Calling handleSubmitForUser for session ${sessionId} (prompt_id=${finalPromptId})`,
							);

							const res = await handleSubmitForUser(
								sessionId,
								incoming,
								config,
								promptQueue,
								messageLogger,
							);

							messageLogger.debug(
								`handleSubmitForUser SUCCESS for session ${sessionId}, prompt_id=${finalPromptId}`,
							);

							const responseMessage = {
								data: res,
								type: "prompt_accepted",
							} satisfies ProxyWsOutbound;

							// Check WebSocket state before sending
							if (ws.readyState !== 1) {
								messageLogger.error(
									"WebSocket not in OPEN state! Cannot send prompt_accepted",
								);
								return;
							}

							ws.send(JSON.stringify(responseMessage));
							messageLogger.debug(
								`Sent prompt_accepted to user ${userId} for prompt_id=${finalPromptId}`,
							);
						} catch (error) {
							messageLogger.error(
								`FAILURE in submit_prompt flow for session ${sessionId}, user ${userId}, prompt_id=${finalPromptId}:`,
								error,
							);
							const errorMsg = (error as Error).message;
							let errorCode: ErrorCode = ErrorCode.UNKNOWN_ERROR;
							let retryable = false;

							if (
								errorMsg.includes("unavailable") ||
								errorMsg.includes("timed out")
							) {
								errorCode = ErrorCode.UNKNOWN_ERROR;
								retryable = true;
							} else if (errorMsg.includes("queue")) {
								errorCode = ErrorCode.QUEUE_FULL;
								retryable = true;
							}

							ws.send(
								createErrorJSON(errorMsg, errorCode, {
									retryable,
									userId,
								}),
							);
						}
					}
				}
			} catch (e) {
				const errorMsg = (e as Error).message;
				let errorCode: ErrorCode = ErrorCode.UNKNOWN_ERROR;
				let retryable = false;

				if (errorMsg.includes("connections per user")) {
					errorCode = ErrorCode.MAX_CONNECTIONS_EXCEEDED;
				} else if (errorMsg.includes("connection")) {
					errorCode = ErrorCode.UNKNOWN_ERROR;
					retryable = true;
				}

				ws.send(createErrorJSON(errorMsg, errorCode, { retryable }));
			}
		},
		async open(ws) {
			try {
				// Generate correlation ID for this WebSocket connection
				const correlationId = crypto.randomUUID();
				const sessionLogger = logger.withCorrelationId(correlationId);

				sessionLogger.debug("WebSocket connection opened, verifying auth...");
				// Auth is already verified in the fetch handler before upgrade
				// Just use the userId from ws.data
				const userId = ws.data?.userId;
				if (!userId) {
					throw new Error("User ID not found in WebSocket data");
				}
				sessionLogger.debug(`Auth successful for user: ${userId}`, {
					correlationId,
				});

				// Check and enforce connection limit for this user
				const userSessions = getSessionsForUser(userId);
				sessionLogger.debug(
					`User ${userId} currently has ${userSessions.length} active session(s)`,
				);

				if (userSessions.length >= config.MAX_CONNECTIONS_PER_USER) {
					sessionLogger.debug(
						`User ${userId} at capacity (${userSessions.length}/${config.MAX_CONNECTIONS_PER_USER}), attempting eviction...`,
					);
					const evicted = evictLeastActiveIfNeeded(
						userId,
						sessionManagerConfig,
					);
					if (!evicted) {
						const errorMsg = `Maximum connections exceeded (${config.MAX_CONNECTIONS_PER_USER} per user). Please close existing connections first.`;
						sessionLogger.error(errorMsg);
						ws.send(
							createErrorJSON(errorMsg, ErrorCode.MAX_CONNECTIONS_EXCEEDED, {
								retryable: false,
								userId,
							}),
						);
						ws.close();
						return;
					}
					sessionLogger.debug(
						`Evicted least active session for user ${userId}`,
					);
				}

				// Generate session ID for this connection
				const clientId = crypto.randomUUID().replace(/-/g, "");
				const sessionId = `${userId}:${clientId}`;

				sessionLogger.debug(
					`Creating session ${sessionId} for user: ${userId}`,
				);
				const protocol = config.PROXY_COMFY_URL.startsWith("https://")
					? "wss://"
					: "ws://";
				const comfyWsUrl = `${config.PROXY_COMFY_URL.replace(/^https?:\/\//, protocol)}/ws?clientId=${clientId}`;
				sessionLogger.debug(`Connecting to ComfyUI: ${comfyWsUrl}`);

				let comfyConnection: WebSocket;

				comfyConnection = new WebSocket(comfyWsUrl);

				// Create promise that will resolve when sid is received
				let markReady: (() => void) | undefined;
				const readyPromise = new Promise<void>((resolve) => {
					markReady = resolve;
				});

				// Wait for the ComfyUI connection to be established
				await new Promise<void>((resolve, reject) => {
					const timeout = setTimeout(() => {
						const errorMsg = `Timeout establishing ComfyUI connection. Please ensure ComfyUI is running on ${config.PROXY_COMFY_URL}`;
						sessionLogger.error(
							`Timeout establishing ComfyUI connection for user ${userId}`,
						);
						sessionLogger.error(`ComfyUI URL: ${comfyWsUrl}`);

						// Send error to client before closing
						ws.send(
							createErrorJSON(errorMsg, ErrorCode.TIMEOUT, {
								retryable: true,
								userId,
							}),
						);

						reject(new Error(errorMsg));
					}, config.CONNECTION_TIMEOUT_MS);

					const onOpenHandler = () => {
						clearTimeout(timeout);
						sessionLogger.debug(
							`ComfyUI connection established for user: ${userId}`,
						);
						// Clear the onerror handler we set here to avoid conflicts
						comfyConnection.onerror = null;
						resolve();
					};

					const onErrorHandler = (error: unknown) => {
						clearTimeout(timeout);
						comfyConnection.onopen = null;
						reject(error);
					};

					comfyConnection.onopen = onOpenHandler;
					comfyConnection.onerror = onErrorHandler;
				});

				// Create session with the dedicated connection
				const extendedWs = ws as ExtendedServerWebSocket<WsData>;
				upsertSession(userId, {
					clientId,
					clientWs: extendedWs,
					comfyWs: comfyConnection,
					connectionState: "connected",
					correlationId, // Store correlation ID in session
					isReconnecting: false,
					markReady,
					readyPromise,
					reconnectAttempts: 0, // Reset reconnect attempts for new connection
					sessionId,
				});

				// Set up connection handlers after connection is established
				setupComfyConnectionHandlers(
					comfyConnection,
					sessionId,
					userId,
					extendedWs,
					config,
					sessionLogger,
				);
				extendedWs.userId = userId;
				extendedWs.sessionId = sessionId;
				sessionLogger.debug(
					`Session created: ${sessionId} for user: ${userId} with dedicated ComfyUI connection`,
				);

				// Process any queued messages for this userId
				const queuedMessages = messageQueue.get(userId);
				if (queuedMessages && queuedMessages.length > 0) {
					sessionLogger.debug(
						`Processing ${queuedMessages.length} queued message(s) for user ${userId}...`,
					);
					for (const queuedMessage of queuedMessages) {
						// Process the message as if it just arrived
						// This will trigger the normal message handler with sessionId now set
						const result = handler.message(ws, queuedMessage);
						if (result instanceof Promise) {
							result.catch((err: unknown) => {
								sessionLogger.error(
									`Error processing queued message for user ${userId}:`,
									err,
								);
							});
						}
					}
					// Clear the queue after processing
					messageQueue.delete(userId);
					sessionLogger.debug(`Processed and cleared queue for user ${userId}`);
				}
			} catch (e) {
				const correlationId = logger.getCorrelationId();
				const errorLogger = correlationId
					? logger
					: logger.withCorrelationId(crypto.randomUUID());
				errorLogger.error("Error in WebSocket open handler:", e);
				errorLogger.error(`ComfyUI URL: ${config.PROXY_COMFY_URL}`);

				const userId = ws.data?.userId;
				// Clear message queue on connection failure to prevent memory leaks
				if (userId) {
					messageQueue.delete(userId);
				}
				const baseErrorMsg = (e as Error).message;
				let errorMsg = baseErrorMsg;
				let errorCode: ErrorCode = ErrorCode.UNKNOWN_ERROR;

				// Enhance error messages for common cases
				if (
					baseErrorMsg.includes("connections") ||
					baseErrorMsg.includes("capacity")
				) {
					errorCode = ErrorCode.MAX_CONNECTIONS_EXCEEDED;
				} else if (
					baseErrorMsg.includes("Time") ||
					baseErrorMsg.includes("timeout")
				) {
					errorCode = ErrorCode.TIMEOUT;
					errorMsg = `Connection timeout: ${baseErrorMsg}. Please check that ComfyUI is running on ${config.PROXY_COMFY_URL}`;
				} else if (
					baseErrorMsg.includes("Failed to connect") ||
					baseErrorMsg.includes("ECONNREFUSED")
				) {
					errorMsg = `Cannot connect to ComfyUI: ${baseErrorMsg}. Please ensure ComfyUI is running on ${config.PROXY_COMFY_URL}`;
				}

				ws.send(
					createErrorJSON(errorMsg, errorCode, {
						retryable: true,
						userId,
					}),
				);
				ws.close();
			}
		},
	};

	return handler;
}

/**
 * Starts the ComfyUI proxy server
 *
 * Initializes session management, prompt queue, and WebSocket handlers.
 * The server listens on the configured port and proxies connections to ComfyUI.
 *
 * @param config - Proxy configuration (typically from env.ts)
 * @example
 * ```typescript
 * import { startProxy } from './proxy';
 * import { env } from './env';
 *
 * const config: ProxyConfig = {
 *   // ... config from env
 * };
 *
 * startProxy(config);
 * ```
 */
export function startProxy(config: ProxyConfig): void {
	// Create logger
	const logger = new Logger(config.LOG_LEVEL, "[Proxy]");

	// Initialize session manager with config
	const sessionManagerConfig: SessionManagerConfig = {
		CLEANUP_INTERVAL_MS: config.CLEANUP_INTERVAL_MS,
		MAX_CONNECTIONS_PER_USER: config.MAX_CONNECTIONS_PER_USER,
		SESSION_IDLE_EVICTION_MS: config.SESSION_IDLE_EVICTION_MS,
		SESSION_TIMEOUT_MS: config.SESSION_TIMEOUT_MS,
	};
	initializeSessionManager(sessionManagerConfig);

	// Create prompt queue with config
	const circuitBreakerConfig: CircuitBreakerConfig = {
		threshold: config.CIRCUIT_BREAKER_THRESHOLD,
		timeoutMs: config.CIRCUIT_BREAKER_TIMEOUT_MS,
	};
	const promptQueue = createPromptQueue(
		config.MAX_QUEUED_PROMPTS_PER_USER,
		circuitBreakerConfig,
		logger,
	);

	// Create WebSocket handler with config
	const wsHandler = createWsHandler(config, promptQueue, logger);

	serve({
		async fetch(req, serverInstance) {
			const url = new URL(req.url);

			// Handle CORS preflight
			if (req.method === "OPTIONS") {
				return new Response(null, {
					headers: withCorsHeaders(),
					status: 204,
				});
			}

			// Health check endpoints
			if (url.pathname === "/health" || url.pathname === "/") {
				return new Response(JSON.stringify({ ok: true }), {
					headers: withCorsHeaders({ "content-type": "application/json" }),
				});
			}

			// Live endpoint (process is alive)
			if (url.pathname === "/live") {
				return new Response(
					JSON.stringify({ status: "alive", uptime: process.uptime() }),
					{
						headers: withCorsHeaders({ "content-type": "application/json" }),
					},
				);
			}

			// Ready endpoint (ready to accept traffic)
			if (url.pathname === "/ready") {
				const activeSessions = getSessionCount();
				return new Response(
					JSON.stringify({
						active_connections: activeSessions,
						status: "ready",
					}),
					{
						headers: withCorsHeaders({ "content-type": "application/json" }),
						status: 200,
					},
				);
			}

			// Metrics endpoint with secret authentication
			if (url.pathname === "/metrics") {
				const auth = req.headers.get("authorization");
				const expectedSecret = config.METRICS_SECRET;

				if (auth !== `Bearer ${expectedSecret}`) {
					return badRequest("Unauthorized", 401);
				}

				const detailed = url.searchParams.get("detailed") === "true";
				const sessions = getAllSessions();

				const queueSizes = promptQueue.getAllQueueSizes();
				const metrics: MetricsResponse = {
					active_connections: getSessionCount(),
					active_sessions: getSessionCount(),
					circuit_breaker_state: promptQueue.getCircuitBreakerState(),
					memory_usage: process.memoryUsage(),
					queued_prompts: Object.fromEntries(queueSizes),
					uptime_seconds: process.uptime(),
				};

				if (detailed) {
					metrics.detailed_sessions = sessions.map((session) => ({
						clientId: session.clientId,
						connectionState: session.connectionState,
						lastActiveAt: session.lastActiveAt,
						userId: session.userId,
					}));
				}

				return new Response(JSON.stringify(metrics), {
					headers: withCorsHeaders({ "content-type": "application/json" }),
				});
			}

			// Handle WebSocket upgrade requests
			if (url.pathname === "/ws") {
				// Extract token from query parameter (WebSockets don't support headers)
				const tokenParam = url.searchParams.get("token");
				if (!tokenParam) {
					return badRequest("Missing token query parameter", 401);
				}
				const auth = `Bearer ${tokenParam}`;

				try {
					const { userId } = await verifyAuthHeader(
						auth,
						config.PROXY_COMFY_JWT_SECRET,
					);

					// Perform the upgrade manually - this is required
					const upgraded = serverInstance.upgrade(req, {
						data: { userId },
					});

					if (!upgraded) {
						return badRequest("Failed to upgrade WebSocket connection", 400);
					}

					// Return 101 Switching Protocols
					return new Response(null, { status: 101 });
				} catch (e) {
					return badRequest((e as Error).message, 401);
				}
			}

			return badRequest("Not found", 404);
		},
		port: config.PROXY_PORT,
		websocket: wsHandler,
	});

	logger.info(
		`Proxy listening on :${config.PROXY_PORT} (-> ${config.PROXY_COMFY_URL})`,
	);
	logger.info(
		"Ready to accept connections with dedicated ComfyUI connections per client",
	);
}
