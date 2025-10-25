import type { WebSocket, WebSocketHandler } from "bun";
import { serve } from "bun";
import { env } from "./env";
import { circuitBreaker, promptQueue } from "./errorHandling";
import { createErrorJSON } from "./errorHelpers";
import { verifyAuthHeader } from "./jwt";
import {
	acquireSessionCreationLock,
	canCreateSession,
	clearSession,
	getAllSessions,
	getSession,
	getSessionCount,
	releaseSessionCreationLock,
	updateLastActive,
	upsertSession,
} from "./sessionManager";
import type {
	ComfyWsMessage,
	ExtendedServerWebSocket,
	MetricsResponse,
	ProxyWsInbound,
	ProxyWsOutbound,
	SubmitPromptBody,
} from "./types";
import { ErrorCode } from "./types";
import { validatePrompt } from "./validation";

// Health check function for ComfyUI server
async function checkComfyHealth(): Promise<boolean> {
	try {
		const healthUrl = `${env.COMFY_URL}/system_stats`;

		// Use Promise.race to implement timeout
		const timeoutPromise = new Promise<never>((_, reject) => {
			setTimeout(() => reject(new Error("Health check timeout")), 5000);
		});

		const fetchPromise = fetch(healthUrl, {
			method: "GET",
		});

		const response = await Promise.race([fetchPromise, timeoutPromise]);

		const isHealthy = response.ok;
		if (!isHealthy) {
			console.error(
				`❌ ComfyUI health check failed (${response.status} ${response.statusText})`,
			);
		}

		return isHealthy;
	} catch (error) {
		console.error("❌ ComfyUI health check error:", error);
		return false;
	}
}

// Periodic health check function
let healthCheckInterval: Timer | null = null;

function startPeriodicHealthCheck(): void {
	// Clear any existing interval
	if (healthCheckInterval) {
		clearInterval(healthCheckInterval);
	}

	// Start periodic health checks every 10 seconds
	healthCheckInterval = setInterval(async () => {
		const isHealthy = await checkComfyHealth();
		if (!isHealthy) {
			console.warn("⚠️ ComfyUI server appears to be unhealthy");
		}
	}, 10000);

	console.log("🔄 Started periodic ComfyUI health checks (every 10s)");
}

function stopPeriodicHealthCheck(): void {
	if (healthCheckInterval) {
		clearInterval(healthCheckInterval);
		healthCheckInterval = null;
		console.log("🛑 Stopped periodic ComfyUI health checks");
	}
}

function badRequest(message: string, code?: number): Response {
	return new Response(JSON.stringify({ error: message }), {
		headers: { "content-type": "application/json" },
		status: code ?? 400,
	});
}

async function waitForComfySessionReady(
	userId: string,
	timeoutMs = 10_000,
): Promise<void> {
	const start = Date.now();
	// Poll session until sid is present and Comfy WS is open
	while (Date.now() - start < timeoutMs) {
		const session = getSession(userId);
		const comfyOpen = Boolean(
			session?.comfyWs &&
				(session.comfyWs as WebSocket & { readyState?: number }).readyState ===
					1,
		);
		if (session?.sid && comfyOpen) return;
		await new Promise((r) => setTimeout(r, 100));
	}
	throw new Error("Timeout establishing ComfyUI session");
}

async function handleSubmitForUser(userId: string, body: SubmitPromptBody) {
	const session = getSession(userId);
	if (!session?.comfyWs) throw new Error("WS session not ready");
	// Ensure ComfyUI session is fully ready (sid available)
	if (!session.sid) {
		await waitForComfySessionReady(userId, 10_000);
	}

	// Validate prompt before forwarding
	const validation = validatePrompt(body.prompt);
	if (!validation.valid) {
		throw new Error(`Invalid prompt: ${validation.error}`);
	}

	// Check circuit breaker
	if (!promptQueue.canProcess()) {
		// Try to queue the prompt instead
		const queuedPrompt = {
			extra_data: body.extra_data,
			partial_execution_targets: body.partial_execution_targets,
			prompt: body.prompt,
			prompt_id: body.prompt_id,
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
	const requestBody = {
		client_id: session.clientId,
		extra_data: body.extra_data,
		partial_execution_targets: body.partial_execution_targets,
		prompt: body.prompt,
		prompt_id: body.prompt_id,
	};
	const promptUrl = `${env.COMFY_URL}/prompt`;
	console.log(
		`📤 [HTTP] POST ${promptUrl} (user=${userId}, client_id=${session.clientId}, prompt_id=${body.prompt_id ?? "(auto)"})`,
	);
	const startedAt = Date.now();

	const response = await fetch(promptUrl, {
		body: JSON.stringify(requestBody),
		headers: { "Content-Type": "application/json" },
		method: "POST",
	});
	const elapsedMs = Date.now() - startedAt;
	console.log(
		`📥 [HTTP] Response ${response.status} ${response.statusText} in ${elapsedMs}ms (user=${userId})`,
	);

	if (!response.ok) {
		const errorData = (await response
			.json()
			.catch(() => ({ error: "Unknown error" }))) as { error?: string };
		circuitBreaker.recordFailure();
		console.error("❌ [HTTP] ComfyUI error response:", errorData);
		console.error(`❌ [HTTP] ComfyUI URL: ${promptUrl}`);
		console.error(
			`❌ [HTTP] ComfyUI server may be unreachable at ${env.COMFY_URL}`,
		);
		console.error("❌ [HTTP] Please ensure ComfyUI is running and accessible");
		throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
	}

	// Success - record it and process any queued prompts
	circuitBreaker.recordSuccess();
	let json: unknown;
	try {
		json = await response.json();
		console.log(`✅ [HTTP] /prompt OK (user=${userId})`);
	} catch (e) {
		console.warn("⚠️ [HTTP] /prompt OK but JSON parse failed:", e);
		json = {};
	}

	// Process queued prompts for this user
	processQueuedPrompts(userId);

	return json;
}

async function processQueuedPrompts(userId: string) {
	while (promptQueue.getQueueSize(userId) > 0 && promptQueue.canProcess()) {
		const queuedPrompt = promptQueue.getNextPrompt(userId);
		if (!queuedPrompt) break;

		try {
			await handleSubmitForUser(userId, queuedPrompt);
			console.log(`Processed queued prompt for user ${userId}`);
		} catch (error) {
			console.error(
				`Failed to process queued prompt for user ${userId}:`,
				error,
			);
			// Re-queue the prompt at the front
			promptQueue.addPrompt(userId, queuedPrompt);
			break;
		}
	}
}

type WsData = { userId: string; authorization?: string };
const wsHandler: WebSocketHandler<WsData> = {
	close(ws) {
		const extendedWs = ws as ExtendedServerWebSocket<WsData>;
		const userId = extendedWs.userId;
		if (!userId) return;

		console.log(`🔌 Closing session for user: ${userId}`);

		// Close the ComfyUI connection
		const session = getSession(userId);
		if (session?.comfyWs) {
			try {
				session.comfyWs.close();
			} catch (error) {
				console.error(
					`Error closing ComfyUI connection for user ${userId}:`,
					error,
				);
			}
		}

		clearSession(userId);
	},
	async message(ws, message) {
		const userId = ws.data?.userId;
		console.log(
			`📨 [WS←Client] Message handler called for user ${userId}, type: ${typeof message}`,
		);
		if (!userId) {
			console.log("⚠️ [WS←Client] No userId found in WebSocket, data:", ws.data);
			return;
		}

		// Update last active timestamp
		updateLastActive(userId);

		try {
			if (typeof message === "string") {
				console.log(
					`📨 [WS←Client] Received message from user ${userId}: ${message.substring(0, 200)}`,
				);
				let parsed: ProxyWsInbound;
				try {
					parsed = JSON.parse(message);
				} catch {
					console.log(
						`⚠️ [WS←Client] Failed to parse message from user ${userId}`,
					);
					return;
				}
				console.log(
					`🔎 [WS←Client] Parsed message type=${parsed.type} from user ${userId}`,
				);
				if (parsed.type === "submit_prompt") {
					console.log(
						`📤 [WS←Client] Processing submit_prompt from user ${userId}`,
					);

					// Wait for session to be ready
					const session = getSession(userId);
					if (!session?.comfyWs) {
						console.log(
							`⏳ Waiting for ComfyUI session to be ready for user ${userId}`,
						);
						try {
							await waitForComfySessionReady(userId, 10000);
						} catch (error) {
							console.error(
								`❌ Timeout waiting for ComfyUI session for user ${userId}:`,
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

					// Capture webhook configuration and ensure prompt_id is set
					const incoming = parsed.data as SubmitPromptBody;
					const finalPromptId =
						incoming.prompt_id ||
						`proxy-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
					incoming.prompt_id = finalPromptId;
					if (incoming.webhook_url) {
						// Validate webhook URL
						try {
							const webhookUrl = new URL(incoming.webhook_url);
							// Only allow http and https protocols
							if (
								webhookUrl.protocol !== "http:" &&
								webhookUrl.protocol !== "https:"
							) {
								throw new Error(
									`Invalid webhook URL protocol: ${webhookUrl.protocol}. Only http and https are allowed.`,
								);
							}
							// Reject localhost/private IPs in production (optional, uncomment if needed)
							// if (webhookUrl.hostname === "localhost" || webhookUrl.hostname.startsWith("127.") || webhookUrl.hostname.startsWith("192.168.")) {
							// 	throw new Error("Webhook URLs cannot point to localhost or private IPs");
							// }
						} catch (error) {
							if (error instanceof TypeError) {
								throw new Error(
									`Invalid webhook URL format: ${incoming.webhook_url}`,
								);
							}
							throw error;
						}

						const existingWebhooks = session?.webhooks || {};
						existingWebhooks[finalPromptId] = {
							secret: incoming.webhook_secret,
							url: incoming.webhook_url,
						};
						upsertSession(userId, { webhooks: { ...existingWebhooks } });
						console.log(
							`🔔 Registered webhook for prompt_id=${finalPromptId} user=${userId} → ${incoming.webhook_url}`,
						);
					}

					try {
						console.log(`🔄 Calling handleSubmitForUser for user ${userId}`);
						const res = await handleSubmitForUser(userId, incoming);
						console.log("✅ handleSubmitForUser completed, result:", res);
						ws.send(
							JSON.stringify({
								data: res,
								type: "prompt_accepted",
							} satisfies ProxyWsOutbound),
						);
						console.log(`➡️ [WS→Client] Sent prompt_accepted to user ${userId}`);
					} catch (error) {
						console.error(
							`❌ Error handling submit_prompt for user ${userId}:`,
							error,
						);
						const errorMsg = (error as Error).message;
						let errorCode: ErrorCode = ErrorCode.UNKNOWN_ERROR;
						let retryable = false;

						if (errorMsg.includes("invalid webhook")) {
							errorCode = ErrorCode.INVALID;
						} else if (
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
			console.log("🔌 WebSocket connection opened, verifying auth...");
			// Auth is already verified in the fetch handler before upgrade
			// Just use the userId from ws.data
			const userId = ws.data?.userId;
			if (!userId) {
				throw new Error("User ID not found in WebSocket data");
			}
			console.log(`🔐 Auth successful for user: ${userId}`);

			// Check if user can create a new session
			if (!canCreateSession(userId)) {
				throw new Error("Maximum connections per user exceeded");
			}

			// Acquire atomic lock to prevent race conditions
			if (!acquireSessionCreationLock(userId)) {
				throw new Error(
					"Another connection is being established for this user",
				);
			}

			// Create a dedicated ComfyUI WebSocket connection for this client
			console.log(
				`🔗 Creating dedicated ComfyUI connection for user: ${userId}`,
			);
			const clientId = crypto.randomUUID().replace(/-/g, "");
			const comfyWsUrl = `${env.COMFY_URL.replace(/^https?:\/\//, "ws://")}/ws?clientId=${clientId}`;
			console.log(`🌐 Connecting to ComfyUI: ${comfyWsUrl}`);

			const comfyConnection = new WebSocket(comfyWsUrl);

			// Set up message forwarding from ComfyUI to client
			comfyConnection.onmessage = (evt: MessageEvent) => {
				console.log(
					`📨 [WS←Comfy] Message for user ${userId}:`,
					typeof evt.data === "string"
						? (evt.data as string).substring(0, 200)
						: "binary data",
				);
				if (typeof evt.data === "string") {
					try {
						const msg = JSON.parse(evt.data as string) as ComfyWsMessage;
						if (msg?.type) {
							console.log(`🔎 [WS←Comfy] Parsed type=${msg.type}`);
						}
						if (
							msg.type === "status" &&
							(msg.data as { status?: unknown; sid?: string })?.sid
						) {
							const sid = (msg.data as { sid?: string }).sid;
							if (sid) {
								console.log(
									`🔑 Updating session with sid for user ${userId}: ${sid}`,
								);
								upsertSession(userId, { sid });
							}
						}

						// If execution finished, post webhook if configured for this prompt
						if (
							msg.type === "execution_success" ||
							msg.type === "execution_error"
						) {
							const session = getSession(userId);
							const promptId = (msg.data as { prompt_id?: string })?.prompt_id;
							const hook = promptId && session?.webhooks?.[promptId];
							if (hook) {
								(async () => {
									try {
										console.log(
											`📣 Posting webhook for prompt_id=${promptId} user=${userId} → ${hook.url}`,
										);
										const res = await fetch(hook.url, {
											body: JSON.stringify({
												clientId: session?.clientId,
												event: msg.type,
												userId,
												...((msg.data as Record<string, unknown>) || {}),
											}),
											headers: { "content-type": "application/json" },
											method: "POST",
										});
										const statusText = `${res.status} ${res.statusText}`;
										let bodySnippet = "<no-body>";
										try {
											const txt = await res.text();
											bodySnippet = txt.slice(0, 200);
										} catch {}
										console.log(
											`📣 Webhook response: ${statusText} body=${bodySnippet}`,
										);
										// Remove hook after posting
										const { [promptId]: _removed, ...rest } =
											session?.webhooks || {};
										upsertSession(userId, { webhooks: { ...rest } });
									} catch (err) {
										console.error("❌ Webhook post failed:", err);
									}
								})();
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
								console.error("❌ [EXECUTION ERROR]", {
									exceptionMessage: norm.exception_message,
									exceptionType: norm.exception_type,
									nodeId: norm.node_id,
									promptId: norm.prompt_id,
									traceback: norm.traceback,
									userId,
								});
								ws.send(
									createErrorJSON(
										norm.exception_message || "Execution error",
										ErrorCode.UNKNOWN_ERROR,
										{
											promptId: norm.prompt_id,
											userId,
										},
									),
								);
							} catch {}
						}
						// Relay to client
						ws.send(evt.data as string);
					} catch {
						ws.send(evt.data as string);
					}
				} else if (evt.data instanceof ArrayBuffer) {
					console.log(
						`📦 [WS←Comfy] Binary data (${evt.data.byteLength} bytes) for user ${userId}`,
					);
					ws.send(evt.data);
				}
			};

			comfyConnection.onclose = () => {
				console.log(`🔌 ComfyUI connection closed for user: ${userId}`);
				ws.close();
			};

			comfyConnection.onerror = (error) => {
				console.error(`❌ ComfyUI connection error for user ${userId}:`, error);
				console.error(`❌ ComfyUI URL: ${comfyWsUrl}`);
				console.error(
					`❌ ComfyUI server may be unreachable at ${env.COMFY_URL}`,
				);
				console.error("❌ Please ensure ComfyUI is running and accessible");
				ws.close();
			};

			// Wait for the ComfyUI connection to be established
			await new Promise<void>((resolve, reject) => {
				const timeout = setTimeout(() => {
					console.error(
						`⏰ Timeout establishing ComfyUI connection for user ${userId}`,
					);
					console.error(`⏰ ComfyUI URL: ${comfyWsUrl}`);
					console.error(
						`⏰ ComfyUI server may be unreachable at ${env.COMFY_URL}`,
					);
					console.error("⏰ Please ensure ComfyUI is running and accessible");
					reject(
						new Error(
							`Timeout establishing ComfyUI connection to ${comfyWsUrl}`,
						),
					);
				}, 10000);

				comfyConnection.onopen = () => {
					clearTimeout(timeout);
					console.log(`✅ ComfyUI connection established for user: ${userId}`);
					resolve();
				};

				comfyConnection.onerror = (error) => {
					clearTimeout(timeout);
					reject(error);
				};
			});

			// Create session with the dedicated connection
			const extendedWs = ws as ExtendedServerWebSocket<WsData>;
			upsertSession(userId, {
				clientId,
				clientWs: extendedWs,
				comfyWs: comfyConnection,
				connectionState: "connected",
			});
			extendedWs.userId = userId;
			console.log(
				`📝 Session created for user: ${userId} with dedicated ComfyUI connection`,
			);
			console.log(
				`✅ User ${userId} connected with dedicated ComfyUI connection`,
			);

			// Release the session creation lock now that session is established
			releaseSessionCreationLock(userId);
		} catch (e) {
			console.error("❌ Error in WebSocket open handler:", e);
			console.error("❌ Error stack:", (e as Error).stack);
			console.error(`❌ ComfyUI URL: ${env.COMFY_URL}`);
			console.error("❌ Please ensure ComfyUI is running and accessible");

			// Release the session creation lock on error
			const userId = ws.data?.userId;
			if (userId) {
				releaseSessionCreationLock(userId);
			}

			const errorMsg = (e as Error).message;
			let errorCode: ErrorCode = ErrorCode.UNKNOWN_ERROR;
			if (errorMsg.includes("connections")) {
				errorCode = ErrorCode.MAX_CONNECTIONS_EXCEEDED;
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

serve({
	async fetch(req, serverInstance) {
		const url = new URL(req.url);

		// Health check endpoints
		if (url.pathname === "/health" || url.pathname === "/") {
			return new Response(JSON.stringify({ ok: true }), {
				headers: { "content-type": "application/json" },
			});
		}

		// Live endpoint (process is alive)
		if (url.pathname === "/live") {
			return new Response(
				JSON.stringify({ status: "alive", uptime: process.uptime() }),
				{
					headers: { "content-type": "application/json" },
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
					headers: { "content-type": "application/json" },
					status: 200,
				},
			);
		}

		// Metrics endpoint with secret authentication
		if (url.pathname === "/metrics") {
			const auth = req.headers.get("authorization");
			const expectedSecret = env.METRICS_SECRET;

			if (auth !== `Bearer ${expectedSecret}`) {
				return badRequest("Unauthorized", 401);
			}

			const detailed = url.searchParams.get("detailed") === "true";
			const sessions = getAllSessions();

			const queueSizes = promptQueue.getAllQueueSizes();
			const metrics: MetricsResponse = {
				active_connections: getSessionCount(),
				active_sessions: getSessionCount(),
				circuit_breaker_state: circuitBreaker.getState(),
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
				headers: { "content-type": "application/json" },
			});
		}

		// Handle WebSocket upgrade requests
		if (url.pathname === "/ws") {
			// Extract auth from query parameters or headers
			let auth = req.headers.get("authorization") || undefined;
			const tokenParam = url.searchParams.get("token");
			if (!auth && tokenParam) {
				auth = `Bearer ${tokenParam}`;
			}

			try {
				const { userId } = await verifyAuthHeader(auth);

				// Perform the upgrade manually - this is required
				const upgraded = serverInstance.upgrade(req, {
					data: { authorization: auth, userId },
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
	port: env.PORT,
	websocket: wsHandler,
});

console.log(`Proxy listening on :${env.PORT} (-> ${env.COMFY_URL})`);
console.log(
	"🚀 Ready to accept connections with dedicated ComfyUI connections per client",
);

// Perform startup health check
(async () => {
	console.log("🏥 Performing startup health check...");
	const isHealthy = await checkComfyHealth();
	if (!isHealthy) {
		console.error(
			"❌ Startup health check failed - ComfyUI server may not be running",
		);
		console.error(
			`❌ Please ensure ComfyUI is accessible at: ${env.COMFY_URL}`,
		);
	} else {
		console.log("✅ Startup health check passed - ComfyUI server is running");
	}

	// Start periodic health checks
	startPeriodicHealthCheck();
})();

// Graceful shutdown
process.on("SIGINT", async () => {
	console.log("Received SIGINT, shutting down gracefully...");
	stopPeriodicHealthCheck();
	process.exit(0);
});

process.on("SIGTERM", async () => {
	console.log("Received SIGTERM, shutting down gracefully...");
	stopPeriodicHealthCheck();
	process.exit(0);
});
