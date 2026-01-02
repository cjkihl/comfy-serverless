import { createEnv } from "@t3-oss/env-core";
import { z } from "zod/v4";

/**
 * ⚠️ IMPORTANT: ComfyUI runs on a REMOTE server!
 * Set PROXY_COMFY_URL environment variable to your remote ComfyUI instance.
 * Example: export PROXY_COMFY_URL=http://remote-server:8188
 */
export const env = createEnv({
	emptyStringAsUndefined: true,

	runtimeEnv: process.env,
	server: {
		CIRCUIT_BREAKER_THRESHOLD: z.coerce.number().int().positive().default(5),
		CIRCUIT_BREAKER_TIMEOUT_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(30000),

		// Optional with defaults
		CLEANUP_INTERVAL_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(5 * 60 * 1000),
		CONNECTION_TIMEOUT_MS: z.coerce.number().int().positive().default(10000),
		HEALTH_CHECK_INTERVAL_MS: z.coerce.number().int().positive().default(10000),
		HEALTH_CHECK_TIMEOUT_MS: z.coerce.number().int().positive().default(5000),
		HTTP_REQUEST_TIMEOUT_MS: z.coerce.number().int().positive().default(30000),
		INITIAL_RETRY_DELAY_MS: z.coerce.number().int().positive().default(1000),
		LOG_LEVEL: z
			.enum(["debug", "info", "warn", "error", "silent"])
			.default("info"),
		MAX_CONNECTIONS_PER_USER: z.coerce.number().int().positive().default(5),
		MAX_PROMPT_RETRIES: z.coerce.number().int().positive().default(3),
		MAX_QUEUED_PROMPTS_PER_USER: z.coerce
			.number()
			.int()
			.positive()
			.default(100),
		MAX_RETRY_DELAY_MS: z.coerce.number().int().positive().default(30000),
		METRICS_SECRET: z.string().default("123456"),
		PROXY_COMFY_JWT_SECRET: z.string().min(1),
		PROXY_COMFY_RECONNECT_ENABLED: z.coerce.boolean().default(true),
		PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(1000),
		PROXY_COMFY_RECONNECT_MAX_DELAY_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(30000),
		PROXY_COMFY_RECONNECT_MAX_RETRIES: z.coerce
			.number()
			.int()
			.positive()
			.default(5),
		PROXY_COMFY_URL: z.url(),
		PROXY_PORT: z.coerce.number().int().positive().default(8190),
		SESSION_IDLE_EVICTION_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(5 * 60 * 1000),
		SESSION_READY_TIMEOUT_MS: z.coerce.number().int().positive().default(10000),
		SESSION_TIMEOUT_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(30 * 60 * 1000),
	},
});
