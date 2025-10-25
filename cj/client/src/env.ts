import { createEnv } from "@t3-oss/env-core";
import { z } from "zod/v4";

export const env = createEnv({
	emptyStringAsUndefined: true,
	runtimeEnv: process.env,
	server: {
		// Logging configuration
		CLIENT_LOG_LEVEL: z
			.enum(["debug", "info", "warn", "error", "silent"])
			.default("info"),
		CLIENT_LOG_PREFIX: z.string().default("[ComfyClient]"),
		CLIENT_RECONNECT_BACKOFF_MULTIPLIER: z.coerce
			.number()
			.positive()
			.default(2),

		// Reconnection configuration
		CLIENT_RECONNECT_ENABLED: z.coerce.boolean().default(true),
		CLIENT_RECONNECT_INITIAL_DELAY: z.coerce
			.number()
			.int()
			.positive()
			.default(1000),
		CLIENT_RECONNECT_MAX_DELAY: z.coerce
			.number()
			.int()
			.positive()
			.default(30000),
		CLIENT_RECONNECT_MAX_RETRIES: z.coerce.number().int().positive().default(5),

		// Client configuration
		CLIENT_TIMEOUT_CONNECT: z.coerce.number().int().positive().default(10000),
		CLIENT_TIMEOUT_MESSAGE: z.coerce.number().int().positive().default(30000),
		CLIENT_TIMEOUT_OPERATION: z.coerce
			.number()
			.int()
			.positive()
			.default(120000),

		// Development flags
		DEV_MODE: z.coerce.boolean().default(false),
		METRICS_SECRET: z.string().optional().default("123456"),
		METRICS_URL: z.url().optional().default("http://localhost:8190/metrics"),
		PROXY_HTTP_URL: z.url().optional().default("http://localhost:8190"),
		// Proxy server configuration (for E2E tests)
		PROXY_URL: z.url().optional().default("ws://localhost:8190/ws"),

		// Test configuration
		TEST_JWT_SECRET: z.string().optional().default("test-secret"),
		TEST_USER_PREFIX: z.string().optional().default("test-user"),
		VERBOSE_LOGGING: z.coerce.boolean().default(false),
	},
});
