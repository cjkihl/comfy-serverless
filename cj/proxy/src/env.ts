import { createEnv } from "@t3-oss/env-core";
import { z } from "zod";

/**
 * ⚠️ IMPORTANT: ComfyUI runs on a REMOTE server!
 * Set COMFY_URL environment variable to your remote ComfyUI instance.
 * Example: export COMFY_URL=http://remote-server:8188
 */
export const env = createEnv({
	emptyStringAsUndefined: true,

	runtimeEnv: process.env,
	server: {
		CLEANUP_INTERVAL_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(5 * 60 * 1000),
		// Required - Remote ComfyUI server URL
		COMFY_URL: z.url(),
		JWT_ALG_ALLOWLIST: z.string().default("EdDSA"),
		JWT_AUDIENCES: z.string().min(1).optional(),
		JWT_ISSUERS: z.string().min(1).optional(),
		JWT_JWKS_URL: z.string().url().optional(),
		MAX_CONNECTIONS_PER_USER: z.coerce.number().int().positive().default(1),
		MAX_QUEUED_PROMPTS_PER_USER: z.coerce
			.number()
			.int()
			.positive()
			.default(100),
		METRICS_SECRET: z.string().default("123456"),
		NO_VERIFY: z.coerce.boolean().default(false),

		// Optional with defaults
		PORT: z.coerce.number().int().positive().default(8190),

		// Test environment variables
		SESSION_TIMEOUT_MS: z.coerce
			.number()
			.int()
			.positive()
			.default(30 * 60 * 1000),
	},
});
