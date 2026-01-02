/**
 * Centralized environment variable definitions
 * This is the single source of truth for all environment variable names used by ComfyUI and the Proxy
 * Values come from the container environment (set via -e flags at runtime)
 */

/**
 * Environment variables used by ComfyUI
 */
export const COMFY_ENV_VARS = [
	"AWS_ACCESS_KEY_ID",
	"AWS_SECRET_ACCESS_KEY",
	"S3_ENDPOINT_URL",
	"COMFY_ARGS",
] as const;

/**
 * Environment variables used by the Proxy
 */
export const PROXY_ENV_VARS = [
	"PROXY_COMFY_URL",
	"PROXY_PORT",
	"MAX_CONNECTIONS_PER_USER",
	"METRICS_SECRET",
	"PROXY_COMFY_JWT_SECRET",
] as const;

