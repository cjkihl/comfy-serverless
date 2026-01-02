#!/usr/bin/env node

import { Command } from "commander";
import { env } from "./env";
import { Logger } from "./logger";
import {
	checkComfyHealth,
	type ProxyConfig,
	startPeriodicHealthCheck,
	startProxy,
	stopPeriodicHealthCheck,
} from "./proxy";

const program = new Command();

program
	.name("comfy-proxy")
	.description("ComfyUI Proxy Server")
	.version("0.1.0");

program
	.option("--comfyUrl <url>", "ComfyUI server URL (PROXY_COMFY_URL)")
	.option("--port <number>", "Proxy server port (PROXY_PORT)", (val: string) =>
		Number.parseInt(val, 10),
	)
	.option(
		"--jwtSecret <secret>",
		"JWT secret for authentication (PROXY_COMFY_JWT_SECRET)",
	)
	.option(
		"--metricsSecret <secret>",
		"Secret for metrics endpoint (METRICS_SECRET)",
	)
	.option(
		"--maxConnectionsPerUser <number>",
		"Max connections per user (MAX_CONNECTIONS_PER_USER)",
		(val: string) => Number.parseInt(val, 10),
	)
	.option(
		"--maxQueuedPromptsPerUser <number>",
		"Max queued prompts per user (MAX_QUEUED_PROMPTS_PER_USER)",
		(val: string) => Number.parseInt(val, 10),
	)
	.option(
		"--cleanupIntervalMs <number>",
		"Cleanup interval in milliseconds (CLEANUP_INTERVAL_MS)",
		(val: string) => Number.parseInt(val, 10),
	)
	.option(
		"--sessionIdleEvictionMs <number>",
		"Session idle eviction timeout in milliseconds (SESSION_IDLE_EVICTION_MS)",
		(val: string) => Number.parseInt(val, 10),
	)
	.option(
		"--sessionTimeoutMs <number>",
		"Session timeout in milliseconds (SESSION_TIMEOUT_MS)",
		(val: string) => Number.parseInt(val, 10),
	);

program.parse();

const options = program.opts();

// Merge env with CLI args (CLI args override env)
const config: ProxyConfig = {
	CIRCUIT_BREAKER_THRESHOLD: env.CIRCUIT_BREAKER_THRESHOLD,
	CIRCUIT_BREAKER_TIMEOUT_MS: env.CIRCUIT_BREAKER_TIMEOUT_MS,
	CLEANUP_INTERVAL_MS: options.cleanupIntervalMs ?? env.CLEANUP_INTERVAL_MS,
	CONNECTION_TIMEOUT_MS: env.CONNECTION_TIMEOUT_MS,
	HEALTH_CHECK_INTERVAL_MS: env.HEALTH_CHECK_INTERVAL_MS,
	HEALTH_CHECK_TIMEOUT_MS: env.HEALTH_CHECK_TIMEOUT_MS,
	HTTP_REQUEST_TIMEOUT_MS: env.HTTP_REQUEST_TIMEOUT_MS,
	INITIAL_RETRY_DELAY_MS: env.INITIAL_RETRY_DELAY_MS,
	LOG_LEVEL: env.LOG_LEVEL,
	MAX_CONNECTIONS_PER_USER:
		options.maxConnectionsPerUser ?? env.MAX_CONNECTIONS_PER_USER,
	MAX_PROMPT_RETRIES: env.MAX_PROMPT_RETRIES,
	MAX_QUEUED_PROMPTS_PER_USER:
		options.maxQueuedPromptsPerUser ?? env.MAX_QUEUED_PROMPTS_PER_USER,
	MAX_RETRY_DELAY_MS: env.MAX_RETRY_DELAY_MS,
	METRICS_SECRET: options.metricsSecret ?? env.METRICS_SECRET,
	PROXY_COMFY_JWT_SECRET: options.jwtSecret ?? env.PROXY_COMFY_JWT_SECRET,
	PROXY_COMFY_RECONNECT_ENABLED: env.PROXY_COMFY_RECONNECT_ENABLED,
	PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS:
		env.PROXY_COMFY_RECONNECT_INITIAL_DELAY_MS,
	PROXY_COMFY_RECONNECT_MAX_DELAY_MS: env.PROXY_COMFY_RECONNECT_MAX_DELAY_MS,
	PROXY_COMFY_RECONNECT_MAX_RETRIES: env.PROXY_COMFY_RECONNECT_MAX_RETRIES,
	PROXY_COMFY_URL: options.comfyUrl ?? env.PROXY_COMFY_URL,
	PROXY_PORT: options.port ?? env.PROXY_PORT,
	SESSION_IDLE_EVICTION_MS:
		options.sessionIdleEvictionMs ?? env.SESSION_IDLE_EVICTION_MS,
	SESSION_READY_TIMEOUT_MS: env.SESSION_READY_TIMEOUT_MS,
	SESSION_TIMEOUT_MS: options.sessionTimeoutMs ?? env.SESSION_TIMEOUT_MS,
};

// Create logger for health checks and shutdown
const logger = new Logger(config.LOG_LEVEL, "[Proxy]");

// Start the proxy server
startProxy(config);

// Perform startup health check
(async () => {
	logger.info("Performing startup health check...");
	const isHealthy = await checkComfyHealth(
		config.PROXY_COMFY_URL,
		config.HEALTH_CHECK_TIMEOUT_MS,
	);
	if (!isHealthy) {
		logger.error(
			"Startup health check failed - ComfyUI server may not be running",
		);
		logger.error(
			`Please ensure ComfyUI is accessible at: ${config.PROXY_COMFY_URL}`,
		);
		// Note: We don't block startup, but log the error
	} else {
		logger.info("Startup health check passed - ComfyUI server is running");
	}

	// Start periodic health checks
	startPeriodicHealthCheck(
		config.PROXY_COMFY_URL,
		config.HEALTH_CHECK_INTERVAL_MS,
		config.HEALTH_CHECK_TIMEOUT_MS,
		logger,
	);
})();

// Graceful shutdown
process.on("SIGINT", async () => {
	logger.info("Received SIGINT, shutting down gracefully...");
	stopPeriodicHealthCheck(logger);
	process.exit(0);
});

process.on("SIGTERM", async () => {
	logger.info("Received SIGTERM, shutting down gracefully...");
	stopPeriodicHealthCheck(logger);
	process.exit(0);
});
