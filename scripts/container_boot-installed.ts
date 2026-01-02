#!/usr/bin/env bun
/**
 * Container boot script for pre-installed image
 *
 * This boot script is used for the gpu-installed target where all code and extensions
 * are already installed in the image. It skips all installation steps and only handles
 * models persistence and supervisor startup.
 *
 * Responsibilities:
 * - Ensure models persistence: seed configs and link /comfy/models -> /models
 * - Start Supervisor which launches ComfyUI
 */

import { existsSync, mkdirSync, symlinkSync, renameSync, rmSync, cpSync, statSync } from "node:fs";
import { execSync } from "node:child_process";
import { generateSupervisorConfig } from "./generate_supervisor_config";

const CODE_DIR = "/comfy";

// Load .env file if it exists in CODE_DIR
const envFile = `${CODE_DIR}/.env`;
if (existsSync(envFile)) {
	console.log("[container_boot-installed] Loading environment variables from .env");
	// Bun automatically loads .env files
}

console.log(`[container_boot-installed] Running in ${CODE_DIR} (pre-installed)`);
console.log("[container_boot-installed] Skipping all installation steps");

// Ensure models mount and symlink
mkdirSync("/models", { recursive: true });

const codeModelsConfig = `${CODE_DIR}/models/config`;
const modelsConfig = "/models/config";
if (existsSync(codeModelsConfig) && !existsSync(modelsConfig)) {
	console.log("[container_boot-installed] Seeding /models/config from repo ...");
	cpSync(codeModelsConfig, modelsConfig, { recursive: true });
}

const codeModelsConfigs = `${CODE_DIR}/models/configs`;
const modelsConfigs = "/models/configs";
if (existsSync(codeModelsConfigs) && !existsSync(modelsConfigs)) {
	console.log("[container_boot-installed] Seeding /models/configs from repo ...");
	cpSync(codeModelsConfigs, modelsConfigs, { recursive: true });
}

// Use atomic symlink creation to avoid race conditions
const codeModels = `${CODE_DIR}/models`;
if (existsSync(codeModels)) {
	// Check if it's not a symlink
	try {
		const stats = statSync(codeModels);
		if (stats.isDirectory()) {
			rmSync(codeModels, { recursive: true, force: true });
		}
	} catch {
		// Ignore errors
	}
}

// Create symlink atomically
const tmpLink = `${codeModels}.tmp`;
try {
	if (existsSync(tmpLink)) {
		rmSync(tmpLink, { force: true });
	}
	symlinkSync("/models", tmpLink);
	renameSync(tmpLink, codeModels);
} catch (error) {
	console.error(`[container_boot-installed] Failed to create models symlink: ${error}`);
	process.exit(1);
}

// Generate supervisor config with all variables expanded
console.log(`[container_boot-installed] Generating supervisor config with CODE_DIR=${CODE_DIR}`);
const supervisorConfig = "/tmp/supervisord.conf";
const templateFile = `${CODE_DIR}/docker/supervisord.conf`;

generateSupervisorConfig(supervisorConfig, templateFile);

console.log("[container_boot-installed] Boot complete; starting Supervisor");
execSync(`supervisord -c ${supervisorConfig}`, { stdio: "inherit" });

