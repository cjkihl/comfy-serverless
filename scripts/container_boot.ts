#!/usr/bin/env bun
/**
 * Container boot script that runs inside the cloned repository (CODE_DIR)
 *
 * Responsibilities:
 * - Install dependencies using repo-provided installers or requirements.txt
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
	console.log("[container_boot] Loading environment variables from .env");
	// Bun automatically loads .env files, but we need to ensure it's loaded
	// The .env file will be loaded by Bun's runtime
}

console.log(`[container_boot] Running in ${CODE_DIR}`);

// Prefer repo-owned installer to keep logic versioned in the repo
const installScript = `${CODE_DIR}/scripts/install_requirements.sh`;
const requirementsFile = `${CODE_DIR}/requirements.txt`;

if (existsSync(installScript)) {
	console.log("[container_boot] Running install_requirements.sh via Pixi ...");
	execSync(`cd ${CODE_DIR} && pixi run bash -lc './scripts/install_requirements.sh'`, {
		stdio: "inherit",
	});
} else if (existsSync(requirementsFile)) {
	console.log("[container_boot] Installing requirements from requirements.txt via Pixi ...");
	execSync(`pixi run python -m pip install -r ${requirementsFile}`, {
		stdio: "inherit",
	});

	// Install custom_nodes requirements
	const customNodesDir = `${CODE_DIR}/custom_nodes`;
	if (existsSync(customNodesDir)) {
		console.log("[container_boot] Installing custom_nodes requirements ...");
		// Use find to locate all requirements.txt files in custom_nodes
		execSync(
			`find ${customNodesDir} -name 'requirements.txt' -type f -exec pixi run python -m pip install -r {} \\;`,
			{ stdio: "inherit" },
		);
	}
}

// Install proxy dependencies
const proxyDir = `${CODE_DIR}/cj/proxy`;
if (existsSync(proxyDir)) {
	console.log("[container_boot] Installing proxy dependencies...");
	execSync(`cd ${proxyDir} && bun install`, { stdio: "inherit" });
}

// Ensure models mount and symlink
mkdirSync("/models", { recursive: true });

const codeModelsConfig = `${CODE_DIR}/models/config`;
const modelsConfig = "/models/config";
if (existsSync(codeModelsConfig) && !existsSync(modelsConfig)) {
	console.log("[container_boot] Seeding /models/config from repo ...");
	cpSync(codeModelsConfig, modelsConfig, { recursive: true });
}

const codeModelsConfigs = `${CODE_DIR}/models/configs`;
const modelsConfigs = "/models/configs";
if (existsSync(codeModelsConfigs) && !existsSync(modelsConfigs)) {
	console.log("[container_boot] Seeding /models/configs from repo ...");
	cpSync(codeModelsConfigs, modelsConfigs, { recursive: true });
}

// Use atomic symlink creation to avoid race conditions
const codeModels = `${CODE_DIR}/models`;
if (existsSync(codeModels) && !existsSync(`${codeModels}/.git`)) {
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
	console.error(`[container_boot] Failed to create models symlink: ${error}`);
	process.exit(1);
}

// Generate supervisor config with all variables expanded
console.log(`[container_boot] Generating supervisor config with CODE_DIR=${CODE_DIR}`);
const supervisorConfig = "/tmp/supervisord.conf";
const templateFile = `${CODE_DIR}/docker/supervisord.conf`;

generateSupervisorConfig(supervisorConfig, templateFile);

console.log("[container_boot] Boot complete; starting Supervisor");
execSync(`supervisord -c ${supervisorConfig}`, { stdio: "inherit" });

