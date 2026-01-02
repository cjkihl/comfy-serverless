#!/usr/bin/env bun
/**
 * Generate supervisor config from template
 * Uses centralized env_vars.ts as single source of truth for variable names
 * All values come from container environment (set via -e flags)
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { COMFY_ENV_VARS, PROXY_ENV_VARS } from "../docker/env_vars";

/**
 * Escape value for supervisord environment directive
 * Format: VAR="value" - need to escape quotes and backslashes
 */
function escapeEnvValue(value: string): string {
	// Escape backslashes first
	let escaped = value.replace(/\\/g, "\\\\");
	// Escape double quotes
	escaped = escaped.replace(/"/g, '\\"');
	return escaped;
}

/**
 * Build environment string from array of variable names
 */
function buildEnvString(
	varNames: readonly string[],
	env: NodeJS.ProcessEnv,
): string {
	const parts: string[] = [];

	for (const varName of varNames) {
		const value = env[varName];
		if (value !== undefined && value !== "") {
			const escapedValue = escapeEnvValue(value);
			parts.push(`${varName}="${escapedValue}"`);
		}
	}

	return parts.join(",");
}

/**
 * Find env_vars.ts file in multiple possible locations
 */
function findEnvVarsFile(codeDir: string): string | null {
	const possiblePaths = [
		"/docker/env_vars.ts",
		`${codeDir}/docker/env_vars.ts`,
	];

	for (const path of possiblePaths) {
		if (existsSync(path)) {
			return path;
		}
	}

	return null;
}

/**
 * Generate supervisor config from template
 */
function generateSupervisorConfig(
	supervisorConfig: string,
	templateFile: string,
	env: NodeJS.ProcessEnv = process.env,
): void {
	if (!existsSync(templateFile)) {
		console.error(`Error: Template file not found: ${templateFile}`);
		process.exit(1);
	}

	const codeDir = "/comfy";

	// Verify env_vars.ts exists (it should be imported, but check for clarity)
	const envVarsFile = findEnvVarsFile(codeDir);
	if (!envVarsFile) {
		console.error(
			`Error: env_vars.ts not found. Expected at /docker/env_vars.ts or ${codeDir}/docker/env_vars.ts`,
		);
		process.exit(1);
	}

	// Build ComfyUI environment string
	const comfyEnvString = buildEnvString(COMFY_ENV_VARS, env);

	// Log ComfyUI variables being loaded (names only, not values)
	const comfyVarsLoaded = COMFY_ENV_VARS.filter(
		(varName) => env[varName] !== undefined && env[varName] !== "",
	);
	if (comfyVarsLoaded.length > 0) {
		console.log(`[env] Loading ComfyUI variables: ${comfyVarsLoaded.join(", ")}`);
	} else {
		console.log("[env] No ComfyUI variables loaded");
	}

	// Build proxy environment string
	const proxyEnvString = buildEnvString(PROXY_ENV_VARS, env);

	// Log proxy variables being loaded (names only, not values)
	const proxyVarsLoaded = PROXY_ENV_VARS.filter(
		(varName) => env[varName] !== undefined && env[varName] !== "",
	);
	if (proxyVarsLoaded.length > 0) {
		console.log(`[env] Loading proxy variables: ${proxyVarsLoaded.join(", ")}`);
	} else {
		console.log("[env] No proxy variables loaded");
	}

	// Validate required variables
	if (!env.PROXY_COMFY_URL) {
		console.error(
			"[env] WARNING: PROXY_COMFY_URL is not set. Proxy will fail at startup.",
		);
	}

	// Set defaults for variables used in template expansion
	const codeDirValue = "/comfy";
	const comfyArgs = env.COMFY_ARGS || "";

	// Read template
	let template = readFileSync(templateFile, "utf-8");

	// Replace placeholders
	template = template.replace(/__COMFY_ENV_VARS__/g, comfyEnvString);
	template = template.replace(/__PROXY_ENV_VARS__/g, proxyEnvString);
	template = template.replace(/\$\{CODE_DIR:-\/comfy\}/g, codeDirValue);
	template = template.replace(/\$\{COMFY_ARGS\}/g, comfyArgs);

	// Write output
	writeFileSync(supervisorConfig, template, "utf-8");
}

// If run directly (not imported), execute the function
if (import.meta.main) {
	const supervisorConfig = process.argv[2] || "/tmp/supervisord.conf";
	const templateFile = process.argv[3] || "/etc/supervisord.conf";

	generateSupervisorConfig(supervisorConfig, templateFile);
}

export { generateSupervisorConfig };
