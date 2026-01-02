import type { KnipConfig } from "knip";

const config: KnipConfig = {
	ignore: [
		"**/node_modules/**",
		"**/dist/**",
		"**/build/**",
		"**/.git/**",
		"**/coverage/**",
		"**/*.gen.*", // Ignore generated files
		"**/*.d.ts",
		"**/examples/**",
		"**/.pixi/**",
		// Docker and container scripts (used by shell scripts, not directly imported)
		"docker/**",
		"scripts/**",
		// Test files are entry points for test runners, not imported
		"**/tests/**/*.test.ts",
		"**/tests/**/*.spec.ts",
		"**/tests/**/fixtures.ts",
		"**/tests/**/performance-tracker.ts",
	],
	workspaces: {
		"cj/comfy-client": {
			project: [
				"**/*.{ts,tsx,js,jsx}",
			],
		},
		"cj/comfy-proxy": {
			entry: [
				"index.bin.ts",
				"tests/**/*.test.ts",
			],
			project: [
				"**/*.{ts,tsx,js,jsx}",
			],
		},
		"cj/comfy-web": {
			entry: [
				"src/router.tsx",
				"src/routes/**/*.tsx",
				"src/shared/**/*.ts",
				"scripts/**/*.ts",
				"tests/**/*.ts",
				"tests/**/*.spec.ts",
			],
			project: [
				"src/**/*.{ts,tsx,js,jsx,css}",
				"scripts/**/*.{ts,tsx}",
				"tests/**/*.{ts,tsx}",
			],
		},
	},
	ignoreExportsUsedInFile: true,
};

export default config;
