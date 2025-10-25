import type { KnipConfig } from "knip";

const config: KnipConfig = {
	ignore: [
		"**/node_modules/**",
		"**/dist/**",
		"**/build/**",
		"**/.git/**",
		"**/coverage/**",
		"**/*.d.ts",
		"**/examples/**",
		"**/tests/**",
		"**/.pixi/**",
	],
	ignoreDependencies: [
		"@cjkihl/create-exports",
	],
	ignoreExportsUsedInFile: true,
	workspaces: {
		"cj/client": {
			entry: ["src/index.ts"],
			ignoreDependencies: ["bun-types"],
			project: [
				"src/**/*.{ts,tsx,js,jsx}",
				"tests/**/*.{ts,tsx,js,jsx}",
				"!src/**/*.test.*",
				"!src/**/*.spec.*",
			],
		},
		"cj/proxy": {
			entry: ["src/server.ts"],
			project: [
				"src/**/*.{ts,tsx,js,jsx}",
				"!src/**/*.test.*",
				"!src/**/*.spec.*",
			],
		},
	},
};

export default config;
