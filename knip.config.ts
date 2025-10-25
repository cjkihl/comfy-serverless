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
	],
	ignoreDependencies: [
		"@biomejs/biome",
		"@cjkihl/create-exports",
		"@cjkihl/tsconfig",
		"@cjkihl/with-env",
		"@manypkg/cli",
		"sort-package-json",
		"@types/bun",
		"@types/node",
		"typescript",
	],
	ignoreExportsUsedInFile: true,
	workspaces: {
		client: {
			entry: ["src/index.ts"],
			ignoreDependencies: ["bun-types"],
			project: [
				"src/**/*.{ts,tsx,js,jsx}",
				"!src/**/*.test.*",
				"!src/**/*.spec.*",
				"!tests/**",
			],
		},
		proxy: {
			entry: ["src/server.ts"],
			ignoreDependencies: ["@types/jsonwebtoken"],
			project: [
				"src/**/*.{ts,tsx,js,jsx}",
				"!src/**/*.test.*",
				"!src/**/*.spec.*",
				"!tests/**",
			],
		},
	},
};

export default config;
