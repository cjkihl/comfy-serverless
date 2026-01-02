import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PlaywrightTestConfig } from "@playwright/test";

const PORT = 5173;

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, "test-results");

const config: PlaywrightTestConfig = {
	outputDir: OUTPUT_DIR,
	reporter: [["list"]],
	testDir: "./tests",
	testIgnore: [
		"**/session-management.spec.ts", // Bun test, not Playwright
		"**/bun-tests.ts", // Bun test
	],
	testMatch: /.*\.spec\.ts$/,
	timeout: 5 * 60 * 1000,
	use: {
		headless: true,
		screenshot: "only-on-failure",
		trace: "off",
		video: "off",
	},
	webServer: {
		command: `node node_modules/.bin/tsc && node node_modules/.bin/vite build && node node_modules/.bin/vite preview --port=${PORT}`,
		cwd: __dirname,
		port: PORT,
		reuseExistingServer: true,
	},
};

export default config;
