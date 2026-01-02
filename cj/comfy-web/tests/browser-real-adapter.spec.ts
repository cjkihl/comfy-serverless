import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, "../test-results");

test("runs real adapter suite in browser and saves artifacts", async ({
	page,
}) => {
	// Ensure output dir
	try {
		mkdirSync(OUTPUT_DIR, { recursive: true });
	} catch {}

	// Bridge artifact saving from the browser
	await page.exposeFunction(
		"saveArtifact",
		(name: string, base64: string, mime: string) => {
			const ext = (mime || "webp").toLowerCase();
			const file = join(OUTPUT_DIR, `${name}.${ext}`);
			writeFileSync(file, Buffer.from(base64, "base64"));
		},
	);

	// Listen to messages from the page
	page.on("console", (msg) => {
		// Mirror browser logs to test output (optional)
		// eslint-disable-next-line no-console
		console.log(`[browser] ${msg.type()}: ${msg.text()}`);
	});

	// biome-ignore lint/suspicious/noExplicitAny: Playwright requires loose types for exposed functions
	await page.exposeFunction("onSuiteMessage", (payload: any) => {
		if (payload?.type === "artifact") {
			// biome-ignore lint/suspicious/noExplicitAny: Playwright exposed function
			return (global as any).saveArtifact(
				payload.name,
				payload.base64,
				payload.mime,
			);
		}
		return undefined;
	});

	await page.addInitScript(() => {
		window.addEventListener("message", (ev: MessageEvent) => {
			// biome-ignore lint/suspicious/noExplicitAny: Playwright injected function
			if (typeof (window as any).onSuiteMessage === "function") {
				// biome-ignore lint/suspicious/noExplicitAny: Playwright injected function
				(window as any).onSuiteMessage(ev.data);
			}
		});
	});

	await page.goto("/tests/real-adapter");

	// Wait for completion signal
	await page
		.waitForEvent("console", {
			predicate: (msg) => msg.text().includes("suite-finished"),
			timeout: 5 * 60 * 1000,
		})
		.catch(() => null);

	// Fallback: also listen via page.evaluate handle
	const passed = await page.evaluate(async () => {
		return await new Promise<boolean>((resolve) => {
			const handler = (ev: MessageEvent) => {
				if (ev.data?.type === "suite-finished") {
					window.removeEventListener("message", handler);
					resolve(Boolean(ev.data.passed));
				}
			};
			window.addEventListener("message", handler);
		});
	});

	// Take a screenshot of the final page state
	const timestamp = Date.now();
	await page.screenshot({
		fullPage: true,
		path: join(OUTPUT_DIR, `browser-test-${timestamp}.png`),
	});

	expect(passed).toBe(true);
});
