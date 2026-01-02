import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, "../test-results");
const PORT = 5173;

// Ensure output directory exists
if (!existsSync(OUTPUT_DIR)) {
	mkdirSync(OUTPUT_DIR, { recursive: true });
}

test.describe("/auto route", () => {
	test("autoconnects, submits prompt, shows image, no console errors", async ({
		page,
	}) => {
		const consoleErrors: string[] = [];

		page.on("pageerror", (err) =>
			consoleErrors.push(`pageerror: ${err.message}`),
		);
		page.on("console", (msg) => {
			if (msg.type() === "error") consoleErrors.push(`console: ${msg.text()}`);
		});

		await page.goto(`http://localhost:${PORT}/auto`);

		// Wait for autoconnect via status data-attribute
		await expect(page.locator("#status-container")).toHaveAttribute(
			"data-status",
			/connected/,
		);

		// Click the submit test prompt button
		console.log("Clicking submit test prompt button...");
		await page.locator("#submit-test-prompt").click();

		console.log("Waiting for image to appear in gallery (max 5 seconds)...");
		// Wait for image to appear in gallery within 5 seconds
		const imageGallery = page.locator("#image-gallery");
		await expect(imageGallery).toContainText("Generated Image", {
			timeout: 5_000,
		});

		// Assert at least one image is visible
		const firstImage = page.locator("#image-gallery img").first();
		await expect(firstImage).toBeVisible({ timeout: 5_000 });

		// Save screenshot of the generated image
		const timestamp = Date.now();
		await firstImage.screenshot({
			path: join(OUTPUT_DIR, `generated-${timestamp}.png`),
		});

		console.log("Image appeared successfully, checking for errors...");

		// Ensure no console/page errors occurred
		if (consoleErrors.length > 0) {
			console.error("Console errors detected:", consoleErrors);
			throw new Error(
				`Test failed due to console errors: ${JSON.stringify(consoleErrors)}`,
			);
		}
	});
});
