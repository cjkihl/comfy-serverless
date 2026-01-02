/**
 * Browser-specific test utilities
 * These utilities use browser APIs and cannot run in Node.js/Bun
 */

/**
 * Load image from URL and convert to base64 (without data: prefix)
 * @param url - Image URL to load
 * @returns Base64 string without data: prefix
 */
export async function loadImageAsBase64(url: string): Promise<string> {
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`Failed to load image: ${response.statusText}`);
	}
	const blob = await response.blob();
	return new Promise<string>((resolve, reject) => {
		const reader = new FileReader();
		reader.onloadend = () => {
			const result = reader.result;
			if (typeof result === "string") {
				// Remove data:image/...;base64, prefix if present
				const base64 = result.includes(",") ? result.split(",")[1] : result;
				resolve(base64 || "");
			} else {
				reject(new Error("Failed to convert image to base64"));
			}
		};
		reader.onerror = reject;
		reader.readAsDataURL(blob);
	});
}

/**
 * Default test image URL for browser tests
 */
export const TEST_IMAGE_URL =
	"https://placehold.co/512/orange/white.png?text=Hello+World";
