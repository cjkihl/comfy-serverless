import type { ComfyMessage } from "./types";
import {
	BinaryMessageSchema,
	ExecutedMessageSchema,
	UiMessageSchema,
} from "./types";

/**
 * Converts a binary ArrayBuffer to a base64 data URL.
 * Handles the ComfyUI binary message format (8-byte header + image data).
 *
 * @param buffer - The ArrayBuffer containing the binary image data
 * @returns A data URL string (data:image/png;base64,...) or null if conversion fails
 */
export function convertBinaryToDataUrl(buffer: ArrayBuffer): string | null {
	try {
		const uint8Array = new Uint8Array(buffer);
		if (uint8Array.byteLength <= 8) {
			return null;
		}

		// Skip the 8-byte header (ComfyUI binary message format)
		const imageBytes = uint8Array.subarray(8);

		// Convert to base64 in chunks to avoid stack overflow
		let binary = "";
		const chunkSize = 0x8000;
		for (let i = 0; i < imageBytes.length; i += chunkSize) {
			const chunk = imageBytes.subarray(i, i + chunkSize);
			binary += String.fromCharCode.apply(null, Array.from(chunk));
		}

		const base64 = btoa(binary);
		return `data:image/png;base64,${base64}`;
	} catch {
		return null;
	}
}

/**
 * Normalizes an image string to ensure it has a data: prefix.
 * If the string is already a data URL, it's returned as-is.
 * Otherwise, it's wrapped with data:image/webp;base64, prefix.
 *
 * @param imageData - The image data string (may or may not have data: prefix)
 * @returns A normalized data URL string
 */
export function normalizeImageDataUrl(imageData: string): string {
	if (imageData.startsWith("data:")) {
		return imageData;
	}
	return `data:image/webp;base64,${imageData}`;
}

/**
 * Extracts all image data URLs from an ExecutedMessage.
 * Handles multiple output formats:
 * - data.output.images (array of { data: string })
 * - data.output.result (array of { image: string }) - SaveImageBase64 format
 * - data.ui.result (array of { image: string })
 *
 * @param msg - The ExecutedMessage to extract images from
 * @returns An array of image data URL strings
 */
export function extractImagesFromExecuted(msg: unknown): string[] {
	const parsed = ExecutedMessageSchema.safeParse(msg);
	if (!parsed.success) {
		return [];
	}

	const images: string[] = [];
	const data = parsed.data.data;

	// 1. Extract from data.output.images
	const outputImages = data.output?.images;
	if (outputImages) {
		for (const image of outputImages) {
			if (image.data) {
				images.push(image.data);
			}
		}
	}

	// 2. Extract from data.output.result (SaveImageBase64 format)
	const outputResult = data.output?.result;
	if (outputResult && Array.isArray(outputResult)) {
		for (const item of outputResult) {
			if (
				item &&
				typeof item === "object" &&
				"image" in item &&
				typeof item.image === "string"
			) {
				images.push(normalizeImageDataUrl(item.image));
			}
		}
	}

	// 3. Extract from data.ui.result
	const uiResults = data.ui?.result;
	if (uiResults) {
		for (const entry of uiResults) {
			if (entry.image) {
				images.push(entry.image);
			}
		}
	}

	return images;
}

/**
 * Extracts all image data URLs from a UiMessage.
 *
 * @param msg - The UiMessage to extract images from
 * @returns An array of image data URL strings
 */
export function extractImagesFromUi(msg: unknown): string[] {
	const parsed = UiMessageSchema.safeParse(msg);
	if (!parsed.success) {
		return [];
	}

	const images: string[] = [];
	const uiResults = parsed.data.data.result;

	if (uiResults) {
		for (const entry of uiResults) {
			if (entry.image) {
				images.push(entry.image);
			}
		}
	}

	return images;
}

/**
 * Extracts all image data URLs from a BinaryMessage.
 *
 * @param msg - The BinaryMessage to extract images from
 * @returns An array containing a single image data URL, or empty array if conversion fails
 */
export function extractImagesFromBinary(msg: unknown): string[] {
	const parsed = BinaryMessageSchema.safeParse(msg);
	if (!parsed.success) {
		return [];
	}

	const dataUrl = convertBinaryToDataUrl(parsed.data.data);
	return dataUrl ? [dataUrl] : [];
}

/**
 * Extracts all image data URLs from any ComfyUI message type.
 * This is the main entry point for image extraction.
 *
 * @param msg - Any ComfyUI message
 * @returns An array of image data URL strings
 */
export function extractImagesFromMessage(
	msg: ComfyMessage | unknown,
): string[] {
	// Handle typed messages first
	if (typeof msg === "object" && msg !== null && "type" in msg) {
		const typedMsg = msg as { type: string };

		switch (typedMsg.type) {
			case "executed":
				return extractImagesFromExecuted(msg);
			case "ui":
				return extractImagesFromUi(msg);
			case "binary":
				return extractImagesFromBinary(msg);
			default:
				return [];
		}
	}

	// Fallback: try to parse as unknown message format
	// Try executed first (most common)
	const executedImages = extractImagesFromExecuted(msg);
	if (executedImages.length > 0) {
		return executedImages;
	}

	// Try UI message
	const uiImages = extractImagesFromUi(msg);
	if (uiImages.length > 0) {
		return uiImages;
	}

	// Try binary (less likely for unknown format, but try anyway)
	return extractImagesFromBinary(msg);
}
