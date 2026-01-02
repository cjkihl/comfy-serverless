#!/usr/bin/env bun

/**
 * Bun-specific test utilities
 * These utilities use Node.js/Bun APIs and cannot run in browser
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Resolve test image path
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Cache for loaded image to avoid repeated I/O
let cachedImageBase64: string | null = null;

/**
 * Load the test image (bear-kid.png) from file system and convert to base64
 * @returns Base64 string without data: prefix
 */
export function loadTestImageBase64(): string {
	if (!cachedImageBase64) {
		const imageBuffer = readFileSync(join(__dirname, "bear-kid.png"));
		const bytes = new Uint8Array(imageBuffer);
		let binary = "";
		for (let i = 0; i < bytes.byteLength; i++) {
			binary += String.fromCharCode(bytes[i]!);
		}
		cachedImageBase64 = btoa(binary);
	}
	return cachedImageBase64;
}
