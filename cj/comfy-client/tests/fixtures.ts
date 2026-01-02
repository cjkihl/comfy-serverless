/**
 * Shared test fixtures and mocks
 * Works in both Bun and Browser
 */

import { generateJWT } from "@cj/comfy-auth";
import { MockWebSocketAdapter } from "../adapters/mock.pub";
import type { ComfyClientConfig, ComfyPrompt } from "../index.pub";
import { ComfyClient } from "../index.pub";

/**
 * Get JWT secret from environment or use default test secret
 */
function getJWTSecret(): string {
	if (!process.env.PROXY_COMFY_JWT_SECRET) {
		throw new Error("PROXY_COMFY_JWT_SECRET is not set");
	}
	return process.env.PROXY_COMFY_JWT_SECRET;
}

/**
 * Create a test JWT token (properly signed HS256 JWT for testing)
 * Uses @cj/comfy-auth package for token generation
 */
function createTestJwt(): string {
	const secret = getJWTSecret();
	return generateJWT("test-user-123", secret, 30);
}

/**
 * Create a test client with default configuration
 */
export function createTestClient(
	overrides?: Partial<ComfyClientConfig>,
): ComfyClient {
	const adapter = new MockWebSocketAdapter();
	const defaultConfig: ComfyClientConfig = {
		adapter,
		auth: { jwt: createTestJwt() },
		autoConnect: false,
		heartbeat: { enabled: false }, // Disable heartbeat for tests
		logging: { level: "silent" },
		url: "ws://localhost:8188/ws",
	};

	return new ComfyClient({ ...defaultConfig, ...overrides });
}

/**
 * Create a simple test prompt. If an image data URL is provided, include it.
 *
 * @param imageBase64 - Optional image data URL (data:image/...). If provided, creates a LoadImageBase64 -> SaveImageWebsocket prompt.
 * @returns A ComfyUI prompt object
 */
export function createTestPrompt(): ComfyPrompt {
	// Minimal placeholder prompt for unit tests that don't hit the backend
	return {
		"1": {
			_meta: { title: "Empty" },
			class_type: "Empty",
			inputs: {},
		},
	};
}
