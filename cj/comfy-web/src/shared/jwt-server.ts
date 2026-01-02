import { createServerFn } from "@tanstack/react-start";
import { generateJWT } from "../../../comfy-auth/index.pub";

/**
 * Get JWT secret from environment (server-side only)
 *
 * ⚠️ IMPORTANT: PROXY_COMFY_JWT_SECRET must be set in your environment.
 * This function runs on the server, so it's safe to access process.env.PROXY_COMFY_JWT_SECRET.
 *
 * @throws Error if PROXY_COMFY_JWT_SECRET is not set
 */
function getJWTSecret(): string {
	if (!process.env.PROXY_COMFY_JWT_SECRET) {
		throw new Error(
			"PROXY_COMFY_JWT_SECRET is not set. Please set it in your environment or use with-env to load from .env file.",
		);
	}
	return process.env.PROXY_COMFY_JWT_SECRET;
}

/**
 * Server function to generate a JWT token with the given userId in the 'sub' claim
 * This runs on the server where PROXY_COMFY_JWT_SECRET is safely accessible
 * Uses @cj/comfy-auth package for token generation
 *
 * @param userId - The user ID to include in the 'sub' claim
 * @param expiresInSeconds - Optional expiration time in seconds (default: 30)
 * @returns A signed JWT token string
 */
export const generateJWTToken = createServerFn({ method: "GET" })
	.inputValidator((data: { userId: string; expiresInSeconds?: number }) => {
		if (!data.userId || typeof data.userId !== "string") {
			throw new Error("userId is required and must be a string");
		}
		return data;
	})
	.handler(async ({ data }) => {
		const secret = getJWTSecret();
		const expiresInSeconds = data.expiresInSeconds ?? 30;
		const token = generateJWT(data.userId, secret, expiresInSeconds);
		return token;
	});
