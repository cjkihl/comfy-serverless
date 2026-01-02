import { generateJWT } from "../../../comfy-auth/index.pub";
import { Logger } from "./logger";

const logger = new Logger("info", "[JWT]");

/**
 * Get JWT secret from environment
 *
 * ⚠️ IMPORTANT: PROXY_COMFY_JWT_SECRET must be set in your environment.
 * For testing, use with-env to load from .env file.
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
 * Generate a JWT token with the given userId in the 'sub' claim
 * This creates a properly signed HS256 JWT that can be verified
 * Uses @cj/comfy-auth package for token generation
 *
 * @param userId - The user ID to include in the 'sub' claim
 * @param expiresInSeconds - Optional expiration time in seconds (default: 30)
 * @returns A signed JWT token string
 */
export function generateTestJWT(userId: string, expiresInSeconds = 30): string {
	logger.info(`Generating JWT for user: ${userId}`);

	const secret = getJWTSecret();
	const token = generateJWT(userId, secret, expiresInSeconds);
	logger.debug(
		`Generated token for user ${userId}: ${token.substring(0, 50)}...`,
	);
	return token;
}
