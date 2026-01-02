import type { JwtPayload } from "jsonwebtoken";
import jwt from "jsonwebtoken";

/**
 * Verify a JWT token using HMAC secret (HS256 algorithm)
 * @param token - The JWT token string
 * @param secret - The HMAC secret used to sign the token
 * @returns Decoded token with userId from 'sub' claim
 */
export async function verifyToken(
	token: string,
	secret: string,
): Promise<{ raw: Record<string, unknown>; userId: string }> {
	if (!secret) {
		throw new Error(
			"JWT verification requires PROXY_COMFY_JWT_SECRET to be set",
		);
	}

	const decoded = await new Promise<JwtPayload>((resolve, reject) => {
		const options: jwt.VerifyOptions = {
			algorithms: ["HS256"],
		};

		// Note: We intentionally do NOT validate issuer or audience
		// as per requirements - tokens can be from any issuer

		jwt.verify(token, secret, options, (err, decoded) => {
			if (err) return reject(err);
			if (!decoded || typeof decoded === "string")
				return reject(new Error("Invalid token payload"));
			resolve(decoded as JwtPayload);
		});
	});

	// Validate exp claim if present (no hardcoded duration enforcement)
	if (decoded.exp !== undefined) {
		const now = Math.floor(Date.now() / 1000);
		if (decoded.exp < now) {
			throw new Error("Token has expired");
		}
	}

	const sub = decoded.sub;
	if (!sub) throw new Error("Missing sub in JWT");
	return { raw: decoded as Record<string, unknown>, userId: sub };
}

/**
 * Verify an Authorization header containing a Bearer token
 * @param authHeader - The Authorization header value (e.g., "Bearer <token>")
 * @param secret - The HMAC secret used to sign the token
 * @returns Decoded token with userId from 'sub' claim
 */
export async function verifyAuthHeader(
	authHeader: string | undefined,
	secret: string,
): Promise<{ raw: Record<string, unknown>; userId: string }> {
	if (!authHeader) throw new Error("Missing Authorization header");
	const [scheme, token] = authHeader.split(" ");
	if (scheme !== "Bearer" || !token)
		throw new Error("Invalid Authorization header");
	return verifyToken(token, secret);
}

/**
 * Generate a JWT token signed with HS256
 * @param userId - The user ID to include in the 'sub' claim
 * @param secret - The HMAC secret used to sign the token
 * @param expiresInSeconds - Optional expiration time in seconds (default: 30)
 * @returns A signed JWT token string
 */
export function generateJWT(
	userId: string,
	secret: string,
	expiresInSeconds = 30,
): string {
	if (!secret) {
		throw new Error("PROXY_COMFY_JWT_SECRET is required to generate tokens");
	}

	const payload: JwtPayload = {
		exp: Math.floor(Date.now() / 1000) + expiresInSeconds,
		iat: Math.floor(Date.now() / 1000),
		sub: userId,
	};

	return jwt.sign(payload, secret, {
		algorithm: "HS256",
	});
}
