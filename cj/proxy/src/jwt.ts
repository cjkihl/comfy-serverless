import jwt from "jsonwebtoken";
import jwksClient from "jwks-rsa";
import { env } from "./env";

// Only initialize JWKS client if JWT_JWKS_URL is provided
const client = env.JWT_JWKS_URL
	? jwksClient({
			cache: true,
			cacheMaxAge: 86400000,
			jwksUri: env.JWT_JWKS_URL,
		})
	: null;

function parseCommaSeparated(value?: string): string[] {
	if (!value) return [];
	return value
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);
}

async function getKey(header: jwt.JwtHeader): Promise<string> {
	if (!client) {
		throw new Error("JWKS client not initialized - JWT_JWKS_URL not provided");
	}
	const key = await client.getSigningKey(header.kid);
	return key.getPublicKey();
}

export async function verifyAuthHeader(authHeader?: string) {
	// Skip JWT verification for testing when NO_VERIFY=true
	if (env.NO_VERIFY) {
		if (!authHeader) throw new Error("Missing Authorization header");
		const [scheme, token] = authHeader.split(" ");
		if (scheme !== "Bearer" || !token)
			throw new Error("Invalid Authorization header");

		// Decode without verification - jwt.decode fails on malformed tokens
		// So we manually decode the payload
		const parts = token.split(".");
		if (parts.length !== 3) {
			throw new Error("Invalid JWT format");
		}
		const payload = JSON.parse(Buffer.from(parts[1]!, "base64").toString()) as jwt.JwtPayload;
		if (!payload?.sub) throw new Error("Missing sub in JWT");
		return { raw: payload as Record<string, unknown>, userId: payload.sub };
	}

	if (!authHeader) throw new Error("Missing Authorization header");
	const [scheme, token] = authHeader.split(" ");
	if (scheme !== "Bearer" || !token)
		throw new Error("Invalid Authorization header");
	return verifyToken(token);
}

export async function verifyToken(token: string) {
	if (!env.JWT_JWKS_URL) {
		throw new Error("JWT verification requires JWT_JWKS_URL to be set");
	}

	const decoded = await new Promise<jwt.JwtPayload>((resolve, reject) => {
		const options: jwt.VerifyOptions = {
			algorithms: parseCommaSeparated(env.JWT_ALG_ALLOWLIST) as jwt.Algorithm[],
		};

		const issuers = parseCommaSeparated(env.JWT_ISSUERS);
		if (issuers.length > 0) {
			options.issuer =
				issuers.length === 1 ? issuers[0] : (issuers as [string, ...string[]]);
		}

		const audiences = parseCommaSeparated(env.JWT_AUDIENCES);
		if (audiences.length > 0) {
			options.audience =
				audiences.length === 1
					? audiences[0]
					: (audiences as [string, ...string[]]);
		}

		jwt.verify(token, getKey, options, (err, decoded) => {
			if (err) return reject(err);
			if (!decoded || typeof decoded === "string")
				return reject(new Error("Invalid token payload"));
			resolve(decoded as jwt.JwtPayload);
		});
	});

	const sub = decoded.sub;
	if (!sub) throw new Error("Missing sub in JWT");
	return { raw: decoded as Record<string, unknown>, userId: sub };
}
