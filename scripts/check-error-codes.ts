#!/usr/bin/env bun

/**
 * Checks that ErrorCode definitions are kept in sync between proxy and client
 */

import { ErrorCode as ProxyErrorCode } from "../cj/proxy/src/types.js";
import { ErrorCode as ClientErrorCode } from "../cj/client/src/types.js";

function extractCodes(errorCode: Record<string, string>): Set<string> {
	return new Set(Object.values(errorCode));
}

const proxyCodes = extractCodes(ProxyErrorCode);
const clientCodes = extractCodes(ClientErrorCode);

const inProxyOnly = [...proxyCodes].filter((code) => !clientCodes.has(code));
const inClientOnly = [...clientCodes].filter((code) => !proxyCodes.has(code));

if (inProxyOnly.length > 0 || inClientOnly.length > 0) {
	console.error("❌ ErrorCode definitions are out of sync!");
	if (inProxyOnly.length > 0) {
		console.error(`   In proxy only: ${inProxyOnly.join(", ")}`);
	}
	if (inClientOnly.length > 0) {
		console.error(`   In client only: ${inClientOnly.join(", ")}`);
	}
	process.exit(1);
}

console.log("✅ ErrorCode definitions are in sync between proxy and client");
