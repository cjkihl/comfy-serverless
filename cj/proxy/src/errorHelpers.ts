import type { ErrorCode, ProxyError, ProxyWsOutbound } from "./types";

/**
 * Creates a standardized error response for WebSocket clients
 */
export function createErrorResponse(
	message: string,
	code: ErrorCode,
	options?: {
		userId?: string;
		promptId?: string;
		retryable?: boolean;
		context?: Record<string, unknown>;
	},
): ProxyWsOutbound {
	const error: ProxyError = {
		code,
		message,
		retryable: options?.retryable ?? false,
		timestamp: Date.now(),
		...(options?.userId && { userId: options.userId }),
		...(options?.promptId && { promptId: options.promptId }),
		...(options?.context && { context: options.context }),
	};

	return {
		data: error,
		type: "error",
	};
}

/**
 * Creates an error response for JSON stringification
 */
export function createErrorJSON(
	message: string,
	code: ErrorCode,
	options?: {
		userId?: string;
		promptId?: string;
		retryable?: boolean;
		context?: Record<string, unknown>;
	},
): string {
	return JSON.stringify(createErrorResponse(message, code, options));
}
