import type { ProxyError, ProxyWsOutbound } from "./types";
import { ErrorCode } from "./types";

/**
 * Sanitizes error messages to prevent leaking backend implementation details.
 * Logs the detailed error server-side and returns a generic message to clients.
 */
export function sanitizeError(
	originalMessage: string,
	code: ErrorCode,
	options?: {
		userId?: string;
		promptId?: string;
		retryable?: boolean;
		context?: Record<string, unknown>;
	},
): ProxyWsOutbound {
	// Log the detailed error server-side for debugging
	console.error(`[ERROR CODE: ${code}]`, {
		context: options?.context,
		message: originalMessage,
		promptId: options?.promptId,
		userId: options?.userId,
	});

	// Map error codes to generic user-facing messages
	const sanitizedMessages: Record<ErrorCode, string> = {
		[ErrorCode.UNKNOWN_ERROR]:
			"An error occurred while processing your request",
		[ErrorCode.TIMEOUT]: "The request timed out",
		[ErrorCode.INVALID]: "Invalid request",
		[ErrorCode.MAX_CONNECTIONS_EXCEEDED]: "Maximum connections exceeded",
		[ErrorCode.QUEUE_FULL]: "Queue is full, please try again later",
		[ErrorCode.SESSION_NOT_READY]: "Session not ready, please try again",
	} as const satisfies Record<ErrorCode, string>;

	const error: ProxyError = {
		code,
		message: sanitizedMessages[code],
		retryable: options?.retryable ?? false,
		timestamp: Date.now(),
		...(options?.userId && { userId: options.userId }),
		...(options?.promptId && { promptId: options.promptId }),
		// Don't include context to avoid leaking sensitive information
	};

	return {
		data: error,
		type: "error",
	};
}

/**
 * Creates an error response for JSON stringification
 * Sanitizes error messages to prevent leaking backend implementation details
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
	return JSON.stringify(sanitizeError(message, code, options));
}
