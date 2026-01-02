import { z } from "zod/v4";

/**
 * Zod schemas for validating ComfyUI message structures
 * Used for parsing and extracting image data from messages
 */

export const ImageDataSchema = z.object({ data: z.string().min(1).optional() });

export const OutputResultEntrySchema = z.object({
	image: z.string().min(1).optional(),
});

export const OutputSchema = z
	.object({
		images: z.array(ImageDataSchema).optional(),
		result: z.array(OutputResultEntrySchema).optional(),
	})
	.optional();

export const UiResultEntrySchema = z.object({
	image: z.string().min(1).optional(),
});

export const UiSchema = z
	.object({ result: z.array(UiResultEntrySchema).optional() })
	.optional();

export const ExecutedMessageSchema = z.object({
	data: z
		.object({
			output: OutputSchema.unwrap().optional(),
			ui: UiSchema.unwrap().optional(),
		})
		.catchall(z.unknown()),
	type: z.literal("executed"),
});

export const UiMessageSchema = z.object({
	data: z
		.object({ result: z.array(UiResultEntrySchema).optional() })
		.catchall(z.unknown()),
	type: z.literal("ui"),
});

export const BinaryMessageSchema = z.object({
	data: z.instanceof(ArrayBuffer),
	type: z.literal("binary"),
});

/**
 * Result type for explicit error handling
 *
 * All async operations return a Result type instead of throwing exceptions.
 * This enables explicit error handling and prevents uncaught exceptions.
 *
 * @example
 * ```typescript
 * const result = await client.connect();
 * if (result.success) {
 *   console.log('Connected:', result.data);
 * } else {
 *   console.error('Error:', result.error);
 * }
 * ```
 */
export type Result<T, E = Error> =
	| { success: true; data: T }
	| { success: false; error: E };

/**
 * Creates a successful Result value
 *
 * @param data - The success data
 * @returns Result with success: true
 */
export function ok<T>(data: T): Result<T> {
	return { data, success: true };
}

/**
 * Creates a failed Result value
 *
 * @param error - The error that occurred
 * @returns Result with success: false
 */
export function err<E = Error>(error: E): Result<never, E> {
	return { error, success: false };
}

// ComfyUI specific types (copied from proxy/src/types.ts)
export type ComfyInputNode = {
	inputs: Record<string, [string, number] | string | number>;
	class_type: string;
	_meta?: {
		title: string;
	};
};

export type ComfyPrompt = Record<string, ComfyInputNode>;

export type SubmitPromptBody = {
	prompt: ComfyPrompt;
	prompt_id?: string;
	extra_data?: Record<string, unknown>;
	partial_execution_targets?: string[];
	webhook_url?: string;
	webhook_secret?: string;
};

export type PromptAccepted = {
	prompt_id: string;
	number: number;
};

// Individual message types from ComfyUI
export type PromptAcceptedMessage = {
	type: "prompt_accepted";
	data: PromptAccepted;
};

export type StatusMessage = {
	type: "status";
	data: {
		status: {
			exec_info: {
				queue_remaining: number;
			};
		};
	};
};

export type ExecutingMessage = {
	type: "executing";
	data: {
		node: string | null;
		display_node?: string;
		prompt_id: string;
	};
};

// Output structure types for ExecutedMessage
export type ImageData = {
	data?: string;
};

export type OutputResultEntry = {
	image?: string;
};

export type NodeOutput = {
	images?: ImageData[];
	result?: OutputResultEntry[];
} & Record<string, unknown>;

export type UiResultEntry = {
	image?: string;
};

export type NodeUi = {
	result?: UiResultEntry[];
} & Record<string, unknown>;

export type ExecutedMessage = {
	type: "executed";
	data: {
		node: string;
		display_node?: string;
		output?: NodeOutput;
		ui?: NodeUi;
		prompt_id: string;
	} & Record<string, unknown>;
};

export type ProgressMessage = {
	type: "progress";
	data: {
		value: number;
		max: number;
		prompt_id?: string;
		node?: string;
	};
};

export type ProgressStateMessage = {
	type: "progress_state";
	data: {
		prompt_id: string;
		nodes: Record<string, unknown>;
	};
};

export type ExecutionErrorMessage = {
	type: "execution_error";
	data: {
		prompt_id: string;
		node_id: string;
		node_type: string;
		executed: string[];
		exception_message: string;
		exception_type: string;
		traceback: string[];
		current_inputs: unknown[];
		current_outputs: unknown[];
	};
};

export type ExecutionSuccessMessage = {
	type: "execution_success";
	data: {
		prompt_id: string;
	};
};

export type ExecutionInterruptedMessage = {
	type: "execution_interrupted";
	data: {
		prompt_id: string;
		node_id: string;
		node_type: string;
		executed: string[];
	};
};

export type ExecutionCachedMessage = {
	type: "execution_cached";
	data: {
		nodes: string[];
		prompt_id: string;
	};
};

export type ExecutionStartMessage = {
	type: "execution_start";
	data: {
		prompt_id: string;
	};
};

// Error codes matching the proxy (must be kept in sync with cj/proxy/src/types.ts)
export const ErrorCode = {
	INVALID: "INVALID",
	MAX_CONNECTIONS_EXCEEDED: "MAX_CONNECTIONS_EXCEEDED",
	QUEUE_FULL: "QUEUE_FULL",
	SESSION_NOT_READY: "SESSION_NOT_READY",
	TIMEOUT: "TIMEOUT",
	UNKNOWN_ERROR: "UNKNOWN_ERROR",
} as const;

export type ErrorCode = (typeof ErrorCode)[keyof typeof ErrorCode];

export type ErrorMessage = {
	type: "error";
	data: {
		message: string;
		code?: ErrorCode;
	};
};

export type BinaryMessage = {
	type: "binary";
	data: ArrayBuffer;
};

export type UiMessage = {
	type: "ui";
	data: {
		result?: UiResultEntry[];
	} & Record<string, unknown>;
};

// Discriminated union of all ComfyUI messages
export type ComfyMessage =
	| PromptAcceptedMessage
	| StatusMessage
	| ExecutingMessage
	| ExecutedMessage
	| ProgressMessage
	| ProgressStateMessage
	| ExecutionErrorMessage
	| ExecutionSuccessMessage
	| ExecutionInterruptedMessage
	| ExecutionCachedMessage
	| ExecutionStartMessage
	| ErrorMessage
	| UiMessage
	| BinaryMessage;

// Legacy type for backwards compatibility
export type ComfyWsMessage = { type: string; data: unknown };

export type ConnectionState =
	| "connecting"
	| "connected"
	| "disconnected"
	| "reconnecting";

export type EventCollection = {
	events: unknown[];
	binaryData: ArrayBuffer[];
	completed: boolean;
	error?: string;
};

export type SubmitOptions = {
	promptId?: string;
	webhookUrl?: string;
	webhookSecret?: string;
	extraData?: Record<string, unknown>;
};

export type CollectOptions = {
	timeout?: number;
	waitForCompletion?: boolean;
};

export type ReconnectConfig = {
	enabled: boolean;
	maxRetries: number;
	initialDelay: number;
	maxDelay: number;
	backoffMultiplier: number;
};

export type TimeoutConfig = {
	connect?: number;
	message?: number;
	operation?: number;
};

export type LogConfig = {
	level: "debug" | "info" | "warn" | "error" | "silent";
	prefix?: string;
};

export type HeartbeatConfig = {
	enabled: boolean;
	interval: number;
};
