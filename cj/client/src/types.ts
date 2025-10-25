export type Result<T, E = Error> =
	| { success: true; data: T }
	| { success: false; error: E };

export function ok<T>(data: T): Result<T> {
	return { data, success: true };
}

export function err<E = Error>(error: E): Result<never, E> {
	return { error, success: false };
}

// ComfyUI specific types (copied from proxy/src/types.ts)
export type ComfyNode = {
	inputs: Record<string, [string, number] | string | number>;
	class_type: string;
	_meta?: {
		title: string;
	};
};

export type ComfyPrompt = Record<string, ComfyNode>;

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

export type ExecutedMessage = {
	type: "executed";
	data: {
		node: string;
		display_node?: string;
		output: unknown;
		prompt_id: string;
	};
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

export type ErrorMessage = {
	type: "error";
	data: {
		message: string;
		code?: string;
	};
};

export type BinaryMessage = {
	type: "binary";
	data: ArrayBuffer;
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
