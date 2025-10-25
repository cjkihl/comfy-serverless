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
