import { z } from "zod";

/**
 * Validation schemas for incoming data
 */

export const comfyNodeSchema = z.object({
	_meta: z
		.object({
			title: z.string(),
		})
		.optional(),
	class_type: z.string(),
	inputs: z.record(
		z.union([z.tuple([z.string(), z.number()]), z.string(), z.number()]),
	),
});

export const comfyPromptSchema = z
	.record(z.string(), comfyNodeSchema)
	.refine((data) => Object.keys(data).length > 0, {
		message: "Prompt must contain at least one node",
	});

export const submitPromptBodySchema = z.object({
	extra_data: z.record(z.unknown()).optional(),
	partial_execution_targets: z.array(z.string()).optional(),
	prompt: comfyPromptSchema,
	prompt_id: z.string().optional(),
	webhook_secret: z.string().optional(),
	webhook_url: z.string().url().optional(),
});

/**
 * Validates a ComfyUI prompt structure
 */
export function validatePrompt(prompt: unknown): {
	valid: boolean;
	error?: string;
} {
	try {
		comfyPromptSchema.parse(prompt);
		return { valid: true };
	} catch (error) {
		if (error instanceof z.ZodError) {
			return {
				error: error.errors
					.map((e) => `${e.path.join(".")}: ${e.message}`)
					.join(", "),
				valid: false,
			};
		}
		return {
			error: String(error),
			valid: false,
		};
	}
}

/**
 * Validates a submit prompt body
 */
export function validateSubmitPromptBody(body: unknown): {
	valid: boolean;
	error?: string;
} {
	try {
		submitPromptBodySchema.parse(body);
		return { valid: true };
	} catch (error) {
		if (error instanceof z.ZodError) {
			return {
				error: error.errors
					.map((e) => `${e.path.join(".")}: ${e.message}`)
					.join(", "),
				valid: false,
			};
		}
		return {
			error: String(error),
			valid: false,
		};
	}
}
