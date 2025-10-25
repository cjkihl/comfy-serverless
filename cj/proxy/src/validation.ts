import { z } from "zod/v4";

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
		z.string(),
		z.union([z.tuple([z.string(), z.number()]), z.string(), z.number()]),
	),
});

export const comfyPromptSchema = z
	.record(z.string(), comfyNodeSchema)
	.refine((data) => Object.keys(data).length > 0, {
		message: "Prompt must contain at least one node",
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
				error: error.issues
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
