import type { ComfyPrompt } from "@cj/comfy-client";
import type React from "react";
import { useEffect, useState } from "react";
import { Logger } from "../shared/logger";
import { getLightTestPrompt, getTestPrompt } from "../shared/test-prompt";

const logger = new Logger("info", "[PromptTester]");

// Helper to load image from a URL and return data URL (base64)

async function loadImageAsBase64(url: string): Promise<string> {
	return new Promise((resolve, reject) => {
		const img = new window.Image();
		img.crossOrigin = "Anonymous";
		img.onload = () => {
			const canvas = document.createElement("canvas");
			canvas.width = img.width;
			canvas.height = img.height;
			const ctx = canvas.getContext("2d");
			if (!ctx) {
				reject(new Error("Could not get 2D context"));
				return;
			}
			ctx.drawImage(img, 0, 0);
			const dataURL = canvas.toDataURL("image/png");
			resolve(dataURL);
		};
		img.onerror = (err) => {
			reject(new Error(`Failed to load image: ${url} - ${err}`));
		};
		img.src = url;
	});
}

const TEST_IMAGE_URL =
	"https://placehold.co/512/orange/white.png?text=Hello+World";

interface PromptTesterProps {
	onSubmitPrompt: (prompt: ComfyPrompt) => void;
	onPing: () => void;
	canSubmit: boolean;
	onDisplayImage?: (base64: string) => void;
}

export const PromptTester: React.FC<PromptTesterProps> = ({
	onPing,
	canSubmit,
	onSubmitPrompt,
	onDisplayImage,
}) => {
	const [testPrompt, setTestPrompt] = useState<ComfyPrompt | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const [testImageBase64, setTestImageBase64] = useState<string>("");

	useEffect(() => {
		let isMounted = true;

		loadImageAsBase64(TEST_IMAGE_URL)
			.then((base64) => {
				if (isMounted) {
					setTestImageBase64(base64);
					setTestPrompt(
						import.meta.env.VITE_LIGHT_PROMPT === "true"
							? getLightTestPrompt(base64)
							: getTestPrompt(base64),
					);
					setIsLoading(false);
				}
			})
			.catch((error) => {
				logger.error("Failed to load test image:", error);
				if (isMounted) {
					setIsLoading(false);
				}
			});

		return () => {
			isMounted = false;
		};
	}, []);

	return (
		<div className="mb-8">
			<div className="flex gap-3 flex-wrap">
				<button
					className="px-6 py-3 bg-indigo-500 text-white font-semibold rounded-lg transition-all hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
					disabled={!canSubmit || isLoading || !testPrompt}
					id="submit-test-prompt"
					onClick={() => {
						if (testPrompt) {
							onSubmitPrompt(testPrompt);
							// Proactively display the test image to confirm UI flow
							if (onDisplayImage && testImageBase64) {
								onDisplayImage(testImageBase64);
							}
						}
					}}
				>
					{isLoading ? "Loading..." : "Submit Test Prompt"}
				</button>
				<button
					className="px-6 py-3 bg-gray-500 text-white font-semibold rounded-lg transition-all hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
					disabled={!canSubmit}
					onClick={onPing}
				>
					Ping
				</button>
			</div>
		</div>
	);
};
