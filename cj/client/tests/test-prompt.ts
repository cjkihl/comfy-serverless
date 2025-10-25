import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ComfyPrompt } from "../src/types";

// Resolve test image path and convert to base64 for LoadImageBase64 node
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const imageBase64 = readFileSync(join(__dirname, "bear-kid.png")).toString(
	"base64",
);

export const testPrompt: ComfyPrompt = {
	"2": {
		_meta: {
			title: "Load Checkpoint",
		},
		class_type: "CheckpointLoaderSimple",
		inputs: {
			ckpt_name: "sdxl_lightning_2step.safetensors",
		},
	},
	"3": {
		_meta: {
			title: "KSampler",
		},
		class_type: "KSampler",
		inputs: {
			cfg: 1,
			denoise: 0.8,
			latent_image: ["13", 0],
			model: ["2", 0],
			negative: ["5", 0],
			positive: ["4", 0],
			sampler_name: "euler",
			scheduler: "sgm_uniform",
			seed: 147426702562207,
			steps: 4,
		},
	},
	"4": {
		_meta: {
			title: "CLIP Text Encode (Prompt)",
		},
		class_type: "CLIPTextEncode",
		inputs: {
			clip: ["2", 1],
			text: "Kids drawing of fluffy teddybear, cute",
		},
	},
	"5": {
		_meta: {
			title: "CLIP Text Encode (Prompt)",
		},
		class_type: "CLIPTextEncode",
		inputs: {
			clip: ["2", 1],
			text: "blurry, scarry, creepy",
		},
	},
	"7": {
		_meta: {
			title: "VAE Decode",
		},
		class_type: "VAEDecode",
		inputs: {
			samples: ["3", 0],
			vae: ["2", 2],
		},
	},
	"10": {
		_meta: {
			title: "VAE Encode",
		},
		class_type: "VAEEncode",
		inputs: {
			pixels: ["16", 0],
			vae: ["2", 2],
		},
	},
	"12": {
		_meta: {
			title: "Save Image Base64",
		},
		class_type: "SaveImageBase64",
		inputs: {
			images: ["7", 0],
			mime_type: "WEBP",
			quality: 100,
		},
	},
	"13": {
		_meta: {
			title: "Upscale Latent",
		},
		class_type: "LatentUpscale",
		inputs: {
			crop: "disabled",
			height: 1024,
			samples: ["10", 0],
			upscale_method: "nearest-exact",
			width: 1024,
		},
	},
	"16": {
		_meta: {
			title: "Load Image Base64",
		},
		class_type: "LoadImageBase64",
		inputs: {
			image_base64: imageBase64,
		},
	},
};
