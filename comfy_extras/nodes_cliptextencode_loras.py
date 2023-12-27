import re
from typing import List

from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP, load_lora_for_models
from comfy.utils import load_torch_file
import folder_paths


class CLIPTextEncodeLoras:
    def __init__(self):
        self.lora_regex = re.compile(r"<lora:([^>]+)>")
        self.whitespace_regex = re.compile(r",\s*,")
        self.lora_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "MODEL", "CLIP")
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, model, clip, text):
        loras, text = self.extract_loras(text)
        if loras:
            clip, _ = self.load_loras(loras, model, clip)

        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)

    def extract_loras(self, input_str: str):
        "Extract loras from input string and returns an array of loars and a cleaned prompt"
        loras = []
        cleaned_prompt = input_str
        for match in self.lora_regex.findall(input_str):
            if match:
                loras.append(match)

        cleaned_prompt = self.lora_regex.sub("", cleaned_prompt)
        cleaned_prompt = self.whitespace_regex.sub(",", cleaned_prompt)

        return loras, cleaned_prompt

    def load_loras(
        self, names: List[str] | None, model: ModelPatcher | None, clip: CLIP | None
    ):
        "Load loras into model and clip"
        if names is None:
            return (model, clip)

        model_with_loras = model
        clip_with_loras = clip
        for n in names:
            parts = n.split(":")
            lora_name = parts[0]
            strength_model = 1.0 if len(parts) < 2 else float(parts[1])
            strength_clip = 1.0 if len(parts) < 3 else float(parts[2])
            print("Applying lora", lora_name, strength_model, strength_clip)
            lora = self.lora_cache.get(lora_name)
            if lora is None:
                lora_path = folder_paths.get_full_path(
                    "loras", lora_name + ".safetensors"
                )
                if lora_path is None:
                    raise FileNotFoundError("Lora not found", lora_name)
                print("Loading lora", lora_name, strength_model, strength_clip)
                self.lora_cache[lora_name] = lora = load_torch_file(
                    lora_path, safe_load=True
                )

            (m, c) = load_lora_for_models(
                model_with_loras, clip_with_loras, lora, strength_model, strength_clip
            )
            model_with_loras = m
            clip_with_loras = c

        return (model_with_loras, clip_with_loras)


NODE_CLASS_MAPPINGS = {
    "CLIP Text Encode With Loras": CLIPTextEncodeLoras,
}
