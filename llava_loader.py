from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForPreTraining,
    LlavaForConditionalGeneration
)
import numpy as np
import torch
import os
import requests
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import transformers
import sys

os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "~/.cache/huggingface/hub"
os.environ["HF_DATASETS_OFFLINE"] = "1"

save_dir = "inv_progress"
os.makedirs(save_dir, exist_ok=True)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def add_model_a_to_b(model_a, model_b):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    # Ensure keys match before subtraction
    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        raise ValueError("Model state dicts do not have the same keys.")

    for key in state_dict_a:
        if state_dict_a[key].shape != state_dict_b[key].shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': {state_dict_a[key].shape} vs {state_dict_b[key].shape}"
            )
        # Subtract model_a's weights from model_b for the matching key
        state_dict_b[key] = state_dict_b[key] + state_dict_a[key]

    # Update model_b with the new weights
    model_b.load_state_dict(state_dict_b)


output_checkpoint = "./models/llava_3_8b_cache_merged"  # set if you don't want to merge every time
hf_checkpoint = "xtuner/llava-llama-3-8b"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

processor = AutoProcessor.from_pretrained(hf_checkpoint)
model = LlavaForConditionalGeneration.from_pretrained(
    hf_checkpoint, dtype=torch.bfloat16, device_map=device
)

if model.language_model.model.embed_tokens.weight[-1].sum() == 0:
    print("adding llama3 weights")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )
    llama3 = pipeline.model
    add_model_a_to_b(llama3, model.language_model)
    if output_checkpoint:
        print("saving weights, so no adding is needed again")
        model.save_pretrained(output_checkpoint)
        processor.save_pretrained(output_checkpoint)

model.requires_grad_(False)
model.to(device)