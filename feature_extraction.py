# --------------------------------------------------------
# Inversion of Visual Language Models paper
# Animesh Jain
# --------------------------------------------------------
import argparse
import gc
import sys
import random
import os
import traceback
import numpy as np
import time
import json

import torch

from transformers import AutoProcessor, AutoModelForPreTraining

from accelerate.utils import set_seed

from helpers.utils import get_target_features_filepath, load_json
from helpers.hooks import FeatureStatsHookManager
from custom_datasets.imagenet import ImageNetDataloader, ImageNetDataset
from helpers.feature_extractor import FeatureExtractor


# Reproducibility
def set_reproducibility(seed=0):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed, device_specific=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants
PROMPT_FILEPATH = "./data/prompts.json"
PROMPT_LISTS = load_json(PROMPT_FILEPATH)
AVALIABLE_CLASSES = [c["class_index"] for c in PROMPT_LISTS]


def get_precision_settings(fp16):
    if fp16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        else:
            return torch.float16, "fp16"
    else:
        return torch.float32, "no"


def run(args):
    data_type, _ = get_precision_settings(args.fp16)
    dataset_dir = args.dataset_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths and experiment/data config
    target_class_prompts = (
        [d for d in PROMPT_LISTS if d["class_index"] == args.target_class]
        if args.target_class is not None
        else PROMPT_LISTS
    )

    os.makedirs(args.target_features_dir, exist_ok=True)

    print("Loading model for feature extraction...")

    vlm_checkpoint = args.vlm_checkpoint
    vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint)
    vlm_model = AutoModelForPreTraining.from_pretrained(vlm_checkpoint)
    # vlm_model.gradient_checkpointing_enable()
    vlm_model.config.use_cache = False
    vlm_processor.patch_size = vlm_model.config.vision_config.patch_size
    vlm_processor.image_processor.size = {
        "shortest_edge": 336
    }  # Needs to be fixed for LLaVA
    vlm_model = vlm_model.to(device=device, dtype=data_type)
    vlm_model.eval()

    base_hook_manager = FeatureStatsHookManager(model=vlm_model, model_type="vlm")
    base_feature_hooks = base_hook_manager.register_hooks()

    print("\n------------ Feature extraction started ------------")
    start_time = time.time()
    tfe = FeatureExtractor(
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        feature_hooks=base_feature_hooks,
        device=device,
        data_type=data_type,
    )
    for prompt in target_class_prompts:
        target_class = prompt["class_index"]
        target_features_filepath = get_target_features_filepath(
            args.target_features_dir, target_class
        )
        if not os.path.exists(target_features_filepath):
            print(
                f"Target feature stats not found at {target_features_filepath}. Computing..."
            )

            dataset = ImageNetDataset(
                class_index=target_class,
                root_dir=dataset_dir,
                split="train",
            )
            dataloader = ImageNetDataloader(dataset=dataset)
            features = tfe.compute(dataloader=dataloader)

            # Save target feature stats only once on main process
            torch.save(features, target_features_filepath)
            print(f"Target features computed and saved to {target_features_filepath}")
            base_hook_manager.reset_hooks()
        else:
            print(
                f"Target feature stats already exist at {target_features_filepath}. Skipping computation."
            )

    end_time = time.time()
    print("\n------------ Feature extraction ended ------------")
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    del tfe
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        required=True,
        help="Path to the VLM checkpoint. Default: ./models/llava_3_8b_cache_merged",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Imagenet dataset directory.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for optimization, if selected BF16 will be preferred",
    )
    parser.add_argument(
        "--target_features_dir",
        type=str,
        default="./target_features",
        help="Directory to store computed target feature stats. Default: ./target_features",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        required=False,
        choices=[c["class_index"] for c in PROMPT_LISTS],
        help="Target label from the ImageNet classes. If not specified, all classes from './data/prompts.json' will be used",
    )
    args = parser.parse_args()
    set_reproducibility(args.seed)
    run(args)


if __name__ == "__main__":
    main()
