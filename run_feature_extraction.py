# --------------------------------------------------------
# Inversion of Visual Language Models paper
# Animesh Jain
# --------------------------------------------------------
import argparse
import gc
import sys
import os
import time

import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm

from accelerate.utils import set_seed

from helpers.utils import load_vlm, VLM_CHOICES, VLM_LAYER_PATTERNS, get_image_dataloader
from helpers.hooks import FeatureStatsHookManager

# Reproducibility
def set_reproducibility(seed=0):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_seed(seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_precision_settings(fp16):
    if fp16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        else:
            print("Warning: BF16 not supported, using FP16")
            return torch.float16, "fp16"
    else:
        return torch.float32, "no"


class RunningStats:
    """Computes running mean and variance using Welford's algorithm or simple accumulation."""
    def __init__(self, shape, device, dtype):
        self.n = 0
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        
        self.sum_mean = torch.zeros(shape, device=device, dtype=dtype)
        self.sum_var = torch.zeros(shape, device=device, dtype=dtype)

    def update(self, batch_means, batch_vars):
        """
        batch_means: [B, D]
        batch_vars: [B, D]
        """
        batch_size = batch_means.shape[0]
        self.n += batch_size
        self.sum_mean += batch_means.sum(dim=0)
        self.sum_var += batch_vars.sum(dim=0)

    def get_stats(self):
        if self.n == 0:
            return self.mean, self.mean # zeros
        return self.sum_mean / self.n, self.sum_var / self.n


def run(args):
    data_type, _ = get_precision_settings(args.fp16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Output Directory
    dataset_name = args.images_dir.rstrip('/').split('/')[-1]
    output_dir = os.path.join(args.target_dir, args.vlm, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load Model
    print(f"Loading {args.vlm}...")
    vlm_processor, _, model = load_vlm(args.vlm)
    model.config.use_cache = False
    model.eval()
    
    # Identify Vision Tower
    vision_tower = None
    if hasattr(model, 'vision_tower'):
        vision_tower = model.vision_tower
    elif hasattr(model, 'visual'):
        vision_tower = model.visual
    elif hasattr(model, 'model'):
         if hasattr(model.model, 'vision_tower'):
             vision_tower = model.model.vision_tower
         elif hasattr(model.model, 'visual'):
             vision_tower = model.model.visual
    
    if vision_tower is None:
        print("Could not identify vision tower automatically. Using full model with hack.")
        print("Dumping modules...")
        for n, m in model.named_children():
            print(n)
        sys.exit("Vision tower not found. Update script.")

    # Move vision tower to device
    vision_tower = vision_tower.to(device=device, dtype=data_type)
    
    layer_regex = VLM_LAYER_PATTERNS.get(args.vlm)
    if not layer_regex:
        sys.exit(f"No regex pattern defined for {args.vlm}")

    # Adjust Regex for Vision Tower only
    
    prefixes_to_strip = [r"vision_tower\.", r"visual\."]
    for prefix in prefixes_to_strip:
        if layer_regex.startswith(prefix):
            layer_regex = layer_regex[len(prefix):] # This is rough, as it's a regex 
            pass
    
    clean_regex = layer_regex.replace(r"vision_tower\.", "").replace(r"visual\.", "")
    print(f"Original Regex: {layer_regex}")
    print(f"Vision-Tower Regex: {clean_regex}")

    # Setup Hooks on Vision Tower
    hook_manager = FeatureStatsHookManager(model=vision_tower, model_type="vlm", layer_regex=clean_regex)
    feature_hooks = hook_manager.register_hooks()
    print(f"Registered {len(feature_hooks)} feature hooks.")
    
    if len(feature_hooks) == 0:
        print("Use Inspect Layers script to check module names of:")
        for name, _ in vision_tower.named_modules():
            print(name)
            break
        sys.exit("No hooks registered! Check regex.")

    # Dataloader
    # Return PIL images so we can use processor correctly for complex models like Qwen
    dataloader = get_image_dataloader(images_dir=args.images_dir, batch_size=args.batch_size, image_size=336, return_pil=True)
    print(f"Found {len(dataloader.dataset)} images.")

    # Running Stats Holders
    # We can't init until we know the dimension D.
    stats_holders = [None] * len(feature_hooks)

    print("\n------------ Feature extraction started ------------")
    start_time = time.time()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features")
    
    with torch.no_grad():
        for i, batch_images in pbar:
            # batch_images is a list of PIL images
            try:
                inputs = vlm_processor(images=batch_images, text=[""]*len(batch_images), return_tensors="pt")
            except Exception:
                # Fallback if text is required strictly
                inputs = vlm_processor(images=batch_images, return_tensors="pt")

            inputs = inputs.to(device=device, dtype=data_type)

            # Extract arguments for vision tower
            forward_kwargs = {}
            if "qwen" in args.vlm.lower():
                 if hasattr(inputs, "pixel_values"): forward_kwargs["hidden_states"] = inputs.pixel_values
                 if hasattr(inputs, "image_grid_thw"): forward_kwargs["grid_thw"] = inputs.image_grid_thw
            else:
                 if hasattr(inputs, "pixel_values"): forward_kwargs["pixel_values"] = inputs.pixel_values
                 elif hasattr(inputs, "images"): forward_kwargs["images"] = inputs.images
            
            # Forward pass (Vision Tower only)
            try:
                if forward_kwargs:
                   _ = vision_tower(**forward_kwargs)
                else:
                   # Try passing directly if processor output weird
                    _ = vision_tower(inputs)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                sys.exit(1)

            # Collect Stats
            for h_idx, hook in enumerate(feature_hooks):
                if hook.data and hook.data.get("r_feature") is not None:
                    # [2, B, D]
                    mean_b, var_b = hook.data["r_feature"]
                    
                    # Init holder if needed
                    if stats_holders[h_idx] is None:
                        D = mean_b.shape[-1]
                        stats_holders[h_idx] = RunningStats([D], device, dtype=torch.float32) # Calc in fp32
                    
                    # Update
                    stats_holders[h_idx].update(mean_b.to(torch.float32), var_b.to(torch.float32))
                    
                    # Clear hook data to save memory
                    hook.data = None
            
            # Update progress bar with stats from the last registered layer (usually deepest)
            if stats_holders and stats_holders[-1] is not None:
                curr_mean, curr_var = stats_holders[-1].get_stats()
                disp_mean = curr_mean.mean().item()
                disp_var = curr_var.mean().item()
                pbar.set_postfix_str(f"Mean: {disp_mean:.2e}, Var: {disp_var:.2e}")
    
    end_time = time.time()
    print(f"Extraction finished in {(end_time - start_time):.2f}s")

    # Save Results
    tensors_to_save = {}
    for h_idx, hook in enumerate(feature_hooks):
        if stats_holders[h_idx]:
            avg_mean, avg_var = stats_holders[h_idx].get_stats()
            layer_name = hook.name.replace(".", "_") # clean name
            tensors_to_save[f"{layer_name}.mean"] = avg_mean
            tensors_to_save[f"{layer_name}.var"] = avg_var
    
    save_path = os.path.join(output_dir, "feature_stats.safetensors")
    save_file(tensors_to_save, save_path)
    print(f"Saved stats to {save_path}")

    # Cleanup
    hook_manager.remove_hooks()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility.")
    parser.add_argument("--vlm", type=str, default="llava-llama3-8b", choices=VLM_CHOICES, help="VLM to use.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16.")
    parser.add_argument("--target_dir", type=str, default="./target_feats", help="Base directory to store target feature stats.")
    args = parser.parse_args()
    
    set_reproducibility(args.seed)
    run(args)


if __name__ == "__main__":
    main()
