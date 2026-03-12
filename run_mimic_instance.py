# --------------------------------------------------------
# Inversion of Visual Language Models paper
# Reformulated for Single Instance Optimization
# --------------------------------------------------------
import argparse
import gc
import os
import traceback
import torch
from tqdm.auto import tqdm
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

import wandb
from datetime import datetime

from accelerate import Accelerator
from accelerate.utils import set_seed

from types import MethodType
from transformers.generation.utils import GenerationMixin

from helpers.utils import get_target_features_filepath, load_json, load_vlm, VLM_CHOICES, VLM_LAYER_PATTERNS, VLM_HYPERPARAMS, get_image_dataloader
from helpers.hooks import FeatureStatsHookManager
from mimic.trainer import MIMICTrainer

# Constants
PROMPT_FILEPATH = "./data/prompts.json"
PROMPT_LISTS = load_json(PROMPT_FILEPATH)
AVALIABLE_CLASSES = [c["class_index"] for c in PROMPT_LISTS]

def extract_features_in_memory(trainer, images_dir, save_path, images_res):
    accelerator = trainer.accelerator
    device = accelerator.device
    vlm_processor = trainer.base_processor
    vlm_model = trainer.base_model
    hooks = trainer.base_feature_hooks
    
    accelerator.print(f"Extracting features from {images_dir} using loaded model...")
    
    # Dataloader
    dataloader = get_image_dataloader(images_dir, batch_size=16, image_size=images_res, return_pil=True)
    
    vlm_model.eval()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting", disable=not accelerator.is_main_process)
    
    class SimpleRunningStats:
        def __init__(self, device):
            self.n = 0
            self.mean = None
            self.M2 = None # Sum of squares of differences from the current mean

        def update(self, batch_tensor):
            # batch_tensor: [B, ...]
            batch_tensor = batch_tensor.to(self.device)
            B = batch_tensor.shape[0]
            
            if self.n == 0:
                self.mean = torch.zeros_like(batch_tensor[0])
                self.M2 = torch.zeros_like(batch_tensor[0])
                self.device = batch_tensor.device   
            pass 

  
    
    # Initialize accumulators
    nums = [0] * len(hooks)
    sums = [None] * len(hooks)
    sq_sums = [None] * len(hooks)

    all_feats = [[] for _ in range(len(hooks))]

    with torch.no_grad():
        for i, (images, _) in pbar:
            inputs = vlm_processor(text=[""]*len(images), images=images, return_tensors="pt", padding=True).to(device)
            
            # We need to run the model and capture hooks
            # Just verify keys.
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=vlm_model.dtype)
            
            # Forward pass
            if hasattr(vlm_model, "generate") and trainer.use_generate:
                 vlm_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values, output_hidden_states=True)
            else:
                 vlm_model(**inputs)

            # Collect features from hooks
            for h_idx, hook in enumerate(hooks):
                feat = hook.feat_out
                all_feats[h_idx].append(feat.detach().cpu())
            
            
    # Compute and Save
    tensors_to_save = {}
    target_features = []
    
    for h_idx, hook in enumerate(hooks):
        # Concatenate all batches: [N_total, ...]
        full_feat = torch.cat(all_feats[h_idx], dim=0).to(torch.float32) # Calculate in float32
        
        # Calculate Mean and Var across dim 0
        mean = full_feat.mean(dim=0)
        var = full_feat.var(dim=0)
        
        # Move back to original dtype if needed, or keep float32 for precision in stats file
        tensors_to_save[f"{hook.name}.mean"] = mean
        tensors_to_save[f"{hook.name}.var"] = var
        
        target_features.append([mean.to(device), var.to(device)])
        
    if accelerator.is_main_process and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_safetensors(tensors_to_save, save_path)
        accelerator.print(f"Saved features to {save_path}")

    return target_features

# Reproducibility
def set_reproducibility(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set_seed(seed, device_specific=False) # accelerate set_seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_precision_settings(fp16):
    if fp16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        else:
            return torch.float16, "fp16"
    else:
        return torch.float32, "no"

def setup_trainer(config):
    set_reproducibility(config["seed"])
    use_generate = config["use_generate"]

    data_type, mixed_precision = get_precision_settings(config["fp16"])
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config["grad_accumulation_steps"],
    )

    accelerator.print("\n----------- Configuration -----------")
    device_count = torch.cuda.device_count()
    
    accelerator.print("Device count: ", device_count)
    accelerator.print("Mixed precision: ", accelerator.mixed_precision)

    # Model loading and setup
    accelerator.print("\n--------------- Setup ---------------")
    
    vlm_name = config["vlm"]
    accelerator.print(f"Loading {vlm_name}...")

    vlm_processor, _, vlm_model = load_vlm(vlm_name)
    vlm_model = vlm_model.to(device=accelerator.device, dtype=data_type)
    vlm_model.eval()

    if use_generate:
        vlm_model.generate = MethodType(GenerationMixin.generate, vlm_model)

    if hasattr(vlm_model.config, "vision_config") and hasattr(vlm_model.config.vision_config, "patch_size"):
        vlm_processor.patch_size = vlm_model.config.vision_config.patch_size
    else:
        # Defaults
        if "patch_size" in config:
            vlm_processor.patch_size = config["patch_size"]
        else:
            vlm_processor.patch_size = 14

    layer_regex = VLM_LAYER_PATTERNS.get(vlm_name)
    base_hook_manager = FeatureStatsHookManager(model=vlm_model, model_type="vlm", layer_regex=layer_regex)
    base_hook_manager.register_hooks()

    accelerator.wait_for_everyone()

    trainer = MIMICTrainer(
        vlm_model,
        vlm_processor,
        accelerator,
        base_hook_manager,
        use_generate=use_generate,
    )

    return trainer, accelerator

def run_single_run(trainer, parameters, accelerator):
    try:
        if accelerator.is_main_process:
            os.makedirs(parameters["outputs_dir"], exist_ok=True)
        
        target_class_idx = parameters["target_class"]
        prompt_info = next((c for c in PROMPT_LISTS if c["class_index"] == target_class_idx), None)
        
        # Determine chat sequence
        chat_sequence = None
        if prompt_info:
            import copy
            chat_sequence = copy.deepcopy(prompt_info.get("chat_sequence"))
        
        # Override with custom args if provided
        custom_prompt = parameters.get("custom_prompt")
        target_text = parameters.get("target_text")
        
        if custom_prompt or target_text:
            if not chat_sequence:
                 # Start with a default generic structure if no base exists
                 chat_sequence = [["", ""]]
            
        
            if custom_prompt:
                # Ensure the format matches what ChatProcessor expects (raw text, image token usually handled by ChatProcessor)
                chat_sequence[0][0] = custom_prompt
            
            if target_text:
                chat_sequence[0][1] = target_text
                
            accelerator.print(f"Using Updated Chat Sequence: {chat_sequence}")

        if not chat_sequence:
             # Fallback if still empty (shouldn't happen if prompts.json is valid or args provided)
             raise ValueError("No chat sequence found. Provide --custom_prompt/--target_text or valid --target_class with prompt info.")
             
        # 2. Resolve Target Features
        vlm_name = parameters["vlm"]
        
        # Use class_folder name if available
        class_str = str(target_class_idx)
        if prompt_info and "class_folder" in prompt_info:
            class_str = prompt_info["class_folder"]
            
        target_dir = parameters["target_features_dir"]
        safetensors_path = os.path.join(target_dir, vlm_name, class_str, "feature_stats.safetensors")
        legacy_path = get_target_features_filepath(target_dir, target_class_idx)
        
        target_features = None
        
        # Try loading
        if os.path.exists(safetensors_path):
            accelerator.print(f"Loading features from {safetensors_path}")
            try:
                stats_dict = load_safetensors(safetensors_path)
                target_features = []
                for hook in trainer.base_feature_hooks:
                    clean_name = hook.name.replace(".", "_")
                    found_key = None
                    if f"{clean_name}.mean" in stats_dict:
                        found_key = clean_name
                    else:
                        # Fuzzy match
                        for k in stats_dict.keys():
                            if k.endswith(".mean"):
                                k_base = k.replace(".mean", "")
                                k_norm = k_base.replace(".", "_")
                                name_norm = clean_name.replace(".", "_")
                                if name_norm.endswith(k_norm) or k_norm.endswith(name_norm):
                                    found_key = k_base
                                    break
                    if found_key:
                        mean = stats_dict[f"{found_key}.mean"].to(accelerator.device)
                        var = stats_dict[f"{found_key}.var"].to(accelerator.device)
                        target_features.append([mean, var])
                    else:
                        # Missing key
                        pass 
                
                if not target_features:
                    target_features = None # Loading failed effectively
                    
            except Exception as e:
                accelerator.print(f"Error loading safetensors: {e}")
                target_features = None
                
        elif os.path.exists(legacy_path):
             accelerator.print(f"Loading legacy features from {legacy_path}")
             target_features = torch.load(legacy_path, map_location=accelerator.device)
        
        # Extraction if not found
        if target_features is None:
            accelerator.print("Target features not found, attempting extraction.")
            if not parameters["images_root"]:
                raise ValueError("Target features missing and --images_root not provided.")
            
            images_root = parameters["images_root"]
            class_folder_path = os.path.join(images_root, class_str)
            
            # Try alt paths if specific folder missing
            if not os.path.exists(class_folder_path):
                 if prompt_info and "class_name" in prompt_info:
                     alt = os.path.join(images_root, prompt_info["class_name"])
                     if os.path.exists(alt): class_folder_path = alt
            
            if not os.path.exists(class_folder_path):
                # Try just index
                alt = os.path.join(images_root, str(target_class_idx))
                if os.path.exists(alt): class_folder_path = alt
                
            if not os.path.exists(class_folder_path):
                raise FileNotFoundError(f"Could not locate image folder for class {target_class_idx} in {images_root}")
                
            resolution = parameters.get("image_resolution", 336) 
            target_features = extract_features_in_memory(trainer, class_folder_path, safetensors_path, resolution)
            
        
        # 3. Run Training
        accelerator.print("\n------------ MIMIC Optimization Started ------------")
        
        labels_info = prompt_info["class_name"] if prompt_info else str(target_class_idx)
        if accelerator.is_main_process and parameters.get("wandb", 1):
            wandb_run_name = f"{parameters['vlm']}_{labels_info}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            parameters["wandb_project"] = parameters.get("wandb_project", "mimic_single")
            
        final_images = trainer.train(
            chat_sequence,
            target_class_idx,
            target_features,
            parameters,
            enable_logging=accelerator.is_main_process and parameters.get("wandb", 1),
            return_images=False # or True if we want to do something with them
        )
        
        accelerator.print("\n------------ MIMIC Optimization Finished ------------")

    except Exception:
        accelerator.print(traceback.format_exc())
    finally:
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument("--vlm", type=str, default="llava-llama3-8b", choices=VLM_CHOICES)
    parser.add_argument("--wandb_project", type=str, default="mimic_single")
    parser.add_argument("--seed", default=0, type=int)
    
    # Custom Trigger
    parser.add_argument("--custom_prompt", type=str, default=None, help='Override the user prompt content (first turn).')
    parser.add_argument("--target_text", type=str, default=None, help="Override the assistant response content (first turn).")

    # Image Optimization Params
    parser.add_argument("--batch_size_per_device", default=1, type=int)
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    
    parser.add_argument("--n_iterations", default=1500, type=int, help="Iterations per MIMIC run")
    parser.add_argument("--n_jitter_iterations", type=int, default=1500)
    parser.add_argument("--jitter", default=16, type=int)
    
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--target_features_dir", type=str, default="./target_features")
    parser.add_argument("--do_flip", action="store_true")
    parser.add_argument("--use_blank_image", action="store_true")
    parser.add_argument("--target_class", type=int, required=True, default=88)
    parser.add_argument("--images_root", type=str, default=None, help="Root directory for images extraction")

    parser.add_argument("--base_feature_loss_type", type=str, default="l2")
    parser.add_argument("--lr", type=float, default=0.05, help="Image optimization LR")
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_length", type=int, default=50)
    
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=0)
    
    # Hyperparams (Scales) - allow override
    parser.add_argument("--main_loss_scale", type=float, default=None)
    parser.add_argument("--feature_base_scale", type=float, default=None)
    parser.add_argument("--patch_prior_scale", type=float, default=None)
    parser.add_argument("--patch_internal_scale", type=float, default=None)
    parser.add_argument("--tv_l1_scale", type=float, default=None)
    parser.add_argument("--tv_l2_scale", type=float, default=None)
    parser.add_argument("--l2_scale", type=float, default=None)
    
    parser.add_argument("--use_generate", action="store_true")
    parser.add_argument("--use_fft", action="store_true")
    parser.add_argument("--wandb", type=int, default=1, help="Use wandb logging")

    args = parser.parse_args()
    
    # Load default hyperparams
    vlm_defaults = VLM_HYPERPARAMS.get(args.vlm, {})
    for k, v in vlm_defaults.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)
            print(f"Used default {k}={v}")
        elif not hasattr(args, k):
            setattr(args, k, v)
    
    parameters = vars(args)

    
    # Directories
    results_dir = args.results_dir+f"/{args.vlm}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    outputs_dir = args.outputs_dir+f"/{args.vlm}"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
    
    config = {k: parameters[k] for k in ["vlm", "fp16", "seed", "grad_accumulation_steps", "use_generate"] if k in parameters}
    
    trainer, accelerator = setup_trainer(config)
    
    run_single_run(trainer, parameters, accelerator)

if __name__ == "__main__":
    main()
