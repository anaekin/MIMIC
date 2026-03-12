# --------------------------------------------------------
# Inversion of Visual Language Models paper
# Reformulated for Direct Optimization
# --------------------------------------------------------
import argparse
import gc
import os
import traceback
import torch
from tqdm.auto import tqdm
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

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

# Reproducibility
def set_reproducibility(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed, device_specific=False)
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

class RunningStats:
    """Computes running mean and variance."""
    def __init__(self, shape, device, dtype):
        self.n = 0
        self.sum_mean = torch.zeros(shape, device=device, dtype=dtype)
        self.sum_var = torch.zeros(shape, device=device, dtype=dtype)

    def update(self, batch_means, batch_vars):
        batch_size = batch_means.shape[0]
        self.n += batch_size
        self.sum_mean += batch_means.sum(dim=0)
        self.sum_var += batch_vars.sum(dim=0)

    def get_stats(self):
        if self.n == 0:
            return self.sum_mean, self.sum_mean 
        return self.sum_mean / self.n, self.sum_var / self.n

def extract_features_in_memory(trainer, images_dir, save_path, images_res):
    accelerator = trainer.accelerator
    device = accelerator.device
    vlm_processor = trainer.base_processor
    vlm_model = trainer.base_model
    hooks = trainer.base_feature_hooks
    
    accelerator.print(f"Extracting features from {images_dir} using loaded model...")
    
    # Dataloader
    dataloader = get_image_dataloader(images_dir, batch_size=16, image_size=images_res, return_pil=True)
    stats_holders = [None] * len(hooks)
    
    vlm_model.eval()
    
    # Only iterate on main process if possible, or use one GPU. 
    # Since we are in accelerator, we should probably just run on one device or gather.
    # But RunningStats is local.
    # For simplicity, let's assume running on main process or replicated.
    # If DDP, each process processes a subset? get_image_dataloader returns global loader.
    # We'll just run on main process if we want to save file, but that might leave others idle/hanging?
    # Actually, DDP is tricky here.
    # If we run on all processes, we split data.
    # `get_image_dataloader` returns standard DataLoader. Accelerated? No.
    # Let's assume we run this ONLY on Main Process and others wait?
    # But `vlm_model` is likely wrapped in DDP if prepare was called? 
    # `setup_trainer` did NOT call `accelerator.prepare(vlm_model)`. It just did `to(device)`.
    # So `vlm_model` is a local model instance.
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting", disable=not accelerator.is_main_process)
    
    with torch.no_grad():
        for i, batch_images in pbar:
            try:
                # Properly prepare inputs using chat template if available to ensure image tokens are present
                BATCH_SIZE = len(batch_images)
                
                # Construct conversation for each image (Llava needs <image> token implicitly via template or explicitly)
                # We use a dummy prompt that includes the image.
                conversations = []
                for _ in range(BATCH_SIZE):
                    conversations.append([
                        {"role": "user", "content": [{"type": "text", "text": "Describe this."}, {"type": "image"}]}
                    ])
                
                texts = [vlm_processor.apply_chat_template(c, add_generation_prompt=True) for c in conversations]
                # Wrap each image in its own list so processors that expect [[img1], [img2], ...]
                # (e.g. Gemma3n, Qwen3) correctly associate one image per text prompt.
                nested_images = [[img] for img in batch_images]
                inputs = vlm_processor(images=nested_images, text=texts, return_tensors="pt", padding=True)

            except Exception as e:
                # Fallback to simple processing if templating fails (e.g. models without chat template)
                # print(f"Template application failed: {e}. Using raw processor.")
                try:
                    inputs = vlm_processor(images=batch_images, text=["<image>"]*len(batch_images), return_tensors="pt", padding=True)
                except:
                     inputs = vlm_processor(images=batch_images, return_tensors="pt")
                
            inputs = inputs.to(device=device)
            
            try:
                # Forward pass
                # LlavaNext needs input_ids to calculate image features position
                if trainer.use_generate:
                     # If we wrapped generate, standard forward might still work as 'generate' is a method on top
                     # vlm_model is the model instance.
                     pass
                
                outputs = vlm_model(**inputs)
                
            except Exception as e:
                # If full model forward fails, try running just the vision tower if accessible.
                # Common issue: "Image features and image tokens do not match" -> fixed above by ensuring proper template?
                # If still failing, fallback to vision tower.
                
                # accelerator.print(f"  > Standard forward failed: {e}. Trying vision tower only...")
                try:
                    vision_tower = None
                    if hasattr(vlm_model, "vision_tower"): vision_tower = vlm_model.vision_tower
                    elif hasattr(vlm_model, "model") and hasattr(vlm_model.model, "vision_tower"): vision_tower = vlm_model.model.vision_tower
                    elif hasattr(vlm_model, "visual"): vision_tower = vlm_model.visual
                    
                    if vision_tower:
                         # Vision tower inputs
                         vt_args = []
                         vt_kwargs = {}
                         
                         # Check signature if possible or standard keys
                         if "pixel_values" in inputs: vt_kwargs["pixel_values"] = inputs["pixel_values"]
                         if "images" in inputs: vt_kwargs["images"] = inputs["images"]
                         if "grid_thw" in inputs: vt_kwargs["grid_thw"] = inputs["grid_thw"]
                         elif "image_grid_thw" in inputs: vt_kwargs["grid_thw"] = inputs["image_grid_thw"]
                         
                         # Call vision tower
                         # Some vision towers (LlavaNext) return a tuple, causing "too many values to unpack" if invoked via some wrappers.
                         # Just calling with kwargs is usually safest.
                         vision_tower(**vt_kwargs)
                         # accelerator.print("  > Vision tower forward successful.")
                    else:
                         raise e
                except Exception as e2:
                    accelerator.print(f"  > Features extraction failed for batch {i}: {e} (Fallback: {e2})")
                    continue

            # Collect Stats
            for h_idx, hook in enumerate(hooks):
                if hook.data and hook.data.get("r_feature") is not None:
                    mean_b, var_b = hook.data["r_feature"]
                    if stats_holders[h_idx] is None:
                        D = mean_b.shape[-1]
                        stats_holders[h_idx] = RunningStats([D], device, torch.float32)
                    stats_holders[h_idx].update(mean_b.to(torch.float32), var_b.to(torch.float32))
                    hook.data = None # Clear
    
    # Save
    tensors_to_save = {}
    target_features_list = []
    
    for h_idx, hook in enumerate(hooks):
        if stats_holders[h_idx]:
            avg_mean, avg_var = stats_holders[h_idx].get_stats()
            layer_name = hook.name.replace(".", "_")
            tensors_to_save[f"{layer_name}.mean"] = avg_mean
            tensors_to_save[f"{layer_name}.var"] = avg_var
            target_features_list.append([avg_mean, avg_var])
        else:
             # Should not happen if hooks triggered
             accelerator.print(f"Warning: No stats for hook {hook.name}")
             target_features_list.append([torch.zeros(1).to(device), torch.zeros(1).to(device)])
             
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_safetensors(tensors_to_save, save_path)
        accelerator.print(f"Saved extracted features to {save_path}")
        
    return target_features_list


def run_single_run(trainer, parameters, accelerator):
    try:
        if accelerator.is_main_process:
            os.makedirs(parameters["outputs_dir"], exist_ok=True)

        prompt = next((c for c in PROMPT_LISTS if c["class_index"] == parameters["target_class"]), None)
        if not prompt:
            raise Exception("Target class not found")
        
        # Resolve Target Features Path
        vlm_name = parameters["vlm"]
        class_idx = parameters["target_class"]
        # Use class_folder (WNID) if available, else index
        class_str = prompt.get("class_folder", str(class_idx))
        
        # New structure from run_feature_extraction: target_dir/vlm/dataset_name/feature_stats.safetensors
        # Here dataset_name would be the class folder.
        # We need to guess the folder name if we don't have it.
        # If images_root provided, checking existence is easy.
        
        # 1. Look for safetensors in expected path (assuming class folder name is str(class_idx))
        target_dir = parameters["target_features_dir"]
        safetensors_path = os.path.join(target_dir, vlm_name, class_str, "feature_stats.safetensors")
        
        # 2. Legacy check
        legacy_path = get_target_features_filepath(target_dir, class_idx)
        
        if os.path.exists(safetensors_path):
            accelerator.print(f"Loading features from {safetensors_path}")
            try:
                stats_dict = load_safetensors(safetensors_path)
                if not stats_dict:
                     accelerator.print(f"Warning: {safetensors_path} is empty.")
                     raise ValueError("Empty safetensors file")
                     
                # Convert to list matching hooks
                target_features = []
                for hook in trainer.base_feature_hooks:
                    # Robust key matching
                    clean_name = hook.name.replace(".", "_")
                    
                    found_key = None
                    if f"{clean_name}.mean" in stats_dict:
                        found_key = clean_name
                    else:
                        # Try fuzzy matching (suffix match)
                        # hook name might be 'model.vision_tower.vision_model...'
                        # file key might be 'vision_model...'
                        # regex usually targets the end of the path
                        for k in stats_dict.keys():
                            if k.endswith(".mean"):
                                # check if the base name matches the suffix of clean_name or vice versa
                                k_base = k.replace(".mean", "")
                                # normalize underscore vs dot
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
                        raise KeyError(f"Could not find match for hook {clean_name} in file keys: {list(stats_dict.keys())[:5]}...")

            except (ValueError, KeyError, Exception) as e:
                accelerator.print(f"Failed to load existing features: {e}")
                accelerator.print("Will attempt to re-extract.")
                target_features = None # Trigger extraction
                
        elif os.path.exists(legacy_path):
            accelerator.print(f"Loading features from {legacy_path}")
            target_features = torch.load(legacy_path, map_location=accelerator.device)
        else:
            # Need extraction
            target_features = None

        if target_features is None:
            # Check if we can extract
            accelerator.print(f"Target features not found or invalid.")
            
            if not parameters["images_root"]:
                raise FileNotFoundError("Features missing and --images_root not provided. Cannot extract.")
            
            # Identify class folder
            images_root = parameters["images_root"]
            class_folder = os.path.join(images_root, class_str)
            
            # Try finding folder by name if not index/wnid
            if not os.path.exists(class_folder) and "class_name" in prompt:
                 class_folder_name = os.path.join(images_root, prompt["class_name"])
                 if os.path.exists(class_folder_name):
                     class_folder = class_folder_name

            # Fallback to index if WNID failed
            if not os.path.exists(class_folder) and class_str != str(class_idx):
                 class_folder_idx = os.path.join(images_root, str(class_idx))
                 if os.path.exists(class_folder_idx):
                     class_folder = class_folder_idx
            
            if not os.path.exists(class_folder):
                 raise FileNotFoundError(f"Could not find images for class {class_idx} (Folder: {class_str}) in {images_root}")
            
            accelerator.print(f"Extracting features from {class_folder} using model in memory...")

            # Run In-Memory Extraction
            target_features = extract_features_in_memory(trainer, class_folder, safetensors_path, VLM_HYPERPARAMS[vlm_name]["image_resolution"])
            accelerator.wait_for_everyone()

        
        accelerator.print("\n------------ MIMIC Optimization Started ------------")

        trainer.train(
            prompt["chat_sequence"],
            prompt["class_index"],
            target_features,
            parameters,
            enable_logging=accelerator.is_main_process and parameters.get("wandb", 1),
            return_images=False,
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
    parser.add_argument("--wandb_project", type=str, default="mimic")
    parser.add_argument("--seed", default=0, type=int)

    # Image Generation Params
    parser.add_argument("--batch_size_per_device", default=1, type=int)
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)

    parser.add_argument("--n_iterations", default=500, type=int, help="Iterations per MIMIC run")
    parser.add_argument("--n_jitter_iterations", type=int, default=500)
    parser.add_argument("--jitter", default=16, type=int)
    
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--target_features_dir", type=str, default="./target_features")
    parser.add_argument("--do_flip", action="store_true")
    parser.add_argument("--use_blank_image", action="store_true")
    parser.add_argument("--target_class", type=int, required=True, choices=AVALIABLE_CLASSES, default=88)
    parser.add_argument("--images_root", type=str, default=None, help="Root directory for images (e.g. ImageNet val/train folder). Used for extraction if stats missing.")

    # Base Coefficients (Initial Policy Mean)
    parser.add_argument("--base_feature_loss_type", type=str, default="l2")
    parser.add_argument("--lr", type=float, default=0.05, help="Image optimization LR")
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_length", type=int, default=50)
    
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=0)

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
    
    vlm_hyperparams = VLM_HYPERPARAMS.get(args.vlm, {})
    for k, v in vlm_hyperparams.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)
            print(f"Used default {k}={v}")
        elif not hasattr(args, k):
            setattr(args, k, v)
    
    results_dir = args.results_dir+f"/{args.vlm}"
    if not os.path.exists(results_dir):
        print(f"Creating results directory at {results_dir}")
        os.makedirs(results_dir)
    outputs_dir = args.outputs_dir+f"/{args.vlm}"
    if not os.path.exists(outputs_dir):
        print(f"Creating outputs directory at {outputs_dir}")
        os.makedirs(outputs_dir)
    parameters = vars(args)
    
    config_keys = ["vlm", "fp16", "seed", "grad_accumulation_steps", "use_generate"]
    config = {k: parameters[k] for k in config_keys}
    
    trainer, accelerator = setup_trainer(config)
    run_single_run(trainer, parameters, accelerator)

if __name__ == "__main__":
    main()
