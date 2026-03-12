import json
import torch
import os
import math
import sys
from einops import rearrange
from transformers import (
    logging,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Gemma3nForConditionalGeneration,
    DeepseekVLHybridForConditionalGeneration,
)
from transformers.image_processing_utils import select_best_resolution
from torch import distributed, nn
from torchvision import transforms
import random
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader


VLM_CHOICES = [
    "llava-llama3-8b",
    "llava-mistral-7b",
    "llava-vicuna-7b",
    "llava-vicuna-13b",
    "gemma-3n-E4B",
    "qwen3-vl-8b",
    "deepseek-vl-7b",
]

VLM_LAYER_PATTERNS = {
    "llava-llama3-8b": r"vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.out_proj$",
    "llava-mistral-7b": r"vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.out_proj$",
    "llava-vicuna-7b": r"vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.out_proj$",
    "llava-vicuna-13b": r"vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.out_proj$",
    "qwen3-vl-8b": r"visual\.blocks\.\d+\.attn\.proj$",
    "gemma-3n-E4B": r"vision_tower\.timm_model\.blocks\.\d+\.\d+\.pw_proj$",
    "deepseek-vl-7b": r"model\.vision_model\.vision_model\.encoder\.layers\.\d+\.mlp\.fc2$",
}


# These are the CORRECT constants for LLaVA's CLIP vision encoder
LLAVA_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
LLAVA_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
# Qwen-VL Normalization (from preprocessor_config.json: image_mean/std = 0.5)
QWEN_MEAN = torch.tensor([0.5, 0.5, 0.5])
QWEN_STD = torch.tensor([0.5, 0.5, 0.5])
# Gemma-3n Normalization (do_normalize=false in preprocessor, model expects [0,1])
GEMMA_MEAN = torch.tensor([0.0, 0.0, 0.0])
GEMMA_STD = torch.tensor([1.0, 1.0, 1.0])


VLM_HYPERPARAMS = {"llava-llama3-8b": {"main_loss_scale": 0.005, 
                                        "feature_base_scale": 0.04, 
                                        "patch_prior_scale": 0.0005, 
                                        "patch_internal_scale": 0.0005, 
                                        "tv_l1_scale": 0.07, 
                                        "tv_l2_scale": 0.04,
                                        "l2_scale":0.0003,
                                        "image_resolution": 336},
                    "llava-mistral-7b": {"main_loss_scale": 0.005, 
                                        "feature_base_scale": 0.04, 
                                        "patch_prior_scale": 0.0005, 
                                        "patch_internal_scale": 0.0005, 
                                        "tv_l1_scale": 0.07, 
                                        "tv_l2_scale": 0.04,
                                        "l2_scale":0.0003,
                                        "image_resolution": 336},
                    "llava-vicuna-7b": {"main_loss_scale": 0.005, 
                                        "feature_base_scale": 0.04, 
                                        "patch_prior_scale": 0.0005, 
                                        "patch_internal_scale": 0.0005, 
                                        "tv_l1_scale": 0.07, 
                                        "tv_l2_scale": 0.04,
                                        "l2_scale":0.0003,
                                        "image_resolution": 336},
                    "llava-vicuna-13b": {"main_loss_scale": 0.005, 
                                        "feature_base_scale": 0.04, 
                                        "patch_prior_scale": 0.0005, 
                                        "patch_internal_scale": 0.0005, 
                                        "tv_l1_scale": 0.07, 
                                        "tv_l2_scale": 0.04,
                                        "l2_scale":0.0003,
                                        "image_resolution": 224}, 
                    "qwen3-vl-8b": {"main_loss_scale": 0.05, 
                                        "feature_base_scale": 0.04, 
                                        "patch_prior_scale": 0.05, 
                                        "patch_internal_scale": 0.005, 
                                        "tv_l1_scale": 0.25, 
                                        "tv_l2_scale": 0.05,
                                        "l2_scale":0.01,
                                        "image_resolution": 448},
                    "gemma-3n-E4B": {"main_loss_scale": 0.01, 
                                        "feature_base_scale": 0.2, 
                                        "patch_prior_scale": 0.04, 
                                        "patch_internal_scale": 0.015, 
                                        "tv_l1_scale": 1.2, 
                                        "tv_l2_scale": 0.45,
                                        "l2_scale": 0.02,
                                        "edge_aware_tv_scale": 20.0,
                                        "blur_every": 100,
                                        "blur_sigma": 1.0,
                                        "blur_kernel_size": 4,
                                        "image_resolution": 448},
    }

def load_vlm(vlm_name):
    if vlm_name == "llava-llama3-8b": 
        ckpt =  "./models/llava-v1.6-llama3-8b-hf"
    elif vlm_name == "llava-mistral-7b": 
        ckpt = "./models/llava-v1.6-mistral-7b-hf"
    elif vlm_name == "gemma-3n-E4B":
        ckpt =  "./models/gemma-3n-E4B-it"
    elif vlm_name == "qwen3-vl-8b":
        ckpt = "./models/qwen3-vl-8b-instruct"
    elif vlm_name == "deepseek-vl-7b":
        ckpt = "./models/deepseek-vl-7b-chat"
    elif vlm_name == "llava-vicuna-7b":
        ckpt = "./models/llava-v1.6-vicuna-7b-hf"
    elif vlm_name == "llava-vicuna-13b":
        ckpt = "./models/llava-v1.6-vicuna-13b-hf"
    else:
        print(f"VLM choice: {vlm_name} not defined. Add choice to {VLM_CHOICES} and adjust `load_vlm` function.")
        sys.exit(1)
    print(f"Loading VLM from checkpoint: {ckpt}")
    vlm_processor = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
    tokenizer = vlm_processor.tokenizer
    
    if 'llava' in vlm_name:
        vlm_model = LlavaNextForConditionalGeneration.from_pretrained(ckpt, local_files_only=True, torch_dtype=torch.bfloat16)
    elif 'qwen' in vlm_name:
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(ckpt, local_files_only=True, torch_dtype=torch.bfloat16)
    elif 'gemma' in vlm_name:
        logging.set_verbosity_error()
        vlm_model = Gemma3nForConditionalGeneration.from_pretrained(ckpt, local_files_only=True, torch_dtype=torch.bfloat16)
    elif 'deepseek' in vlm_name:
        vlm_model = DeepseekVLHybridForConditionalGeneration.from_pretrained(ckpt, local_files_only=True, torch_dtype=torch.bfloat16)
    
    if tokenizer.chat_template is None:
        if 'vicuna' in vlm_name:
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] + ' '}}{% elif content['type'] == 'image' %}{{ '<image>\n' }}{% endif %}{% endfor %}{% endif %}{% if message['role'] == 'assistant' %}{{ '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        
    return vlm_processor, tokenizer, vlm_model

def prepare_vlm_inputs(processor, image, text_prompt, device):
    conversation = [{
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },{
        "role": "user",
        "content": [{"type": "text", "text": text_prompt},
                    {"type": "image"},],
    },]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device,dtype=torch.bfloat16)
    return inputs

def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    if not isinstance(image_size, (list, tuple)):
         image_size = list(image_size)

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches

def forward_llava(model, inputs, pixel_values, labels=None, **kwargs):
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    device = pixel_values.device
    batch_size = pixel_values.shape[0]
    image_res = pixel_values.shape[2]
    model_kwargs = {}
    
    # Image Sizes
    image_sizes = torch.tensor([[image_res, image_res]] * batch_size, device=device)
    model_kwargs["image_sizes"] = image_sizes
    
    # Patch Expansion logic
    config = model.config
    if hasattr(config, "vision_config"):
        vision_config = getattr(config, "vision_config", config)
        patch_size = vision_config.image_size
        grid_pinpoints = config.image_grid_pinpoints
        
        num_patches = image_size_to_num_patches((image_res, image_res), grid_pinpoints, patch_size)
        
        # Expand pixel_values [B, C, H, W] -> [B, N, C, H, W]
        pixel_values = pixel_values.unsqueeze(1).expand(-1, num_patches, -1, -1, -1)
    
    # print(f"Using {num_patches} patches for image resolution {image_res}x{image_res}")
    # print(f"pixel_values shape: {pixel_values.shape}")
    # print(f"Input IDs shape: {input_ids.shape}, Attention Mask shape: {attention_mask.shape}")
    # print(f"Labels shape: {labels.shape}" if labels is not None else "No labels provided.")
    
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        return_dict=True,
        **model_kwargs
    )

def forward_qwen3(model, inputs, pixel_values, labels=None, **kwargs):
    model_kwargs = {}
    
    # Qwen3VL requires image_grid_thw
    if hasattr(inputs, "image_grid_thw"):
        grid_thw = inputs.image_grid_thw
        model_kwargs["image_grid_thw"] = grid_thw
    elif "image_grid_thw" in kwargs:
        grid_thw = kwargs["image_grid_thw"]
        model_kwargs["image_grid_thw"] = grid_thw
    else:
        grid_thw = None
        print("Warning: image_grid_thw not found in inputs or kwargs. Qwen3VL may not process images correctly without it.")

    # Reconstruct correct pixel_values structure from input_images (pixel_values arg)
    # The 'pixel_values' arg passed here is the (B, 3, H, W) tensor we are optimizing.
    # We need to transform it to match the shape expected by Qwen (flattened patches).
    if hasattr(inputs, "pixel_values") and grid_thw is not None:
        target_pv = inputs.pixel_values
        N, D = target_pv.shape
        t_grid, h_grid, w_grid = grid_thw[0] # tensor([1, 20, 20])
        
        patch_size = 14
        channels = 3
        temporal_factor = 2 # Qwen standard
        patch_area = D / (channels * temporal_factor)
        patch_size = int(math.sqrt(patch_area))
        
        # Calculate target resolution
        target_h = int(h_grid * patch_size)
        target_w = int(w_grid * patch_size)
        
        # 1. Resize input image to match grid
        # input (B, C, H, W) -> (B, C, target_h, target_w)
        if pixel_values.shape[-2:] != (target_h, target_w):
             pixel_values = torch.nn.functional.interpolate(
                pixel_values, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
             print(f"Resized input images to {target_h}x{target_w} to match Qwen grid requirements.")
            
        # 2. Replicate for temporal dimension if needed
        # Qwen2-VL usually treats 1 image as 2 frames (or pairs of patches)
        # (B=1, C=3, H, W) -> (B=1, T=2, C=3, H, W)
        pixel_values = pixel_values.unsqueeze(1).repeat(1, temporal_factor, 1, 1, 1)
        
        # 3. Patchify using rearrange to preserve spatial structure
        # (B, T, C, H, W) -> (B * H_grid * W_grid, T * C * P * P)
        pixel_values = rearrange(
            pixel_values, 
            'b t c (h p1) (w p2) -> (b h w) (t c p1 p2)', 
            p1=patch_size, 
            p2=patch_size,
            h=int(h_grid),
            w=int(w_grid)
        )

    return model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        return_dict=True,
        **model_kwargs
    )

def forward_gemma(model, inputs, pixel_values, labels=None, **kwargs):
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        return_dict=True
    )

def forward_vlm(model, inputs, pixel_values, vlm_name, labels=None, **kwargs):
    """
    Dispatcher for VLM forward passes.
    """
    if "llava" in vlm_name:
        return forward_llava(model, inputs, pixel_values, labels, **kwargs)
    elif "qwen" in vlm_name:
        return forward_qwen3(model, inputs, pixel_values, labels, **kwargs)
    elif "gemma" in vlm_name:
        return forward_gemma(model, inputs, pixel_values, labels, **kwargs)
    else:
        # Default fallback
        return model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True
        )

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def clip(image_tensor, mean=LLAVA_CLIP_MEAN, std=LLAVA_CLIP_STD):
    """
    Clip the image tensor in the normalized space to ensure it corresponds
    to a valid [0, 1] image if it were to be denormalized. This operation is in-place.
    """
    mean = mean.to(dtype=image_tensor.dtype, device=image_tensor.device)
    std = std.to(dtype=image_tensor.dtype, device=image_tensor.device)

    for c in range(3):
        m, s = mean[c], std[c]
        lower_bound = -m / s
        upper_bound = (1 - m) / s
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], lower_bound, upper_bound)
    return image_tensor


def normalize(image_tensor, mean=LLAVA_CLIP_MEAN, std=LLAVA_CLIP_STD):
    """
    Normalize the image tensor from the [0, 1] pixel space to the normalized space. This operation is in-place.
    """
    mean = mean.to(dtype=image_tensor.dtype, device=image_tensor.device)
    std = std.to(dtype=image_tensor.dtype, device=image_tensor.device)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = (image_tensor[:, c] - m) / s

    return image_tensor


def denormalize(image_tensor, mean=LLAVA_CLIP_MEAN, std=LLAVA_CLIP_STD):
    """
    Denormalize the image tensor from the normalized space to the [0, 1] pixel space. This operation is in-place.
    """
    mean = mean.to(dtype=image_tensor.dtype, device=image_tensor.device)
    std = std.to(dtype=image_tensor.dtype, device=image_tensor.device)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = image_tensor[:, c] * s + m

    return torch.clamp(image_tensor, 0, 1)


def get_target_features_filepath(target_features_dir, target_class):
    """Get the file path for the target features based on the target label."""
    return os.path.join(target_features_dir, f"target_features_{target_class}.pt")


def load_json(json_file_path):
    """
    Load the JSON file.
    """
    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"File {json_file_path} does not exist.")
    else:
        # Load the JSON file
        with open(json_file_path, "r") as file:
            data = json.load(file)

    return data


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.paths = glob(os.path.join(root, "**/*.*"), recursive=True)
        self.paths = [
            p for p in self.paths if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img  # no label

class SimpleImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Walk through directory
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.image_paths.append(os.path.join(root, file))
        
        if not self.image_paths:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def get_image_dataloader(images_dir, batch_size=32, image_size=336, num_workers=4, return_pil=False, vlm_name="llava-llama3-8b"):
    if return_pil:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])
        # We need a custom dataset that returns PIL images
        class PILDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.image_paths = []
                for root, _, files in os.walk(root_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            self.image_paths.append(os.path.join(root, file))
            def __len__(self): return len(self.image_paths)
            def __getitem__(self, idx):
                try:
                    img = Image.open(self.image_paths[idx]).convert("RGB")
                    if self.transform: img = self.transform(img)
                    return img
                except: return self.__getitem__((idx + 1) % len(self))
        
        dataset = PILDataset(images_dir, transform=transform)
        # Collate to return list of PIL images
        def collate_fn(batch): return batch 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader

    mean, std = get_image_norm(vlm_name)
    normalization = transforms.Normalize(
        mean=mean.tolist(),
        std=std.tolist()
    )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalization
    ])

    dataset = SimpleImageFolder(images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader

def get_image_norm(vlm_name):
    if "qwen" in vlm_name.lower():
        return QWEN_MEAN, QWEN_STD
    elif "gemma" in vlm_name.lower():
        return GEMMA_MEAN, GEMMA_STD
    else:
        return LLAVA_CLIP_MEAN, LLAVA_CLIP_STD

def get_base_feature_scale(hooks):
    rescale = []
    for h in hooks:
        name = h.name
        if "patch_embedding" in name:
            rescale.append(2.0)  # prioritize low-level structure
        elif "mlp.fc2" in name:
            layer_id = int(name.split(".")[-3])
            if layer_id <= 3:
                rescale.append(1.5)  # early transformer layers
            elif layer_id <= 12:
                rescale.append(1.0)  # mid
            elif layer_id <= 20:
                rescale.append(0.5)  # deep
            else:
                rescale.append(0.25)  # very deep
        else:
            rescale.append(1.0)  # fallback
    return rescale
