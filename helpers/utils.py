import json
import torch
import os
from torch import distributed, nn
import random
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader


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


# These are the CORRECT constants for LLaVA's CLIP vision encoder
LLAVA_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
LLAVA_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def clip(image_tensor):
    """
    Clip the image tensor in the CLIP-normalized space to ensure it corresponds
    to a valid [0, 1] image if it were to be denormalized. This operation is in-place.
    """
    mean = LLAVA_CLIP_MEAN.to(dtype=image_tensor.dtype, device=image_tensor.device)
    std = LLAVA_CLIP_STD.to(dtype=image_tensor.dtype, device=image_tensor.device)

    for c in range(3):
        m, s = mean[c], std[c]
        lower_bound = -m / s
        upper_bound = (1 - m) / s
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], lower_bound, upper_bound)
    return image_tensor


def normalize(image_tensor):
    """
    Normalize the image tensor from the [0, 1] pixel space to the CLIP-normalized space. This operation is in-place.
    """
    mean = LLAVA_CLIP_MEAN.to(dtype=image_tensor.dtype, device=image_tensor.device)
    std = LLAVA_CLIP_STD.to(dtype=image_tensor.dtype, device=image_tensor.device)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = (image_tensor[:, c] - m) / s

    return image_tensor


def denormalize(image_tensor):
    """
    Denormalize the image tensor from the CLIP-normalized space to the [0, 1] pixel space. This operation is in-place.
    """
    mean = LLAVA_CLIP_MEAN.to(dtype=image_tensor.dtype, device=image_tensor.device)
    std = LLAVA_CLIP_STD.to(dtype=image_tensor.dtype, device=image_tensor.device)

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
