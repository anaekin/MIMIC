import os
import torch
from accelerate.utils import DistributedType


class FeatureExtractor:
    """
    Compute aggregated target feature statistics (mean and variance)
    for each hooked VLM vision tower layers using all batches from the real images.

    Args:
        vlm_model: VLM base model to extract features
        vlm_processor: VLM processor to preprocess images
        feature_hooks: List of hooks attached to target layers
        accelerator: Accelerator object
    """

    def __init__(
        self,
        vlm_model,
        vlm_processor,
        feature_hooks,
        device="cuda",
        data_type=torch.bfloat16,
    ):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.feature_hooks = feature_hooks
        self.device = device
        self.data_type = data_type

    def compute(self, dataloader):
        """
        Compute aggregated target feature statistics (mean and variance)
        for each hooked VLM vision tower layers using all batches from the real images.
        This version accumulates the statistics in lists and then averages them.

        Args:
            dataloader: DataLoader for real images
        """
        print("Computing target feature statistics...")
        self.vlm_model.eval()

        num_layers = len(self.feature_hooks)
        all_means = [[] for _ in range(num_layers)]
        all_vars = [[] for _ in range(num_layers)]

        with torch.no_grad():
            for images, _ in dataloader:
                inputs = self.vlm_processor(
                    text=["<image>"] * images.shape[0],
                    images=images,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(device=self.device)
                pixel_values = inputs["pixel_values"].to(device=self.device)
                attention_mask = inputs["attention_mask"].to(device=self.device)

                with torch.autocast(device_type=str(self.device), dtype=self.data_type):
                    _ = self.vlm_model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                    )

                for i, layer_hook in enumerate(self.feature_hooks):
                    if layer_hook.data["r_feature"] is None:
                        continue

                    mean_b, var_b = layer_hook.data["r_feature"]  #  [2, B, D]

                    # Split and add each sample separately
                    all_means[i].append(mean_b.detach().cpu())
                    all_vars[i].append(var_b.detach().cpu())

                    layer_hook.data = None

        all_layer_target_stats = []
        for i in range(num_layers):
            # # For per-image stats
            layer_means = torch.cat(all_means[i], dim=0)  # [N, D]
            layer_vars = torch.cat(all_vars[i], dim=0)  # [N, D]

            avg_mean = layer_means.mean(0)  # [D]
            avg_var = layer_vars.mean(0)  # [D]

            all_layer_target_stats.append([avg_mean, avg_var])  # [2, D]
