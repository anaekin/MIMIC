import re
import torch.nn as nn


class FeatureStatsHook:
    """
    Hook to extract features from a VLM model or CNN guide model during forward pass.
    The data from the hook is stored in the `data` attribute.

    Args:
        name: Name of the module
        module: Module to hook
        hook_fn: Hook function to be called, data will be stored in `data` property of the hook class.
    """

    def __init__(self, name, module, hook_fn):
        self.hook = module.register_forward_hook(self._hook_fn)
        self.data = None
        self.name = name
        self.hook_fn = hook_fn

    def _hook_fn(self, module, input, output):
        self.data = self.hook_fn(module, input, output)

    def reset(self):
        self.data = None

    def close(self):
        self.hook.remove()


class FeatureStatsHookManager:
    """
    This class manages the forward hooks for MIMIC. It is used to
    register and remove hooks.

    Args:
        model: Hooks will be registered for this model
        model_type: Type of model, either "vlm" or "cnn"
        custom_hook_fn (optional): Custom hook function

          Example:
              def custom_hook_fn(module, input, output):
                  embeddings = output
                  mean = embeddings.mean(dim=[0, 1])
                  var = embeddings.var(dim=[0, 1], unbiased=False)

                  return {"r_feature": (mean, var)}
    """

    def __init__(self, model, model_type, custom_hook_fn=None):
        self.model = model
        self.custom_hook_fn = custom_hook_fn
        self.model_type = model_type

        self.feature_hooks = []

    def _add_vlm_hooks(self):
        print("Adding hooks to VLM base model...")

        def vlm_hook_fn(module, input, output):
            x = output

            if x.dim() == 4:  # [B, D, H, W] → patch embedding
                B, D, H, W = x.shape
                x = x.view(B, D, -1)  # [B, D, tokens]
                x = x.permute(0, 2, 1)  # [B, tokens, D]

            elif x.dim() == 3:  # [B, N, D] → transformer layer
                pass  # already in correct shape

            else:
                raise ValueError(
                    f"Unsupported tensor shape {x.shape} in unified_vlm_hook_fn"
                )

            # Compute per-image feature stats
            mean = x.mean(dim=1)  # mean over tokens → [B, D]
            var = x.var(dim=1, unbiased=False)

            return {"r_feature": [mean, var]}  # Per layer shape is [2, B, D]

        hook_fn = self.custom_hook_fn or vlm_hook_fn
        hook_layers = [
            (
                r"vision_tower\.vision_model\.embeddings\.patch_embedding",
                "Patch Embedding Layer",
            ),
            (
                r"vision_tower\.vision_model\.encoder\.layers\.\d+\.mlp\.fc2",
                "MLP Layer 2",
            ),
            # Final LayerNorm is not being used by LLava VLM.
            # (r"vision_tower\.vision_model\.post_layernorm", "Final LayerNorm"),
        ]

        for layer_regex, description in hook_layers:
            for name, module in self.model.named_modules():
                if re.search(layer_regex, name) is not None:
                    hook = FeatureStatsHook(name, module, hook_fn=hook_fn)
                    self.feature_hooks.append(hook)
                    print(f"---> Hook added to {description}: {name}")

        print(f"Added a total of {len(self.feature_hooks)} VLM hooks")

    def _add_guide_hooks(self):
        print("Adding hooks to CNN guide model...")

        def guide_hook_fn(module, input, output):
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = (
                input[0]
                .permute(1, 0, 2, 3)
                .contiguous()
                .view([nch, -1])
                .var(1, unbiased=False)
            )
            mean_bn = module.running_mean.data
            var_bn = module.running_var.data

            return {
                "r_feature": [mean, var],
                "r_feature_bn": [mean_bn, var_bn],
            }

        hook_fn = self.custom_hook_fn or guide_hook_fn
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = FeatureStatsHook(name, module, hook_fn=hook_fn)
                self.feature_hooks.append(hook)

        print(f"Added a total of {len(self.feature_hooks)} CNN guide model hooks")

    def register_hooks(self):
        """
        Register forward hooks for VLM base model and CNN guide model. If using `accelerate` library, then call before `accelerate.prepare()` method.

        Returns a list of FeatureStatsHook objects
        """
        if self.model_type == "vlm":
            self._add_vlm_hooks()

        if self.model_type == "cnn":
            self._add_guide_hooks()

        return self.feature_hooks

    def remove_hooks(self):
        """
        Remove all hooks from the VLM base model and CNN guide model
        """
        for hook in self.feature_hooks:
            hook.close()

        self.feature_hooks = []

    def reset_hooks(self):
        """
        Clear data and resets all hooks
        """
        for hook in self.feature_hooks:
            hook.reset()

        return self.feature_hooks
