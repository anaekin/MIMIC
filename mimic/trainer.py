import os
import random
import numpy as np
import wandb
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.utils as vutils
from PIL import Image

from helpers.adapted_loss import AdaptedLoss
from helpers.utils import denormalize, clip, normalize, get_base_feature_scale, forward_vlm, get_image_norm
from helpers.chat_processor import ChatProcessor
from helpers.feature_loss import get_feature_loss
from helpers.regularizers import get_patch_internal_variance, get_patch_prior, get_extra_priors, get_edge_aware_tv, gaussian_blur
from mimic.fftimage import FFTImage

class MIMICTrainer(object):
    def __init__(
        self,
        base_model,
        base_processor,
        accelerator,
        base_hook_manager,
        use_generate=False,
        loss_fn=None,
    ):
        """
        MIMIC Trainer to reconstruct the learned features from a VLM model using visual features from the VLM's vision transformer.

        The training process involves iteratively updating the random or zero input images based on the loss computed from the VLM model's outputs and VLM's vision transformer's features.

        Args:
            base_model: A VLM base model
            base_processor: A processor for the VLM base model
            accelerator: An accelerator object
            base_hook_manager: An instance of `FeatureStatsHookManager` for base model
            use_generate (optional): Whether to use the generate method of the VLM model. Default: False.
            loss_fn (optional): An instance of a function - (logits, labels, input_ids) -> (loss, final_labels). Default: `AdaptedLoss` object.
        """
        self.base_model = base_model
        self.base_processor = base_processor
        self.accelerator = accelerator
        self.use_generate = use_generate
        self.base_hook_manager = base_hook_manager
        
        # Determine normalization constants based on model name
        try:
            model_name = base_model.config.name_or_path
        except:
            model_name = "llava" # Default
            
        self.mean, self.std = get_image_norm(model_name)
        self.mean = self.mean.to(self.accelerator.device)
        self.std = self.std.to(self.accelerator.device)
        self.accelerator.print(f"Using Normalization: Mean={self.mean.cpu().tolist()}, Std={self.std.cpu().tolist()}")
        self.base_feature_hooks = base_hook_manager.feature_hooks

        image_token = "<image>"
        if hasattr(base_model.config, "model_type") and ("qwen" in base_model.config.model_type):
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"

        self.chat_processor = ChatProcessor(
            base_processor,
            image_token=image_token,
            use_generate=self.use_generate,
        )
        self.patch_size = base_processor.patch_size
        self.tokenizer = base_processor.tokenizer
        self.image_processor = base_processor.image_processor
        
        # for llava models
        if hasattr(base_model.config, "image_token_index"):
            image_token_id = base_model.config.image_token_index
        # for qwen models
        elif hasattr(base_model.config, "image_token_id"):
            image_token_id = base_model.config.image_token_id
        
        self.loss_fn = loss_fn or AdaptedLoss(
            image_token_id=image_token_id,
            loss_kwargs={"ignore_index": getattr(base_model.config, "ignore_index", -100)},
            use_generate=self.use_generate,
        )

    def train(
        self,
        chat_sequence,
        target_class,
        target_stats,
        parameters,
        compute_metrics=None,
        compare_results=None,
        enable_logging=True,
        return_images=False,
        path_ctx=None,
    ):
        # self._print_memory_usage(tag="before training")
        self.chat_sequence = chat_sequence  # (user_message, assistant_message)
        self.target_class = target_class
        self.target_stats = target_stats
        self.parameters = parameters
        self.compute_metrics = compute_metrics
        self.compare_results = compare_results
        self.enable_logging = enable_logging

        self.target_label = chat_sequence[0][1]  # assistant message is the target label
        self.best_results = {}
        self.best_iterations = {}
        self.best_images = {}
        self.path_ctx = path_ctx

        if parameters:
            self.wandb_project = parameters["wandb_project"]
            self.use_wandb = parameters.get("wandb", 1) == 1
            self.seed = parameters["seed"]
            self.do_flip = parameters["do_flip"]
            self.use_blank_image = parameters["use_blank_image"]
            self.n_iterations = parameters["n_iterations"]
            self.n_jitter_iterations = parameters["n_jitter_iterations"]
            self.jitter = parameters["jitter"]
            self.batch_size_per_device = parameters["batch_size_per_device"]
            self.grad_accumulation_steps = parameters["grad_accumulation_steps"]
            self.image_resolution = parameters["image_resolution"]
            self.fp16 = parameters["fp16"]
            self.save_every = parameters["save_every"]
            self.log_every = parameters["log_every"]
            self.eval_every = parameters["eval_every"]
            self.outputs_dir = parameters["outputs_dir"]+f"/{parameters['vlm']}"
            self.results_dir = parameters["results_dir"]+f"/{parameters['vlm']}"
            self.warmup_length = parameters["warmup_length"]
            self.base_feature_loss_type = parameters["base_feature_loss_type"]
            self.lr = parameters["lr"]
            self.min_lr = parameters["min_lr"]
            self.main_loss_scale = parameters["main_loss_scale"]
            self.feature_base_scale = parameters["feature_base_scale"]
            self.patch_prior_scale = parameters["patch_prior_scale"]
            self.patch_internal_scale = parameters["patch_internal_scale"]
            self.tv_l1_scale = parameters["tv_l1_scale"]
            self.tv_l2_scale = parameters["tv_l2_scale"]
            self.l2_scale = parameters["l2_scale"]
            self.edge_aware_tv_scale = parameters.get("edge_aware_tv_scale", 0.0)
            self.blur_every = parameters.get("blur_every", 0)
            self.blur_sigma = parameters.get("blur_sigma", 1.0)
            self.blur_kernel_size = parameters.get("blur_kernel_size", 3)
            self.use_fft = parameters.get("use_fft", False)
        else:
            raise Exception("Provide a coefficients dictionary")

        if self.accelerator.is_main_process and self.enable_logging and self.use_wandb:
            self.run = wandb.init(
                project=self.wandb_project,
                config=self.parameters,
                mode="offline",
            )

        self.accelerator.wait_for_everyone()
        final_images = self._get_images(return_images=return_images)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process and self.enable_logging and self.use_wandb:
            self.run.finish()
        
        return final_images

    def _print_memory_usage(self, tag=""):
        self.accelerator.print(
            f"[{tag}] | GPU: [Act/Max:{torch.cuda.memory_allocated() / 1e9:.2f}/{torch.cuda.max_memory_allocated() / 1e9:.2f}GB]",
        )

    def _get_images(self, return_images=False, verbose=False):
        if verbose:
            self.accelerator.print("Generating images...")

        base_model = self.base_model
        
        fft_image_module = None

        # Prepare noise inputs or blank image inputs
        if self.use_fft:
             fft_image_module = FFTImage(
                 batch_size=self.batch_size_per_device,
                 h=self.image_resolution,
                 w=self.image_resolution,
                 decay_power=1.0
             ).to(self.accelerator.device)
             input_images = normalize(fft_image_module().clone(), mean=self.mean, std=self.std)
        elif self.use_blank_image:
            initial_images = torch.zeros(
                (
                    self.batch_size_per_device,
                    3,
                    self.image_resolution,
                    self.image_resolution,
                ),
                device=self.accelerator.device,
            )
            input_images = normalize(initial_images, mean=self.mean, std=self.std).requires_grad_(True)
        else:
            initial_images = torch.randn(
                (
                    self.batch_size_per_device,
                    3,
                    self.image_resolution,
                    self.image_resolution,
                ),
                device=self.accelerator.device,
            )

            input_images = normalize(initial_images, mean=self.mean, std=self.std).requires_grad_(True)

        if verbose:
            self.accelerator.print("Input Images Shape: ", input_images.shape)

        # Prepare dummy images for proper tokenization expansion
        dummy_c_images = []
        for _ in range(self.batch_size_per_device):
            dummy_c_images.append(Image.new("RGB", (self.image_resolution, self.image_resolution)))
             
        # Tokenize the chat and images, but we are not using the pixel values
        input_ids, attention_mask, labels, _, image_inputs = self.chat_processor.preprocess(
            self.chat_sequence,
            images=dummy_c_images,
            batch_size=self.batch_size_per_device,
        )

        input_ids = input_ids.to(device=self.accelerator.device)
        attention_mask = attention_mask.to(device=self.accelerator.device)
        labels = labels.to(device=self.accelerator.device)

        if verbose:
            self.accelerator.print("Input IDs Shape: ", input_ids.shape)
            self.accelerator.print("Labels Shape: ", labels.shape)
            self.accelerator.print("Labels: ", labels.shape)

        # Capture final token dict
        self.final_tok_dict = None
        
        # Training loop
        pooling_function = torch.nn.Identity()
        iteration = 0
        for lower_res, iterations_per_layer in zip(
            [2, 1],
            [
                self.n_jitter_iterations,
                self.n_iterations - self.n_jitter_iterations,
            ],
        ):
            do_clip = True
            params = list(fft_image_module.parameters()) if self.use_fft else [input_images]
            optimizer = optim.AdamW(params, lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
            # optimizer = self.accelerator.prepare(optimizer)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=iterations_per_layer, eta_min=self.min_lr
            )

            if self.n_jitter_iterations > 0:
                lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if verbose:
                self.accelerator.print(f"Running for {iterations_per_layer} iterations...")
            
            pbar_desc = f"Path {self.path_ctx}" if self.path_ctx else "Path Iterations"
            pbar = tqdm(range(iterations_per_layer), disable=not self.accelerator.is_main_process, leave=True, desc=pbar_desc)
            for iteration_loc in pbar:
                iteration += 1
                current_lr = optimizer.param_groups[0]["lr"]
                
                # Update GPU Memory in pbar
                mem_use = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
                pbar.set_postfix(gpu=mem_use)

                is_save_step = self.save_every and iteration % self.save_every == 0
                is_log_step = self.log_every and iteration % self.log_every == 0
                is_eval_step = self.eval_every and iteration % self.eval_every == 0
                is_final_step = iteration == self.n_iterations

                if self.use_fft:
                    input_images = normalize(fft_image_module().clone(), mean=self.mean, std=self.std)

                if iteration_loc < self.n_jitter_iterations:
                    inputs_jit = pooling_function(input_images)
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.nn.functional.pad(inputs_jit, (lim_1, lim_1, lim_0, lim_0), mode='reflect')
                    start_h = lim_0 - off1
                    start_w = lim_1 - off2
                    
                    h_orig, w_orig = input_images.shape[2], input_images.shape[3]
                    inputs_jit = inputs_jit[:, :, start_h : start_h + h_orig, start_w : start_w + w_orig]
                    
                    # inputs_jit = torch.roll(
                    #     inputs_jit, shifts=(off1, off2), dims=(2, 3)
                    # )

                    flip = random.random() > 0.50
                    if self.do_flip and flip:
                        inputs_jit = torch.flip(inputs_jit, dims=(3,))
                else:
                    inputs_jit = input_images


                optimizer.zero_grad()
                base_model.zero_grad()

                with self.accelerator.autocast():
                    if self.use_generate:
                         outputs = base_model.generate(
                            pixel_values=inputs_jit,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            max_new_tokens=labels.shape[1],
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                         logits = torch.stack(outputs.scores, dim=1)
                    else:
                        class VLMInputs:
                            def __init__(self, ids, mask, image_inputs=None):
                                self.input_ids = ids
                                self.attention_mask = mask
                                if image_inputs:
                                    for k, v in image_inputs.items():
                                        if hasattr(v, "to"):
                                            v = v.to(ids.device)
                                        setattr(self, k, v)
                        
                        vlm_inputs = VLMInputs(input_ids, attention_mask, image_inputs)

                        outputs = forward_vlm(
                            model=base_model,
                            inputs=vlm_inputs,
                            pixel_values=inputs_jit,
                            vlm_name=self.parameters.get("vlm", ""),
                            labels=labels
                        )
                        logits = outputs.logits

                    # Adapted CE Loss
                    loss_criterion, final_labels = self.loss_fn(
                        logits, labels, input_ids
                    )

                    feature_base_batch = []
                    for b in range(self.batch_size_per_device):
                        feature_base = []
                        for hook, target_stats in zip(
                            self.base_feature_hooks, self.target_stats
                        ):
                            mean_b, var_b = hook.data[
                                "r_feature"
                            ]  # both tensors of shape [B, D]
                            img_stats = [
                                mean_b[b],
                                var_b[b],
                            ]  # shape [2, D]
                            feature_base.append(
                                [img_stats, target_stats]
                            )  # [L, 2, 2, D]
                        feature_base_batch.append(feature_base)  # [B, L, 2, 2, D]

                    # rescale_vlm = get_vlm_r_feature_rescale(self.base_feature_hooks)
                    rescale_vlm = None
                    if iteration == 1 and verbose:
                        self.accelerator.print("rescale_vlm: ", rescale_vlm)

                    loss_feature_base = torch.stack(
                        [
                            get_feature_loss(
                                feature_per_batch,
                                loss_type=self.base_feature_loss_type,
                                rescale=rescale_vlm,
                            )
                            for feature_per_batch in feature_base_batch
                        ]
                    ).mean()


                    reg_l2, reg_tv_l1, reg_tv_l2 = get_extra_priors(inputs_jit)

                    # Edge-aware TV: smooths background harder, preserves subject edges
                    if self.edge_aware_tv_scale > 0:
                        reg_edge_tv = get_edge_aware_tv(inputs_jit)
                    else:
                        reg_edge_tv = torch.tensor(0.0, device=inputs_jit.device)

                    reg_extra_prior = (
                        self.tv_l1_scale * reg_tv_l1
                        + self.tv_l2_scale * reg_tv_l2
                        + self.l2_scale * reg_l2
                        + self.edge_aware_tv_scale * reg_edge_tv
                    )

                    # Patch prior Loss
                    reg_patch_prior = get_patch_prior(
                        inputs_jit, patch_size=self.patch_size
                    )
                    
                    loss_patch_internal = get_patch_internal_variance(inputs_jit, patch_size=self.patch_size)
                    
                    loss = (
                        self.main_loss_scale * loss_criterion
                        + self.feature_base_scale * loss_feature_base
                        + self.patch_prior_scale * reg_patch_prior
                        + self.patch_internal_scale * loss_patch_internal
                        + reg_extra_prior
                    )
                self.accelerator.backward(loss)

                if self.use_fft:
                     grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in fft_image_module.parameters() if p.grad is not None]))
                else:
                    grad_norm = input_images.grad.norm(p=2)

                optimizer.step()
                scheduler.step()
                pbar.set_description(f"Iter {iteration}/{self.n_iterations} | Loss: {loss.item():.2e} | Grad Norm: {grad_norm.item():.2e}")
                pbar.update(0)

                if self.accelerator.is_main_process:
                     log_dict = {
                        "loss_criterion": loss_criterion.item(),
                        "loss_feature_base": loss_feature_base.item(),
                        "reg_tv_l1": reg_tv_l1.item(),
                        "reg_tv_l2": reg_tv_l2.item(),
                        "reg_l2": reg_l2.item(),
                        "reg_patch_prior": reg_patch_prior.item(),
                        "loss_total": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "lr": current_lr
                    }
                     if self.enable_logging and hasattr(self, 'run'):
                         self.run.log(log_dict, step=iteration)
                     elif self.use_wandb and wandb.run is not None:
                         wandb.log(log_dict)
                
                if do_clip and not self.use_fft:
                    with torch.no_grad():
                        input_images = clip(input_images, mean=self.mean, std=self.std)
                
                # Periodic Gaussian blur to physically remove high-freq noise
                if self.blur_every > 0 and iteration % self.blur_every == 0 and not self.use_fft:
                    with torch.no_grad():
                        input_images.data = gaussian_blur(
                            input_images.data,
                            kernel_size=self.blur_kernel_size,
                            sigma=self.blur_sigma,
                        )
                        
                if not self.enable_logging and not is_final_step:
                    continue

                if is_save_step or is_eval_step or is_final_step:
                    gathered_inputs = self.accelerator.gather_for_metrics(
                        input_images.detach()
                    )

                if is_log_step or is_final_step:
                    gathered_logits = self.accelerator.gather_for_metrics(
                        logits.detach()
                    )
                    gathered_final_labels = self.accelerator.gather_for_metrics(
                        final_labels.detach()
                    )

                    loss_tracker = {
                        "loss_criterion": loss_criterion.detach(),
                        "loss_feature_base": loss_feature_base.detach(),
                        "reg_tv_l1": reg_tv_l1.detach(),
                        "reg_tv_l2": reg_tv_l2.detach(),
                        "reg_l2": reg_l2.detach(),
                        "reg_patch_prior": reg_patch_prior.detach(),
                        "loss_total": loss.detach(),
                        "grad_norm": grad_norm.detach(),
                    }

                    gathered_losses = {
                        k: self.accelerator.reduce(v, "mean")
                        for k, v in loss_tracker.items()
                    }
                    

                        

                if self.accelerator.is_main_process:
                    predicted_sentences = None
                    with torch.no_grad():
                        if is_log_step or is_final_step:
                            logs = {
                                k: v.mean().item() for k, v in gathered_losses.items()
                            }
                            logs["lr"] = current_lr
                            
                            if self.enable_logging:
                                self.run.log(logs, step=iteration)

                            if (verbose == True):
                                print("\n-----------------------------------------------")
                                self._print_memory_usage(
                                    tag=f"{iteration}/{self.n_iterations}"
                                )
                                print("\n------ Losses (Iter {}) -------".format(iteration))
                                self._print_logs(logs)

                                print("\n---- Predictions (Iter {}) ----".format(iteration))
                            
                            predicted_ids, tok_dict = self._get_predicted_tokens(
                                gathered_logits, gathered_final_labels, topk=5
                            )
                            
                            tok_info = []
                            for pos, data in tok_dict.items():
                                g_tok, g_prob, _ = data['golden']
                                # t_tok, t_prob = data['top1']
                                t_tok, t_prob = data['topk'][0]
                                tok_info.append(f"P{pos}:{g_tok}({g_prob:.2f})<->{t_tok}({t_prob:.2f})")
                            
                            pbar.set_postfix_str(" | ".join(tok_info))

                            # Return the decoded response for each image in batch
                            predicted_sentences = self.tokenizer.batch_decode(
                                predicted_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                            if verbose:
                                print(predicted_sentences)

                        if (
                            is_eval_step or is_final_step
                        ) and self.compute_metrics is not None:
                            print("\n----- Accuracy (Iter {}) ------".format(iteration))
                            results = self.compute_metrics(
                                gathered_inputs, self.target_class, self.target_label
                            )
                            print(
                                f"Best Results @ {self.best_iterations}: ",
                                self.best_results,
                            )
                            print(f"Results @ {iteration}: ", results)

                            is_better = self.compare_results(self.best_results, results)

                            for k, v in is_better.items():
                                if v:
                                    self.best_results[k] = results[k]
                                    self.best_iterations[k] = iteration
                                    self.best_images[k] = gathered_inputs.clone()

                            if self.enable_logging:
                                self.run.log(
                                    results,
                                    step=iteration,
                                )

                        if is_save_step or is_final_step:
                            self._save_image_grid(
                                gathered_inputs,
                                iteration)
                        if "tok_dict" in locals():
                            self.final_tok_dict = tok_dict                            

                        if is_final_step:
                            # Save final images
                            final_images = gathered_inputs.clone()
                            self._save_images(final_images, iteration, tag="final")

                            # Save best images if there are any
                            for k in self.best_images.keys():
                                self._save_images(
                                    self.best_images[k], self.best_iterations[k], tag=k
                                )


            optimizer.state.clear()
        
        if "tok_info" in locals():
            print("\n" + "\n".join(tok_info))
        if return_images: 
            return final_images, self.final_tok_dict
        

    def _print_logs(self, logs):
        print("lr", logs["lr"])
        print("loss_criterion", logs["loss_criterion"])
        print("loss_feature_base", logs["loss_feature_base"])
        print("reg_patch_prior", logs["reg_patch_prior"])
        print("reg_tv_l1", logs["reg_tv_l1"])
        print("reg_tv_l2", logs["reg_tv_l2"])
        print("reg_l2", logs["reg_l2"])
        print("loss_total", logs["loss_total"])
    def _save_image_grid(self, images, iteration):
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        vutils.save_image(
            images,
            f"{self.outputs_dir}/output_{self.seed}_{self.target_class}_{iteration:05d}.png",
            normalize=True,
            scale_each=True,
            nrow=10,
        )

    def _save_images(self, images, iteration, tag=None):
        tag = f"_{tag}" if tag else ""
        images = images.float()
        denormalize(images, mean=self.mean, std=self.std)

        for id in range(images.shape[0]):
            img = images[id].detach().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_uint8)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            pil_image.save(
                f"{self.results_dir}/result_{self.seed}_{self.target_class}_{iteration:05d}_{id:03d}{tag}.png"
            )

    def _get_predicted_tokens(self, logits, labels, topk=5, verbose=False):
        """
        Print P(gold) and top-k predictions for every supervised (labels != –100) position.
        """
        B, L, V = logits.shape
        predicted_ids = [[] for _ in range(B)]
        topk = min(topk, V)

        for b in range(B):
            supervised_positions = (
                (labels[b] != -100).nonzero(as_tuple=True)[0].tolist()
            )
            tok_dict = {}
            for pos in supervised_positions:
                logit_idx = pos if self.use_generate else pos - 1
                if logit_idx < 0:
                    continue

                raw_logits = logits[b, logit_idx]
                raw_probs = torch.softmax(raw_logits, dim=-1)

                gold_token_id = labels[b, pos].item()
                gold_prob = raw_probs[gold_token_id].item()
                gold_logit = raw_logits[gold_token_id].item()
                gold_token = self.tokenizer.decode(
                    [gold_token_id], clean_up_tokenization_spaces=False
                )

                top_probs, top_token_ids = torch.topk(raw_probs, topk)
                # Ensure each token ID is decoded individually
                top_tokens = [
                    self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                    for tid in top_token_ids.tolist()
                ]
                topk_str = ", ".join(
                    f"\n\tP({tok!r}): {prob:.4f}"
                    for tok, prob in zip(top_tokens, top_probs.tolist())
                )
                tok_dict[pos] = {
                    'golden': (gold_token, gold_prob, gold_logit), 
                    'topk': list(zip(top_tokens, top_probs.tolist()))
                }
                if verbose:
                    print(
                        f"[Batch {b}] [Pos {pos:3d}] "
                        f"P({gold_token!r}): {gold_prob:.4f} | Logit: {gold_logit:.4f} \nTop-{topk}:{topk_str}"
                    )
                predicted_ids[b].append(int(top_token_ids[0].item()))

        return predicted_ids, tok_dict
