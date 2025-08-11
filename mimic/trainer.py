import random
import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.utils as vutils
from PIL import Image

from helpers.adapted_loss import AdaptedLoss
from helpers.utils import denormalize, clip, normalize, get_base_feature_scale
from helpers.chat_processor import ChatProcessor
from helpers.feature_loss import get_feature_loss
from helpers.regularizers import get_patch_prior, get_extra_priors


class MIMICTrainer(object):
    def __init__(
        self,
        base_model,
        base_processor,
        guide_model,
        accelerator,
        base_hook_manager,
        guide_hook_manager,
        use_generate=False,
        loss_fn=None,
    ):
        """
        MIMIC Trainer to reconstruct the learned features from a VLM model using visual features from the VLM's vision transformer and a CNN guide model.

        The training process involves iteratively updating the random or zero input images based on the loss computed from the VLM model's outputs and the CNN guide model's features and VLM's vision transformer's features.

        Args:
            base_model: A VLM base model
            base_processor: A processor for the VLM base model
            guide_model: A CNN guide model
            accelerator: An accelerator object
            base_hook_manager: An instance of `FeatureStatsHookManager` for base model
            guide_hook_manager: An instance of `FeatureStatsHookManager` for guide model
            use_generate (optional): Whether to use the generate method of the VLM model. Default: False.
            loss_fn (optional): An instance of a function - (logits, labels, input_ids) -> (loss, final_labels). Default: `AdaptedLoss` object.
        """
        self.base_model = base_model
        self.base_processor = base_processor
        self.guide_model = guide_model
        self.accelerator = accelerator
        self.use_generate = use_generate
        self.base_hook_manager = base_hook_manager
        self.guide_hook_manager = guide_hook_manager
        self.base_feature_hooks = base_hook_manager.feature_hooks
        self.guide_feature_hooks = guide_hook_manager.feature_hooks
        self.chat_processor = ChatProcessor(
            base_processor,
            use_generate=self.use_generate,
        )
        self.patch_size = base_processor.patch_size
        self.tokenizer = base_processor.tokenizer
        self.image_processor = base_processor.image_processor
        self.loss_fn = loss_fn or AdaptedLoss(
            image_token_id=base_model.config.image_token_index,
            loss_kwargs={"ignore_index": base_model.config.ignore_index},
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
    ):
        self._print_memory_usage(tag="before training")
        self.chat_sequence = chat_sequence  # (user_message, assistant_message)
        self.target_class = target_class
        self.target_stats = target_stats
        self.parameters = parameters
        self.compute_metrics = compute_metrics
        self.compare_results = compare_results

        self.target_label = chat_sequence[0][1]  # assistant message is the target label
        self.best_results = {}
        self.best_iterations = {}
        self.best_images = {}

        if parameters:
            self.wandb_project = parameters["wandb_project"]
            self.seed = parameters["seed"]
            self.do_flip = parameters["do_flip"]
            self.use_blank_image = parameters["use_blank_image"]
            self.n_iterations = parameters["n_iterations"]
            self.n_jitter_iterations = parameters["n_jitter_iterations"]
            self.jitter = parameters["jitter"]
            self.batch_size_per_device = parameters["batch_size_per_device"]
            self.grad_accumulation_steps = parameters["grad_accumulation_steps"]
            self.image_resolution = parameters["image_resolution"]
            self.verifier_arch = parameters["verifier_arch"]
            self.fp16 = parameters["fp16"]
            self.save_every = parameters["save_every"]
            self.log_every = parameters["log_every"]
            self.eval_every = parameters["eval_every"]
            self.outputs_dir = parameters["outputs_dir"]
            self.results_dir = parameters["results_dir"]
            self.warmup_length = parameters["warmup_length"]
            self.base_feature_loss_type = parameters["base_feature_loss_type"]
            self.lr = parameters["lr"]
            self.min_lr = parameters["min_lr"]
            self.first_bn_multiplier = parameters["first_bn_multiplier"]
            self.main_loss_scale = parameters["main_loss_scale"]
            self.feature_guide_scale = parameters["feature_guide_scale"]
            self.feature_base_scale = parameters["feature_base_scale"]
            self.patch_prior_scale = parameters["patch_prior_scale"]
            self.tv_l1_scale = parameters["tv_l1_scale"]
            self.tv_l2_scale = parameters["tv_l2_scale"]
            self.l2_scale = parameters["l2_scale"]
        else:
            raise Exception("Provide a coefficients dictionary")

        if self.accelerator.is_main_process:
            # TODO: Use accelerator tracker
            self.run = wandb.init(
                project=self.wandb_project,
                config=self.parameters,
                mode="offline",
            )

        self.accelerator.wait_for_everyone()
        self._get_images()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.run.finish()

    def _print_memory_usage(self, tag=""):
        self.accelerator.print(
            f"[{tag}] | GPU: [Act/Max:{torch.cuda.memory_allocated() / 1e9:.2f}/{torch.cuda.max_memory_allocated() / 1e9:.2f}GB]",
        )

    def _get_images(self):
        self.accelerator.print("Generating images...")

        base_model = self.base_model
        guide_model = self.guide_model

        # Prepare noise inputs or blank image inputs
        if self.use_blank_image:
            initial_images = torch.zeros(
                (
                    self.batch_size_per_device,
                    3,
                    self.image_resolution,
                    self.image_resolution,
                ),
                device=self.accelerator.device,
            )
            input_images = normalize(initial_images).requires_grad_(True)
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

            input_images = normalize(initial_images).requires_grad_(True)

        self.accelerator.print("Input Images Shape: ", input_images.shape)
        self.accelerator.print("Input Images: ", input_images)

        # Tokenize the chat and images, but we are not using the pixel values
        input_ids, attention_mask, labels, _ = self.chat_processor.preprocess(
            self.chat_sequence,
            batch_size=self.batch_size_per_device,
        )

        input_ids = input_ids.to(device=self.accelerator.device)
        attention_mask = attention_mask.to(device=self.accelerator.device)
        labels = labels.to(device=self.accelerator.device)

        self.accelerator.print("Input IDs Shape: ", input_ids.shape)
        self.accelerator.print("Input IDs: ", input_ids)
        self.accelerator.print("Labels Shape: ", labels.shape)
        self.accelerator.print("Labels: ", labels)

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
            optimizer = optim.Adam(
                [input_images], lr=self.lr, betas=[0.5, 0.9], eps=1e-8
            )
            # optimizer = self.accelerator.prepare(optimizer)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=iterations_per_layer, eta_min=self.min_lr
            )

            if self.n_jitter_iterations > 0:
                lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            self.accelerator.print(f"Running for {iterations_per_layer} iterations...")

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                current_lr = optimizer.param_groups[0]["lr"]

                is_save_step = self.save_every and iteration % self.save_every == 0
                is_log_step = self.log_every and iteration % self.log_every == 0
                is_eval_step = self.eval_every and iteration % self.eval_every == 0
                is_final_step = iteration == self.n_iterations

                if self.n_jitter_iterations > 0:
                    inputs_jit = pooling_function(input_images)
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.roll(
                        inputs_jit, shifts=(off1, off2), dims=(2, 3)
                    )

                    flip = random.random() > 0.50
                    if self.do_flip and flip:
                        inputs_jit = torch.flip(inputs_jit, dims=(3,))
                else:
                    inputs_jit = input_images

                optimizer.zero_grad()
                base_model.zero_grad()

                # Even though, we are not updating the CNN weights, we need the gradients to flow through the # CNN because we are using the CNN features for loss computation against the generated image.
                guide_model.zero_grad()

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
                        logits = torch.stack(outputs.scores, dim=1)  # [B, L, V]
                    else:
                        outputs = base_model(
                            pixel_values=inputs_jit,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        logits = outputs.logits

                    # Adatped CE Loss
                    loss_criterion, final_labels = self.loss_fn(
                        logits, labels, input_ids
                    )

                    # Base Feature Loss(features from VLM)
                    # Since we don't have BNs in the VLM model,
                    # we need to extract the features from the VLM model
                    # We manually construct the feature stats using the computed stats
                    # from real images of the target class.
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
                    if iteration == 1:
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

                    _ = guide_model(inputs_jit)

                    feature_guide = [
                        [hook.data["r_feature"], hook.data["r_feature_bn"]]
                        for hook in self.guide_feature_hooks
                    ]

                    # Guide regularizer (features from CNN)
                    rescale_guide = [self.first_bn_multiplier] + [
                        1.0 + (self.first_bn_multiplier / (_s + 2))
                        for _s in range(len(feature_guide) - 1)
                    ]

                    if iteration == 1:
                        self.accelerator.print("rescale_guide: ", rescale_guide)

                    reg_feature_guide = get_feature_loss(
                        feature_guide, rescale=rescale_guide
                    )

                    # Get extra prior losses - L2, TV_L1, TV_L2
                    reg_l2, reg_tv_l1, reg_tv_l2 = get_extra_priors(inputs_jit)

                    reg_extra_prior = (
                        self.tv_l1_scale * reg_tv_l1
                        + self.tv_l2_scale * reg_tv_l2
                        + self.l2_scale * reg_l2
                    )

                    # Patch prior Loss
                    reg_patch_prior = get_patch_prior(
                        inputs_jit, patch_size=self.patch_size
                    )

                    loss = (
                        self.main_loss_scale * loss_criterion
                        + self.feature_guide_scale * reg_feature_guide
                        + self.feature_base_scale * loss_feature_base
                        + self.patch_prior_scale * reg_patch_prior
                        + reg_extra_prior
                    )

                self.accelerator.backward(loss)

                grad_norm = input_images.grad.norm(p=2)

                optimizer.step()
                scheduler.step()

                if do_clip:
                    with torch.no_grad():
                        input_images = clip(input_images)

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
                        "reg_feature_guide": reg_feature_guide.detach(),
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
                            self.run.log(logs, step=iteration)

                            print("\n-----------------------------------------------")
                            self._print_memory_usage(
                                tag=f"{iteration}/{self.n_iterations}"
                            )
                            print("\n------ Losses (Iter {}) -------".format(iteration))
                            self._print_logs(logs)

                            print("\n---- Predictions (Iter {}) ----".format(iteration))
                            predicted_ids = self._get_predicted_tokens(
                                gathered_logits, gathered_final_labels, topk=5
                            )
                            # Return the decoded response for each image in batch
                            predicted_sentences = self.tokenizer.batch_decode(
                                predicted_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
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

                            self.run.log(
                                results,
                                step=iteration,
                            )

                        if is_save_step or is_final_step:
                            self._save_image_grid(
                                gathered_inputs,
                                iteration,
                            )

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

    def _print_logs(self, logs):
        print("lr", logs["lr"])
        print("loss_criterion", logs["loss_criterion"])
        print("loss_feature_base", logs["loss_feature_base"])
        print("reg_patch_prior", logs["reg_patch_prior"])
        print("reg_tv_l1", logs["reg_tv_l1"])
        print("reg_tv_l2", logs["reg_tv_l2"])
        print("reg_l2", logs["reg_l2"])
        print("reg_feature_guide", logs["reg_feature_guide"])
        print("loss_total", logs["loss_total"])
        print("grad_norm", logs["grad_norm"])

    def _save_image_grid(self, images, iteration):
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
        images = denormalize(images)

        for id in range(images.shape[0]):
            img = images[id].detach().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_uint8)
            pil_image.save(
                f"{self.results_dir}/result_{self.seed}_{self.target_class}_{iteration:05d}_{id:03d}{tag}.png"
            )

    def _get_predicted_tokens(self, logits, labels, topk=5):
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
                top_tokens = self.tokenizer.batch_decode(
                    top_token_ids.tolist(), clean_up_tokenization_spaces=False
                )
                topk_str = ", ".join(
                    f"\n\tP({tok!r}): {prob:.4f}"
                    for tok, prob in zip(top_tokens, top_probs.tolist())
                )

                print(
                    f"[Batch {b}] [Pos {pos:3d}] "
                    f"P({gold_token!r}): {gold_prob:.4f} | Logit: {gold_logit:.4f} \nTop-{topk}:{topk_str}"
                )

                predicted_ids[b].append(int(top_token_ids[0].item()))

        return predicted_ids
