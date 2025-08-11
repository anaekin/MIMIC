# --------------------------------------------------------
# Inversion of Visual Language Models paper
# Animesh Jain
# --------------------------------------------------------
import argparse
import gc
import random
import os
import traceback
import numpy as np
import time
import json

import torch
import torchvision

from transformers import AutoProcessor, AutoModelForPreTraining

from accelerate import Accelerator
from accelerate.utils import set_seed

from types import MethodType
from transformers.generation.utils import GenerationMixin


from custom_datasets.imagenet import ImageNetDataloader, ImageNetDataset
from helpers.utils import get_target_features_filepath, load_json
from helpers.hooks import FeatureStatsHookManager
from metrics.accuracy import compute_accuracy
from metrics.clip_score import compute_clipscore
from metrics.lpips_score import compute_lpips
from mimic.trainer import MIMICTrainer

# Constants
PROMPT_FILEPATH = "./data/prompts.json"
PROMPT_LISTS = load_json(PROMPT_FILEPATH)
AVALIABLE_CLASSES = [c["class_index"] for c in PROMPT_LISTS]


# Reproducibility
def set_reproducibility(seed=0):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed, device_specific=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(
    images,
    target_class,
    target_label,
    dataloader,
):
    preds = compute_accuracy(
        images,
        target_class,
        topk=(1, 5),
    )
    clipscore = compute_clipscore(images, target_label)
    combined_score = (preds["top1"] + clipscore["clipscore"]) / 2

    # lpips = compute_lpips(images, dataloader)
    return {
        **preds,
        **clipscore,
        # **lpips,
        "combined_score": combined_score,
    }


def compare_results(best_results, results):
    """
    Compare the current results with the best results and return True if the current results are better.
    """
    is_better = {
        "top1": False,
        "clipscore": False,
        "lpips": False,
        "combined_score": False,
    }
    best_top1 = best_results.get("top1", 0.0)
    best_clipscore = best_results.get("clipscore", 0.0)
    best_combined_score = best_results.get("combined_score", 0.0)
    # best_lpips = best_results.get("lpips", float("inf")) # Lower is better

    top1 = results.get("top1")
    clipscore = results.get("clipscore")
    combined_score = results.get("combined_score")
    # lpips = results.get("lpips") # Lower is better

    if top1 >= best_top1:
        is_better["top1"] = True
    if clipscore >= best_clipscore - 0.05:
        is_better["clipscore"] = True
    if combined_score >= best_combined_score - 0.05:
        is_better["combined_score"] = True
    # if lpips and lpips <= best_lpips + 0.05:
    #     is_better["lpips"] = True

    return is_better


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
    num_processes = accelerator.num_processes

    if num_processes > device_count:
        raise Exception(
            f"Number of processes: {num_processes} must be less than or equal to number of GPUs {device_count}"
        )

    accelerator.print("Device count: ", device_count)
    accelerator.print("Num processes: ", accelerator.num_processes)
    accelerator.print("Mixed precision: ", accelerator.mixed_precision)
    accelerator.print("Data type: ", data_type)

    # Model loading and setup
    accelerator.print("\n--------------- Setup ---------------")
    accelerator.print("Loading model for inversion...")

    vlm_checkpoint = config["vlm_checkpoint"]

    vlm_model = AutoModelForPreTraining.from_pretrained(vlm_checkpoint)
    vlm_model = vlm_model.to(device=accelerator.device, dtype=data_type)
    # vlm_model = vlm_model.to(device=accelerator.device)
    vlm_model.eval()

    if use_generate:
        vlm_model.generate = MethodType(GenerationMixin.generate, vlm_model)

    vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint)
    vlm_processor.patch_size = vlm_model.config.vision_config.patch_size
    vlm_processor.image_processor.size = {"shortest_edge": 336}

    guide_model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    guide_model = guide_model.to(device=accelerator.device, dtype=data_type)
    # guide_model = guide_model.to(device=accelerator.device)
    guide_model.eval()

    base_hook_manager = FeatureStatsHookManager(model=vlm_model, model_type="vlm")
    base_hook_manager.register_hooks()

    guide_hook_manager = FeatureStatsHookManager(model=guide_model, model_type="cnn")
    guide_hook_manager.register_hooks()

    accelerator.wait_for_everyone()

    trainer = MIMICTrainer(
        vlm_model,
        vlm_processor,
        guide_model,
        accelerator,
        base_hook_manager,
        guide_hook_manager,
        use_generate=use_generate,
    )

    return trainer, accelerator


def run(trainer, parameters, accelerator):
    try:
        # Make directories
        if accelerator.is_main_process:
            os.makedirs(parameters["outputs_dir"], exist_ok=True)
            os.makedirs(parameters["results_dir"], exist_ok=True)

        # ----------- Load target class info ------------
        accelerator.print("\n-------- Target Class Info ---------------")
        prompt = next(
            (c for c in PROMPT_LISTS if c["class_index"] == parameters["target_class"]),
            None,
        )

        if not prompt:
            raise Exception(
                f"Target class {parameters['target_class']} not found in {PROMPT_FILEPATH}"
            )

        target_class = prompt["class_index"]
        chat_sequence = prompt["chat_sequence"]

        accelerator.print("Target class: ", target_class)
        accelerator.print("Chat Sequence: ", chat_sequence)

        # ------------- Load target features ------------
        target_features_filepath = get_target_features_filepath(
            parameters["target_features_dir"], target_class
        )
        target_features = torch.load(
            target_features_filepath, map_location=accelerator.device
        )
        accelerator.print(
            f"Target feature stats loaded from {target_features_filepath}"
        )

        dataloader = ImageNetDataloader(
            dataset=ImageNetDataset(
                class_index=target_class, root_dir=parameters["dataset_dir"]
            ),
            batch_size=100,
        )

        # Training
        accelerator.print("\n------------ Training started ------------")
        start_time = time.time()
        trainer.train(
            chat_sequence,
            target_class,
            target_features,
            parameters,
            compute_metrics=lambda images, target_class, target_label: compute_metrics(
                images, target_class, target_label, dataloader
            ),
            compare_results=lambda best_results, results: compare_results(
                best_results, results
            ),
        )
        best_results = trainer.best_results
        best_iterations = trainer.best_iterations
        end_time = time.time()

        accelerator.print("\n------------ Training ended ------------")
        print(
            f"Training time @ rank-{accelerator.local_process_index}: {(end_time - start_time) / 60:.2f} minutes"
        )

        return best_results, best_iterations
    except Exception:
        tb_str = traceback.format_exc()
        accelerator.print(tb_str)
    finally:
        # Clean up
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        required=True,
        help="Path to the VLM checkpoint.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mimic",
        required=False,
        help="Wandb project name. '_<target_class>' will be appended. Default: mimic",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Imagenet dataset directory.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for reproducibility. Default: 0",
    )
    parser.add_argument(
        "--batch_size_per_device",
        default=2,
        type=int,
        help="Batch size per device for the image reconstruction of the target class. Default: 2",
    )
    parser.add_argument(
        "--image_resolution",
        default=336,
        type=int,
        help="Image resolution of the image to be inverted. Recommended to use the same resolution as the VLM model. Default: 336",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=4,
        type=int,
        help="Gradient accumulation steps. Default: 4",
    )
    parser.add_argument(
        "--n_iterations",
        default=3000,
        type=int,
        help="Number of total iterations including jitter. Default: 3000",
    )
    parser.add_argument(
        "--n_jitter_iterations",
        type=int,
        default=0,
        help="Number of iterations with jitter, use 0 to use no jitter. Default: 0",
    )
    parser.add_argument("--jitter", default=8, type=int, help="Input jitter")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for optimization, if selected BF16 will be preferred",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="./outputs",
        help="Directory to store inverted images during training. Default: ./outputs",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to store the final best inverted images after the training. Default: ./results",
    )
    parser.add_argument(
        "--target_features_dir",
        type=str,
        default="./target_features",
        help="Directory containing pre-computed target feature stats. Default: ./data/target_features",
    )
    parser.add_argument(
        "--do_flip", action="store_true", help="Apply flip during model inversion"
    )
    parser.add_argument(
        "--use_blank_image",
        action="store_true",
        help="Use blank image as input instead of noise",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        required=True,
        choices=AVALIABLE_CLASSES,
        help="Target class from the ImageNet classes.",
    )

    parser.add_argument(
        "--verifier_arch",
        type=str,
        default="mobilenet_v2",
        help="Arch name from torchvision models to act as a verifier. Default: mobilenet_v2",
    )

    # Coefficients for optimization
    parser.add_argument(
        "--base_feature_loss_type",
        type=str,
        default="l2",
        help="Use L2-norm or KL div for base feature loss. Default: l2",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.25,
        help="Learning rate for optimization. Default: 0.25",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.005,
        help="Minimum learning rate for scheduler. Default: 0.005",
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=100,
        help="Warmup length for cosine scheduler learning rate. Default: 100",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save every n iterations. Default: 500",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=500,
        help="Log every n iterations. Default: 500",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Evaluate every n iterations. Default: 500",
    )
    parser.add_argument(
        "--first_bn_multiplier",
        type=float,
        default=10.0,
        help="Additional multiplier on first BN layer of the guide model. Default: 10.0",
    )
    parser.add_argument(
        "--main_loss_scale",
        type=float,
        default=0.05,
        help="Coefficient for the output loss in optimization",
    )
    parser.add_argument(
        "--feature_guide_scale",
        type=float,
        default=0.005,
        help="Coefficient for the guide model feature regularization",
    )
    parser.add_argument(
        "--feature_base_scale",
        type=float,
        default=0.0001,
        help="Coefficient for the base model feature regularization",
    )
    parser.add_argument(
        "--patch_prior_scale",
        type=float,
        default=0.00001,
        help="Coefficient for the patch prior regularization",
    )
    parser.add_argument(
        "--tv_l1_scale",
        type=float,
        default=0.0001,
        help="Coefficient for Total Variation L1 regularization",
    )
    parser.add_argument(
        "--tv_l2_scale",
        type=float,
        default=0.000001,
        help="coefficient for Total Variation L2 regularization",
    )
    parser.add_argument(
        "--l2_scale",
        type=float,
        default=0.000001,
        help="Coefficient for L2 regularization",
    )
    parser.add_argument(
        "--use_generate",
        action="store_true",
        help="Coefficient for L2 regularization",
    )
    args = parser.parse_args()

    parameters = vars(args)
    parameters["wandb_project"] = (
        parameters["wandb_project"] + f"_{parameters['target_class']}"
    )
    config_keys = [
        "vlm_checkpoint",
        "fp16",
        "seed",
        "grad_accumulation_steps",
        "use_generate",
    ]
    config = {k: parameters[k] for k in config_keys}
    trainer, accelerator = setup_trainer(config)
    parameters["local_rank"] = accelerator.local_process_index

    accelerator.print(
        f"\n------------ Target Class: {parameters['target_class']} | Parameters -------------"
    )
    accelerator.print(f"{json.dumps(parameters, indent=2)}")

    best_results, best_iteration = run(trainer, parameters, accelerator)

    # Since we are gathering inputs for compute_metrics, only rank 0 will have best_results
    accelerator.print(
        f"Best results @ {best_iteration}: {json.dumps(best_results, indent=2)}"
    )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
