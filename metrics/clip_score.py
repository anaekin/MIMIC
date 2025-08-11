from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from helpers.utils import denormalize

# Path to your locally saved CLIP model
clip_checkpoint = "./models/clip-vit-base-patch32"


# Define loader for torchmetrics to accept local model
def _load_clipscore_model():
    model = CLIPModel.from_pretrained(clip_checkpoint).to("cpu").eval()
    processor = CLIPProcessor.from_pretrained(clip_checkpoint)
    processor.image_processor.do_rescale = False
    return model, processor


# TorchMetrics CLIPScore metric using the loader
metric = CLIPScore(model_name_or_path=_load_clipscore_model)
to_pil = ToPILImage()


# CLIPScore calculation function
def compute_clipscore(inputs, target_label):
    input_images = inputs.to("cpu", dtype=torch.float32)
    images = denormalize(input_images)
    targets = [target_label] * len(images)  # One caption per image

    score = metric(images, targets)
    return {"clipscore": score.detach().item()}
