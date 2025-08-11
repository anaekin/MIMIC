import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from helpers.utils import denormalize

lpips_model = (
    LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device="cpu").eval()
)


def compute_lpips(inputs, dataloader) -> dict:
    input_images = inputs.to(device="cpu", dtype=torch.float32)
    gen_images = denormalize(input_images)

    ref_images, _ = next(iter(dataloader))
    ref_images = ref_images.to(device="cpu", dtype=torch.float32).div(255.0)

    with torch.no_grad():
        score = lpips_model(gen_images, ref_images)

    return {"lpips": score.item()}
