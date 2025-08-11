import sys
from transformers import (
    AutoProcessor,
    AutoModelForPreTraining,
)
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from custom_datasets.imagenet import ImageNetDataloader, ImageNetDataset
from helpers.chat_processor import ChatProcessor
from helpers.adapted_loss import AdaptedLoss


def pixel_values_to_pil_image(pv):
    pv = pv.detach().cpu()

    proc = vlm_processor.image_processor
    mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(proc.image_std, dtype=torch.float32).view(3, 1, 1)
    img = pv * std + mean  # now in [0,1] approximately

    img = img.clamp(0.0, 1.0)
    pil = to_pil_image(img)  # by default scales [0,1]→[0,255] uint8

    return pil


device = "cuda" if torch.cuda.is_available() else "cpu"
vlm_checkpoint = "./models/llava_3_8b_cache_merged"
vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint)
vlm_processor.image_processor.size = {"shortest_edge": 336}
tokenizer = vlm_processor.tokenizer
tokenizer.padding_side = "left"

vlm_model = AutoModelForPreTraining.from_pretrained(vlm_checkpoint)
vlm_model = vlm_model.to(device=device)
vlm_model.eval()

img_token = "<image>"

# # Use a local image
save_path = "image.png"
image = Image.open(save_path).convert("RGB")

# or Get image from dataloader
# dataset = ImageNetDataset(
#     class_index=805,
#     root_dir=<imagenet_root_dir>,
# )
# dataloader = ImageNetDataloader(dataset=dataset)
# image = None
# for images, _ in dataloader:
#     image = images[0]
#     break

chat_processor = ChatProcessor(vlm_processor, use_generate=True)
chat_sequence = [
    (
        "Choose the animal in this image: Tiger, Leopard or Lion?",
        "Tiger",
    )
]
input_ids, attention_mask, labels, pixel_values = chat_processor.preprocess(
    chat_sequence, images=[image], batch_size=1
)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
pixel_values = pixel_values.to(device)
labels = labels.to(device) if labels is not None else None

# To test how image looks after processing
# pil = pixel_values_to_pil_image(pv=pixel_values[0])
# pil.save("processed_pixel_values.png")
print("Input IDs:", input_ids.shape)
print("Attention Mask:", attention_mask.shape)
print("Pixel Values:", pixel_values.shape)
print("Labels:", labels.shape)

# # Generate
outputs = vlm_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    max_new_tokens=labels.shape[1],
    return_dict_in_generate=True,
    output_scores=True,
)

sequence = outputs.sequences
scores = outputs.scores
print("Sequence:", sequence.shape)
print("Scores:", len(scores))
print("Scores:", scores[0].shape)

logits = torch.stack(outputs.scores, dim=1)
llava_loss = AdaptedLoss(use_generate=True)
loss, final_labels = llava_loss(logits, labels, input_ids)
print("Loss:", loss.item())

output = vlm_processor.batch_decode(
    sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output)
