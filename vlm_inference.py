import sys
import torch
import argparse
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from custom_datasets.imagenet import ImageNetDataloader, ImageNetDataset
from helpers.chat_processor import ChatProcessor
from helpers.adapted_loss import AdaptedLoss
from helpers.utils import load_vlm, prepare_vlm_inputs, VLM_CHOICES


# def pixel_values_to_pil_image(pv):
#     pv = pv.detach().cpu()

#     proc = vlm_processor.image_processor
#     mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(3, 1, 1)
#     std = torch.tensor(proc.image_std, dtype=torch.float32).view(3, 1, 1)
#     img = pv * std + mean  # now in [0,1] approximately

#     img = img.clamp(0.0, 1.0)
#     pil = to_pil_image(img)  # by default scales [0,1]→[0,255] uint8

#     return pil


def generate_with_grad_at_index(model, inputs, target_index, max_new_tokens=200):
    """
    Generates tokens up to target_index-1, runs a forward pass at target_index with gradients,
    and optionally continues generation.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 1. Generate up to target_index
    current_ids = input_ids
    if target_index > 0:
        with torch.inference_mode():
            # We generate exactly target_index new tokens
            out = model.generate(
                **inputs, 
                max_new_tokens=target_index, 
                return_dict_in_generate=True,
                use_cache=True
            )
            current_ids = out.sequences

    # 2. Forward pass for the target token
    # Extend attention_mask to match current_ids length
    orig_len = attention_mask.shape[1]
    curr_len = current_ids.shape[1]
    
    if curr_len > orig_len:
        pad = torch.ones((attention_mask.shape[0], curr_len - orig_len), device=attention_mask.device, dtype=attention_mask.dtype)
        # Assuming padding on the right for generated tokens
        new_mask = torch.cat([attention_mask, pad], dim=1)
    else:
        new_mask = attention_mask

    # Prepare forward kwargs (exclude input_ids and attention_mask)
    forward_kwargs = {k: v for k, v in inputs.items() if k not in ['input_ids', 'attention_mask']}
    
    # Run forward pass with gradients enabled
    with torch.enable_grad():
        outputs = model(
            input_ids=current_ids,
            attention_mask=new_mask,
            **forward_kwargs
        )
        
    # 3. Resume generation for remaining tokens
    remaining = max_new_tokens - target_index
    final_output = None
    
    if remaining > 0:
        with torch.inference_mode():
            final_output = model.generate(
                input_ids=current_ids,
                attention_mask=new_mask,
                **forward_kwargs,
                max_new_tokens=remaining,
                return_dict_in_generate=True,
                output_scores=True
            )
            
    return outputs, final_output



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vlm", type=str, default="llava-llama3-8b", choices=VLM_CHOICES)
    parser.add_argument("--text_prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--image_path", type=str, default="./data/images/1.jpg", required=False)
    parser.add_argument("--target_token_index", type=int, default=-1, help="Index of token to compute gradients for. If -1, runs standard generation.")
    args = parser.parse_args()
    
    print(" --- VLM Inference Script --- ")
    print(f"Using VLM: {args.vlm}")
    print(f"Using device: {args.device}")
    print(f"Using text prompt: {args.text_prompt}")
    print(f"Using image path: {args.image_path}")
    print("-----------------------------")

    device = args.device
    processor, tokenizer, model = load_vlm(args.vlm)
    model = model.to(device=device)
    model.eval()
    
    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")
        
    
    
    inputs = prepare_vlm_inputs(processor, image, args.text_prompt, device)
    input_len = inputs["input_ids"].shape[-1]
    
    if args.target_token_index >= 0:
        if "pixel_values" in inputs:
            inputs["pixel_values"].requires_grad = True

        outputs_grad, final_output = generate_with_grad_at_index(model, inputs, args.target_token_index, max_new_tokens=200)
        
        if final_output is not None:
             sequence = final_output.sequences
             print(processor.batch_decode(sequence[0][input_len:], skip_special_tokens=True))
    else:
        with torch.inference_mode():
            # autoregressively complete prompt
            output = model.generate(**inputs, max_new_tokens=200,return_dict_in_generate=True,output_scores=True)
            sequence = output.sequences
            scores = output.scores
            print(processor.batch_decode(sequence[0][input_len:], skip_special_tokens=True))
            print("--- Scores length:", len(scores[0].shape))
            indices = [indx[0] for indx in torch.stack(scores).argmax(-1)]
        
    
# input_ids = input_ids.to(device)
# attention_mask = attention_mask.to(device)
# pixel_values = pixel_values.to(device)
# labels = labels.to(device) if labels is not None else None

# # To test how image looks after processing
# # pil = pixel_values_to_pil_image(pv=pixel_values[0])
# # pil.save("processed_pixel_values.png")
# print("Input IDs:", input_ids.shape)
# print("Attention Mask:", attention_mask.shape)
# print("Pixel Values:", pixel_values.shape)
# print("Labels:", labels.shape)

# # # Generate
# outputs = vlm_model.generate(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     pixel_values=pixel_values,
#     max_new_tokens=labels.shape[1],
#     return_dict_in_generate=True,
#     output_scores=True,
# )

# sequence = outputs.sequences
# scores = outputs.scores
# print("Sequence:", sequence.shape)
# print("Scores:", len(scores))
# print("Scores:", scores[0].shape)

# logits = torch.stack(outputs.scores, dim=1)
# llava_loss = AdaptedLoss(use_generate=True)
# loss, final_labels = llava_loss(logits, labels, input_ids)
# print("Loss:", loss.item())

# output = vlm_processor.batch_decode(
#     sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output)
