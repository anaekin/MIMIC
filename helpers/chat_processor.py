import re

import torch


class ChatProcessor:
    """
    Tokenize chat sequence into input_ids, attention_mask, labels, and pixel_values.

    Args:
      vlm_processor: VLMProcessor to use for tokenization. E.g. LLaVAProcessor
      image_token (optional): Token to use for image prompt (default:  `"<image>"`)
      use_generate: If True, the last assistant message is removed and the model is expected to generate a response.
            If False, the last assistant message is included and the model is expected to predict it.

            Example:
            chat = [("What animal is in this image: tiger, leopard or lion?", "tiger")]

    """

    def __init__(self, vlm_processor, image_token="<image>", use_generate=False):
        self.processor = vlm_processor
        self.image_processor = vlm_processor.image_processor
        self.image_token = image_token
        self.tokenizer = vlm_processor.tokenizer
        self.use_generate = use_generate

    def _create_chat_prompt(self, chat):
        """
        Create a single chat prompt from chat sequence.

        Args:
          chat: List of tuples. Each tuple contains the user message and the assistant message.
            Example:
            chat = [("What animal is in this image: tiger, leopard or lion?", "tiger")]
          batch_size: Number of chat templates to create.

        """
        chat_list = [
            {
                "role": "user",
                "content": f"{self.image_token}\n{chat[0][0]}",
            },
            {
                "role": "assistant",
                "content": chat[0][1],
            },
        ]

        for user_msg, assistant_msg in chat[1:]:
            chat_list.append({"role": "user", "content": user_msg})
            chat_list.append({"role": "assistant", "content": assistant_msg})

        chat_list.pop()  # Remove the target text

        chat_prompt = self.tokenizer.apply_chat_template(
            chat_list, tokenize=False, add_generation_prompt=True
        )

        return chat_prompt

    # def _create_labels(self, input_ids):
    #     """
    #     Create labels for the input_ids by masking all tokens except the assistant response.

    #     This function assumes that the input_ids was obtained by text with chat template that has one user message followed by one assistant response.

    #     Example:
    #       <|begin_of_text|><|start_header_id|>user<|end_header_id|><image>What animal is in this image: tiger, leopard or lion?<|eot_id|>
    #       <|start_header_id|>assistant<|end_header_id|>tiger<|eot_id|>
    #     """
    #     labels = input_ids.clone()
    #     labels[:, :] = -100

    #     decoded_prompt = self.tokenizer.decode(
    #         input_ids[0],
    #         skip_special_tokens=False,
    #         spaces_between_special_tokens=False,
    #         clean_up_tokenization_spaces=False,
    #     )
    #     print("Decoded prompt: ", decoded_prompt)

    #     # Regex to extract assistant response span
    #     match = re.search(
    #         r"<\|[^\|]+?\|>assistant<\|[^\|]+?\|>\s{2}(.*?)(<\|eot_id\|>)",
    #         decoded_prompt,
    #         re.DOTALL,
    #     )

    #     assistant_start, assistant_end = match.start(1), match.end(2)

    #     cur = 0
    #     for i, token_id in enumerate(input_ids[0]):
    #         token_str = self.tokenizer.decode(
    #             [token_id],
    #             skip_special_tokens=False,
    #             clean_up_tokenization_spaces=False,
    #         )
    #         start = decoded_prompt.find(token_str, cur)
    #         if start == -1:
    #             continue
    #         end = start + len(token_str)
    #         cur = end

    #         if assistant_start <= start < end <= assistant_end:
    #             labels[:, i] = token_id
    #         else:
    #             labels[:, i] = -100

    #     return labels

    def _create_qwen_chat_list(self, chat, image=None):
        """
        Create a chat list compatible with Qwen3VL processor using explicit structured content.
        """
        user_msg, assistant_msg = chat[0]
        content = []
        if image is not None:
             content.append({"type": "image", "image": image})
        
        content.append({"type": "text", "text": user_msg})
        
        conversation = [
             {
                "role": "user",
                "content": content
             },
             {
                "role": "assistant",
                "content": assistant_msg
             }
        ]
        
        for u, a in chat[1:]:
             conversation.append({"role": "user", "content": u})
             conversation.append({"role": "assistant", "content": a})
        
        conversation.pop() # Remove target
        return conversation

    def preprocess(self, chat, images=None, batch_size=1):
        """
        Tokenize chat sequence into input_ids, attention_mask, labels, and pixel_values.

        Args:
          chat: List of tuples (user_msg, assistant_msg). It will be converted to batch of same chat template.
          images: List of PIL images or numpy arrays or tensors of shape (B, C, H, W)
          batch_size: Batch size for tokenization

        """
        processor_name = self.processor.__class__.__name__.lower()
        is_qwen = "<|vision_start|>" in self.image_token or "qwen" in processor_name
        is_gemma = "gemma" in processor_name
        
        target = chat[-1][1]
        targets = [target] * batch_size

        image_inputs = {}
        if (is_qwen or is_gemma) and images:
            # Qwen/Gemma path: Use processor.apply_chat_template with structured multimodal input
            prompts = []
            for b in range(batch_size):
                img = images[b] if b < len(images) else images[0]
                prompts.append(self._create_qwen_chat_list(chat, img))

            inputs = self.processor.apply_chat_template(
                prompts,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            prompt_ids = inputs["input_ids"]
            # Qwen processor often returns 'pixel_values' and 'image_grid_thw'
            image_inputs = {k: v for k, v in inputs.items() if k != "input_ids" and k != "attention_mask"}
            pixel_values = image_inputs.get("pixel_values")
             
        else:
            # Standard Path (LLaVA-like)
            prompt = self._create_chat_prompt(chat)
            prompts = [prompt] * batch_size

            if images:
                image_inputs = self.image_processor(images, return_tensors="pt")
                pixel_values = image_inputs.get("pixel_values")
            else:
                pixel_values = None

            # Tokenize with return_offsets_mapping for masking
            if images is not None and len(images) > 0 and self.processor:
                # Use processor to correctly expand image tokens
                prompt_ids = self.processor(
                    text=prompts, images=images, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
            else:
                prompt_ids = self.tokenizer(
                    prompts, return_tensors="pt", add_special_tokens=False
                )["input_ids"]

        target_ids = self.tokenizer(
            targets, return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        if self.use_generate:
            input_ids = prompt_ids
            labels = target_ids
            attention_mask = torch.ones_like(input_ids)
        else:
            # labels = self._create_labels(input_ids)
            # Ensure devices match
            if prompt_ids.device != target_ids.device:
                target_ids = target_ids.to(prompt_ids.device)
                
            input_ids = torch.cat([prompt_ids, target_ids], dim=1)
            labels = torch.full_like(input_ids, -100)
            labels[:, -target_ids.shape[1] :] = target_ids
            attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, labels, pixel_values, image_inputs
