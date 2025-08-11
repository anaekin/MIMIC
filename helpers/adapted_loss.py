import torch
import torch.nn.functional as F
import torch.nn as nn


class AdaptedLoss(nn.Module):
    def __init__(
        self,
        loss_cls=nn.CrossEntropyLoss,
        loss_kwargs=None,
        image_token_id=128003,
        num_image_tokens=576,
        use_generate=False,
    ):
        super().__init__()
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens
        self.use_generate = use_generate

        loss_kwargs = loss_kwargs or {}
        self.loss_fn = loss_cls(**(loss_kwargs or {"ignore_index": -100}))

    def forward(self, logits, labels, input_ids):
        if self.use_generate:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)), labels.view(-1)  # [B*L, V]  # [B*L]
            )
            return loss, labels

        # For forward pass, we need to handle the image token and the causal shift
        B, T = labels.shape
        final_labels = torch.full_like(
            logits[:, :, 0],
            self.loss_fn.ignore_index,
            dtype=labels.dtype,
            device=labels.device,
        )

        for b in range(B):
            image_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=False)
            if image_pos.numel() == 0:
                raise ValueError(
                    f"No <image> token (ID={self.image_token_id}) found in input_ids[{b}]"
                )
            image_pos = image_pos.squeeze().item()

            final_labels[b, :image_pos] = labels[b, :image_pos]
            final_labels[b, image_pos + self.num_image_tokens :] = labels[
                b, image_pos + 1 :
            ]  # B, T, V

        # Causal Shift
        # We predict the next token based on the previous tokens
        # i.e. logits at postition 't' predicts label at position 't+1'
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = final_labels[:, 1:].contiguous()  # (B, T-1)

        logits_flat = shift_logits.view(-1, logits.size(-1))  # B * T-1, V
        labels_flat = shift_labels.view(-1)  # B * T-1

        return self.loss_fn(logits_flat, labels_flat.long()), final_labels
