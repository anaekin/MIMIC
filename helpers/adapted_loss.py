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

        # Since input is already expanded, logits and labels should be aligned.
        # We perform standard Causal Shift.

        # Causal Shift
        # We predict the next token based on the previous tokens
        # i.e. logits at postition 't' predicts label at position 't+1'
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, T-1)

        logits_flat = shift_logits.view(-1, logits.size(-1))  # B * T-1, V
        labels_flat = shift_labels.view(-1)  # B * T-1

        return self.loss_fn(logits_flat, labels_flat.long()), labels
