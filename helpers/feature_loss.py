import torch


def get_feature_loss(feature_pairs, loss_type="l2", rescale=None):
    """
    Computes the base feature loss based on the specified loss type.
    Args:
        feature_pairs: List of pair of img features and layer features
            Example:
                [[[[mean_img, var_img], [mean_target, var_target]], ...]....] # [L, 2, 2, C]
        loss_type: Type of loss to compute ("l2" or "kl")
        rescale (optional): List of weights for each layer. Should be equal to the length of `feature_pairs`.

    Returns:
        loss: Computed loss based on the specified type
    """
    if rescale is None:
        rescale = [1.0] * len(feature_pairs)
    else:
        assert len(rescale) == len(
            feature_pairs
        ), "Length of rescale should be equal to the length of feature_pairs"

    if loss_type == "l2":
        return _get_feature_l2(feature_pairs, rescale)
    elif loss_type == "kl":
        return _get_feature_kl(feature_pairs, rescale)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def _get_feature_l2(feature_pairs, rescale):
    l2_losses = []

    for idx, (img_features, target_features) in enumerate(
        feature_pairs
    ):  # Looping over layers L
        mean_img, var_img = img_features  # shape: (C,), (C,)
        mean_layer, var_layer = target_features  # shape: (C,), (C,)

        l2_loss = torch.norm(mean_img - mean_layer, 2) + torch.norm(
            var_img - var_layer, 2
        )
        l2_losses.append(l2_loss * rescale[idx])

    return torch.stack(l2_losses).sum()


def _get_feature_kl(feature_pairs, rescale):
    eps = 1e-6
    kl_losses = []

    for idx, (img_features, target_features) in enumerate(feature_pairs):
        mean_img, var_img = img_features  # shape: (C,), (C,)
        mean_layer, var_layer = target_features  # shape: (C,), (C,)

        var_img = torch.clamp(var_img, min=eps)
        var_layer = torch.clamp(var_layer, min=eps)

        kl_per_channel = 0.5 * (
            torch.log(var_layer / var_img)
            + (var_img + (mean_layer - mean_img).pow(2)) / var_layer
            - 1
        )  # shape: (C,)

        kl_loss = kl_per_channel.mean()  # average over channels
        kl_losses.append(kl_loss * rescale[idx])

    return torch.stack(kl_losses).sum()
