import torch

from einops import rearrange


def get_patch_prior(inputs_jit, patch_size=14):
    """
    Computes patch loss based on the GradViT paper.
    The regularizer is L2 norm of the differences between
    adjacent patches in the vertical and horizontal directions.


    Args:
        inputs_jit: Generated images.
        patch_size (optional): Size of the patches (default: `14`)
    """
    B, C, H, W = inputs_jit.shape
    assert H % patch_size == 0 and W % patch_size == 0

    # Rearrange to patch grid
    patches = rearrange(
        inputs_jit, "b c (h p1) (w p2) -> b h w p1 p2 c", p1=patch_size, p2=patch_size
    )

    # Vertical differences
    v_diff = patches[:, 1:, :, 0, :, :] - patches[:, :-1, :, -1, :, :]

    # Horizontal differences
    h_diff = patches[:, :, 1:, :, 0, :] - patches[:, :, :-1, :, -1, :]

    # reg_patch_ = torch.norm(v_diff, p=2) + torch.norm(h_diff, p=2)
    per_image_loss = (v_diff**2).mean(dim=(1, 2, 3, 4)) + (h_diff**2).mean(
        dim=(1, 2, 3, 4)
    )
    reg_patch = per_image_loss.mean()

    return reg_patch


def get_extra_priors(inputs_jit, diagonal_smoothing=True):
    """
    Computes extra prior regularizers (L2, TV_L1, TV_L2) for the generated images.

    Args:
        inputs_jit: Batch of generated images
        diagonal_smoothing (optional): Whether to include diagonal smoothing for Total Variance regularizers (defult: `False`)
    """
    # Normalizing with number of pixels - C * H * W
    B, C, H, W = inputs_jit.shape
    reg_l2 = torch.norm(inputs_jit.view(inputs_jit.size(0), -1), dim=1).mean() / (
        C * H * W
    )

    # Without normalizing
    # reg_l2 = torch.norm(inputs_jit.view(inputs_jit.size(0), -1), dim=1).mean()

    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]

    reg_tv_l1 = diff1.abs().mean() + diff2.abs().mean()
    # reg_tv_l2 = torch.linalg.norm(diff1) + torch.linalg.norm(diff2)
    reg_tv_l2 = (diff1**2).mean() + (diff2**2).mean()
    # reg_tv_l2 = torch.norm(diff1) + torch.norm(diff2)

    if diagonal_smoothing:
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]  # Diagonal /
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]  # Diagonal \

        reg_tv_l1 += diff3.abs().mean() + diff4.abs().mean()
        # reg_tv_l2 += torch.linalg.norm(diff3) + torch.linalg.norm(diff4)
        reg_tv_l2 += (diff3**2).mean() + (diff4**2).mean()
        # reg_tv_l2 += torch.norm(diff3) + torch.norm(diff4)

    return reg_l2, reg_tv_l1, reg_tv_l2
