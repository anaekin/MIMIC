import torch
import torch.nn.functional as F

from einops import rearrange


def gaussian_blur(image_tensor, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur to image tensor in-place (returns new tensor).
    Used as a periodic denoising step during optimization to physically
    remove high-frequency noise that accumulates from gradient updates.

    Args:
        image_tensor: [B, C, H, W] tensor in normalized space
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel
    Returns:
        Blurred image tensor (same shape)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=image_tensor.dtype, device=image_tensor.device) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Make 2D kernel via outer product
    kernel_2d = g.outer(g)
    # Shape: [C_out, C_in/groups, kH, kW] — depthwise
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    
    pad = kernel_size // 2
    blurred = F.conv2d(image_tensor, kernel_2d, padding=pad, groups=3)
    return blurred


def get_patch_internal_variance(inputs_jit, patch_size=14):
    """
    Penalizes high-frequency noise INSIDE the patches.
    """
    # Rearrange to patch grid: B x (H_grid) x (W_grid) x P x P x C
    patches = rearrange(
        inputs_jit, "b c (h p1) (w p2) -> b h w p1 p2 c", p1=patch_size, p2=patch_size
    )
    
    # Calculate variance within each patch
    # We want low variance (solid colors/smooth gradients) NOT noise
    patch_var = torch.var(patches, dim=(3, 4)) 
    
    return patch_var.mean()


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
    reg_l2 = torch.norm(inputs_jit.reshape(inputs_jit.size(0), -1), dim=1).mean() / (
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


def get_edge_aware_tv(inputs_jit, epsilon=0.01):
    """
    Edge-aware (anisotropic) Total Variation regularizer.
    Smooths aggressively where local gradients are weak (background)
    and gently where they are strong (subject edges).

    Uses the image's own gradient magnitude (detached) to weight the TV penalty,
    so this does not affect gradient flow through the weighting itself.

    Args:
        inputs_jit: Batch of generated images [B, C, H, W]
        epsilon: Small constant to avoid division by zero and control edge sensitivity.
                 Smaller = sharper edge preservation, larger = more uniform smoothing.
    """
    # Pixel differences
    diff_h = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]   # [B, C, H-1, W]
    diff_w = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]   # [B, C, H, W-1]

    # Edge strength: mean absolute gradient across channels (detached)
    edge_h = diff_h.detach().abs().mean(dim=1, keepdim=True)  # [B, 1, H-1, W]
    edge_w = diff_w.detach().abs().mean(dim=1, keepdim=True)  # [B, 1, H, W-1]

    # Inverse weighting: smooth more where edges are weak
    weight_h = 1.0 / (edge_h + epsilon)
    weight_w = 1.0 / (edge_w + epsilon)

    # Normalize so mean weight ≈ 1 (keeps scale comparable to standard TV)
    weight_h = weight_h / (weight_h.mean() + 1e-8)
    weight_w = weight_w / (weight_w.mean() + 1e-8)

    # Weighted TV-L1
    reg = (weight_h * diff_h.abs()).mean() + (weight_w * diff_w.abs()).mean()
    return reg
