"""Render 3D Gaussians using the gsplat library (pip install gsplat)."""

import math
import torch
import torch.nn.functional as F

try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False


def render_gaussians(gaussians: dict,
                      camera,
                      background: torch.Tensor = None,
                      sh_degree: int = None) -> torch.Tensor:
    """
    Render a set of (possibly deformed) Gaussians from a single camera viewpoint.

    Args:
        gaussians: dict with keys xyz [N,3], features [N,K,3], opacity [N],
                   scaling [N,3], rotation [N,4], sh_degree int
        camera:    Camera object (from camera_utils)
        background: [3] background color. Defaults to black.
        sh_degree:  Override SH degree. Defaults to gaussians['sh_degree'].

    Returns:
        rendered: [3, H, W] float32 in [0, 1]
    """
    if not GSPLAT_AVAILABLE:
        raise ImportError("gsplat is required for rendering. Run: pip install gsplat")

    device = gaussians["xyz"].device
    if background is None:
        background = torch.zeros(3, device=device)

    sh_deg = sh_degree if sh_degree is not None else gaussians["sh_degree"]

    means   = gaussians["xyz"]                        # [N, 3]
    quats   = F.normalize(gaussians["rotation"], dim=-1)  # [N, 4]
    scales  = torch.exp(gaussians["scaling"])         # [N, 3] — log→exp
    opacities = torch.sigmoid(gaussians["opacity"])   # [N]   — logit→sigmoid
    colors  = gaussians["features"]                   # [N, K, 3] SH coefficients

    viewmat = camera.world_to_cam.unsqueeze(0).to(device)  # [1, 4, 4]
    K       = camera.K.unsqueeze(0).to(device)             # [1, 3, 3]

    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=camera.width,
        height=camera.height,
        sh_degree=sh_deg,
        backgrounds=background.unsqueeze(0),
    )
    # renders: [1, H, W, 3] → [3, H, W]
    return renders[0].permute(2, 0, 1).clamp(0.0, 1.0)


def render_trajectory(gaussians_canonical: dict,
                       theta_traj: torch.Tensor,
                       rest_joints: torch.Tensor,
                       parents: torch.Tensor,
                       lbs_weights: torch.Tensor,
                       cameras: list,
                       motion_mask: torch.Tensor = None,
                       background: torch.Tensor = None,
                       sh_degree: int = None) -> list:
    """
    Render a sequence of frames.

    theta_traj: [T, N_j, 4]  predicted joint rotations
    cameras:    list of Camera (length T)

    Returns: list of [3, H, W] tensors.
    """
    from .fk_lbs import deform_gaussians

    frames = []
    for t, cam in enumerate(cameras):
        deformed = deform_gaussians(
            gaussians_canonical,
            theta_traj[t],
            rest_joints, parents, lbs_weights, motion_mask,
        )
        img = render_gaussians(deformed, cam, background=background, sh_degree=sh_degree)
        frames.append(img)

    return frames
