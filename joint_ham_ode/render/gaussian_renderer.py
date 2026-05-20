"""
Render 3D Gaussians.

Primary backend: diff-gaussian-rasterization (same CUDA rasterizer as 3DGS / RigGS).
  Install from RigGS submodules directory:
    pip install ./submodules/diff-gaussian-rasterization

Fallback backend: gsplat (pure-pip alternative, slightly different output).
  pip install gsplat
"""

import math
import torch
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    DGR_AVAILABLE = True
except Exception as _dgr_err:
    DGR_AVAILABLE = False
    _dgr_err_msg = str(_dgr_err)

try:
    from gsplat import rasterization as _gsplat_rasterization
    GSPLAT_AVAILABLE = True
except Exception:
    GSPLAT_AVAILABLE = False

# Print backend selection once at import time so the user can verify
if DGR_AVAILABLE:
    print("[gaussian_renderer] Backend: diff-gaussian-rasterization (3DGS/RigGS native)")
elif GSPLAT_AVAILABLE:
    print("[gaussian_renderer] Backend: gsplat (fallback — diff-gaussian-rasterization not found)")
    if not DGR_AVAILABLE:
        print(f"[gaussian_renderer]   diff-gaussian-rasterization import failed: {_dgr_err_msg}")
else:
    print("[gaussian_renderer] WARNING: no rasterizer installed. "
          "Run: pip install ./submodules/diff-gaussian-rasterization")


# ---------------------------------------------------------------------------
# Camera helpers (3DGS convention)
# ---------------------------------------------------------------------------

def _get_projection_matrix(fovx: float, fovy: float,
                             znear: float = 0.01, zfar: float = 100.0,
                             device: str = "cuda") -> torch.Tensor:
    """OpenGL-style perspective matrix used by 3DGS/RigGS rasterizer."""
    tanHalfFovX = math.tan(fovx / 2)
    tanHalfFovY = math.tan(fovy / 2)

    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 1.0 / tanHalfFovX
    P[1, 1] = 1.0 / tanHalfFovY
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _camera_matrices(camera, device: str):
    """
    Compute 3DGS-convention camera matrices from our Camera object.

    Returns: world_view_transform [4,4], full_proj_transform [4,4], camera_center [3]
    """
    c2w = camera.c2w.to(device)                      # [4, 4]
    w2c = torch.linalg.inv(c2w)                      # [4, 4]
    world_view_transform = w2c.T                      # transposed, 3DGS convention
    proj = _get_projection_matrix(camera.fovx, camera.fovy, device=device)
    full_proj = (world_view_transform.unsqueeze(0) @ proj.unsqueeze(0)).squeeze(0)
    camera_center = c2w[:3, 3]                        # world-space position
    return world_view_transform, full_proj, camera_center


# ---------------------------------------------------------------------------
# diff-gaussian-rasterization backend (3DGS / RigGS native)
# ---------------------------------------------------------------------------

def _render_dgr(gaussians: dict, camera, background: torch.Tensor,
                sh_degree: int) -> torch.Tensor:
    """
    Render using diff-gaussian-rasterization (3DGS CUDA rasterizer).
    Returns [3, H, W] float32 in [0, 1].
    """
    device = gaussians["xyz"].device

    means3D    = gaussians["xyz"]                              # [N, 3]
    shs        = gaussians["features"]                         # [N, K, 3]
    scales     = torch.exp(gaussians["scaling"])               # [N, 3] or [N, 1] if isotropic
    if scales.shape[-1] == 1:
        scales = scales.expand(-1, 3)                          # isotropic → broadcast
    rotations  = F.normalize(gaussians["rotation"], dim=-1)    # [N, 4]
    opacity    = torch.sigmoid(gaussians["opacity"])           # [N] or [N,1]
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(-1)                        # [N, 1]

    world_view_transform, full_proj, camera_center = _camera_matrices(camera, device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=math.tan(camera.fovx / 2),
        tanfovy=math.tan(camera.fovy / 2),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj,
        sh_degree=sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    screenspace_points = torch.zeros_like(means3D, requires_grad=False)

    out = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # diff-gaussian-rasterization returns (image, radii)
    # depth-diff-gaussian-rasterization may return (image, radii, depth) — handle both
    rendered_image = out[0] if isinstance(out, (tuple, list)) else out
    return rendered_image.clamp(0.0, 1.0)  # [3, H, W]


# ---------------------------------------------------------------------------
# gsplat fallback backend
# ---------------------------------------------------------------------------

def _render_gsplat(gaussians: dict, camera, background: torch.Tensor,
                    sh_degree: int) -> torch.Tensor:
    """Render using gsplat library. Returns [3, H, W] float32 in [0, 1]."""
    device = gaussians["xyz"].device

    means     = gaussians["xyz"]
    quats     = F.normalize(gaussians["rotation"], dim=-1)
    scales    = torch.exp(gaussians["scaling"])
    if scales.shape[-1] == 1:
        scales = scales.expand(-1, 3)                          # isotropic → broadcast
    opacities = torch.sigmoid(gaussians["opacity"])
    colors    = gaussians["features"]                          # [N, K, 3]

    viewmat = camera.world_to_cam.unsqueeze(0).to(device)      # [1, 4, 4]
    K       = camera.K.unsqueeze(0).to(device)                 # [1, 3, 3]

    renders, _, _ = _gsplat_rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmat, Ks=K,
        width=camera.width, height=camera.height,
        sh_degree=sh_degree,
        backgrounds=background.to(device),  # [C] — gsplat rasterize_to_pixels expects (C,)
    )
    return renders[0].permute(2, 0, 1).clamp(0.0, 1.0)        # [3, H, W]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_gaussians(gaussians: dict,
                      camera,
                      background: torch.Tensor = None,
                      sh_degree: int = None) -> torch.Tensor:
    """
    Render deformed Gaussians from a single camera viewpoint.

    Uses diff-gaussian-rasterization if available (preferred),
    falls back to gsplat otherwise.

    Args:
        gaussians:  dict from ply_loader / fk_lbs  (xyz, features, opacity, scaling, rotation, sh_degree)
        camera:     Camera object (camera_utils.Camera)
        background: [3] float tensor. Defaults to black.
        sh_degree:  override SH degree (default: from gaussians dict)

    Returns: [3, H, W] float32 in [0, 1]
    """
    device = gaussians["xyz"].device
    if background is None:
        background = torch.zeros(3, device=device)

    sh_deg = sh_degree if sh_degree is not None else gaussians["sh_degree"]

    if DGR_AVAILABLE:
        return _render_dgr(gaussians, camera, background, sh_deg)
    elif GSPLAT_AVAILABLE:
        return _render_gsplat(gaussians, camera, background, sh_deg)
    else:
        raise ImportError(
            "No rasterizer available. Install one of:\n"
            "  (recommended) pip install ./submodules/diff-gaussian-rasterization\n"
            "  (fallback)    pip install gsplat"
        )


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
    Render a full trajectory frame-by-frame.

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
        frames.append(render_gaussians(deformed, cam, background=background, sh_degree=sh_degree))

    return frames
