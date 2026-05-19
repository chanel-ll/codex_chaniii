"""
Rendering utilities.

Standalone path (no RigGS env required):
    Use joint_ham_ode.render.* pipeline — ply_loader, skeleton_loader, fk_lbs, gsplat_renderer.

Legacy RigGS path (requires RigGS env):
    Use apply_predicted_rotations() + render_frame() / batch_render_trajectory() which call into RigGS directly.
    These are kept for backward-compat but the standalone path is preferred.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from gaussian_renderer import render as _riggs_render
    RIGGS_RENDER_AVAILABLE = True
except ImportError:
    RIGGS_RENDER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Legacy RigGS path (requires RigGS env)
# ---------------------------------------------------------------------------

def apply_predicted_rotations(gaussians, skeleton, joint_rotations_t: torch.Tensor) -> dict:
    """
    [RigGS env] Inject predicted rotations into skeleton and run FK+LBS.
    joint_rotations_t: [N_j, 4]
    """
    joint_rotations_t = F.normalize(joint_rotations_t, dim=-1)
    with torch.no_grad():
        joint_transforms = skeleton.deform.fk(joint_rotations_t)
        d_values = skeleton.lbs(
            gaussians.get_xyz.detach(),
            joint_transforms,
            motion_mask=gaussians.motion_mask,
        )
    return d_values


def render_frame(gaussians, skeleton, theta_pred: torch.Tensor,
                 camera, pipe, background: torch.Tensor) -> dict:
    """[RigGS env] Render one frame. theta_pred: [N_j, 4]."""
    if not RIGGS_RENDER_AVAILABLE:
        raise ImportError("gaussian_renderer not found. Use standalone render pipeline instead.")
    d = apply_predicted_rotations(gaussians, skeleton, theta_pred)
    return _riggs_render(
        camera, gaussians, pipe, background,
        d["d_xyz"], d["d_rotation"], d["d_scaling"],
        d_opacity=d.get("d_opacity"), d_color=d.get("d_color"),
        d_rot_as_res=skeleton.d_rot_as_res,
    )


def batch_render_trajectory(gaussians, skeleton,
                             theta_traj: torch.Tensor,
                             cameras: list, pipe, background: torch.Tensor) -> list:
    """[RigGS env] Render a sequence of frames."""
    return [render_frame(gaussians, skeleton, theta_traj[t], cam, pipe, background)["render"].clamp(0, 1)
            for t, cam in enumerate(cameras)]


# ---------------------------------------------------------------------------
# Image / video I/O  (shared by both paths)
# ---------------------------------------------------------------------------

def save_frames(frames: list, save_dir: str, prefix: str = "frame") -> list:
    """Save [C, H, W] float tensors as PNG files. Returns paths."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        arr = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def frames_to_mp4(frames: list, save_path: str, fps: float = 10.0) -> str:
    """Write [C, H, W] tensors to MP4 (falls back to GIF if ffmpeg unavailable)."""
    try:
        import imageio
    except ImportError:
        raise ImportError("Run: pip install imageio[ffmpeg]")

    arrays = [
        (f.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        for f in frames
    ]
    try:
        writer = imageio.get_writer(save_path, fps=fps, codec="libx264",
                                    pixelformat="yuv420p",
                                    output_params=["-crf", "18"])
        for arr in arrays:
            writer.append_data(arr)
        writer.close()
    except Exception:
        gif_path = save_path.replace(".mp4", ".gif")
        imageio.mimsave(gif_path, arrays, fps=fps)
        print(f"  ffmpeg unavailable — saved GIF: {gif_path}")
        return gif_path
    return save_path


def save_comparison_grid(pred_frames: list, gt_frames: list,
                          save_dir: str, prefix: str = "cmp") -> list:
    """Save side-by-side pred | gt comparison images."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, (pred, gt) in enumerate(zip(pred_frames, gt_frames)):
        p = (pred.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        g = (gt.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        sep = np.ones((p.shape[0], 4, 3), dtype=np.uint8) * 255
        grid = np.concatenate([p, sep, g], axis=1)
        path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
        Image.fromarray(grid).save(path)
        paths.append(path)
    return paths
