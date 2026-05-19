# [REF: RigGS/train_rig.py - render_and_cal_loss()]
# [REF: RigGS/scene/skeleton_model.py - SkeletonModel.step()]
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from gaussian_renderer import render
    RIGGS_RENDER_AVAILABLE = True
except ImportError:
    RIGGS_RENDER_AVAILABLE = False


def apply_predicted_rotations(gaussians, skeleton, joint_rotations_t: torch.Tensor) -> dict:
    """
    Inject predicted joint rotations [N_j, 4] into the skeleton and run
    Forward Kinematics + LBS to obtain Gaussian deformation fields.

    Method B: Extract the FK and LBS sub-steps from skeleton.step() so that
    we can provide our own local_rotation instead of computing it from time_input.

    ⚠️  The exact implementation depends on RigGS/scene/skeleton_model.py.
        The function assumes SkeletonModel exposes:
          skeleton.deform.fk(local_rotation)  →  joint world transforms
          skeleton.lbs(xyz, joint_transforms, motion_mask)  →  d_values dict
        Adjust to the actual API after inspecting skeleton_model.py.

    Args:
        gaussians:          GaussianModel with .get_xyz and .motion_mask
        skeleton:           SkeletonModel (eval mode)
        joint_rotations_t:  [N_j, 4] predicted quaternion for one frame

    Returns:
        d_values dict with keys: d_xyz, d_rotation, d_scaling, d_opacity, d_color
    """
    joint_rotations_t = F.normalize(joint_rotations_t, dim=-1)

    with torch.no_grad():
        # --- Method B: bypass the time-based rotation lookup in skeleton.deform ---
        # Step 1: FK — compute joint world transforms from predicted local rotations
        joint_transforms = skeleton.deform.fk(joint_rotations_t)

        # Step 2: LBS — apply skinning to canonical Gaussian positions
        d_values = skeleton.lbs(
            gaussians.get_xyz.detach(),
            joint_transforms,
            motion_mask=gaussians.motion_mask,
        )

    return d_values


def render_frame(gaussians, skeleton, theta_pred: torch.Tensor,
                 camera, pipe, background: torch.Tensor) -> dict:
    """
    Predict joint rotations → LBS → render one frame.
    [REF: RigGS/train_rig.py - render_and_cal_loss()]

    theta_pred: [N_j, 4]
    """
    if not RIGGS_RENDER_AVAILABLE:
        raise ImportError(
            "gaussian_renderer not found. Ensure RigGS is on the Python path."
        )

    d_values = apply_predicted_rotations(gaussians, skeleton, theta_pred)

    render_pkg = render(
        camera, gaussians, pipe, background,
        d_values["d_xyz"],
        d_values["d_rotation"],
        d_values["d_scaling"],
        d_opacity=d_values.get("d_opacity"),
        d_color=d_values.get("d_color"),
        d_rot_as_res=skeleton.d_rot_as_res,
    )
    return render_pkg


def batch_render_trajectory(gaussians, skeleton,
                             theta_traj: torch.Tensor,
                             cameras: list, pipe, background: torch.Tensor,
                             ) -> list:
    """
    Render a sequence of frames from predicted joint trajectories.

    theta_traj: [T, N_j, 4]
    cameras:    list of viewpoint cameras (same length as T)

    Returns list of rendered image tensors [C, H, W] in [0, 1].
    """
    rendered = []
    for t_idx, cam in enumerate(cameras):
        pkg = render_frame(gaussians, skeleton, theta_traj[t_idx], cam, pipe, background)
        rendered.append(pkg["render"].clamp(0.0, 1.0))
    return rendered


# ---------------------------------------------------------------------------
# Image / video I/O
# ---------------------------------------------------------------------------

def save_frames(frames: list, save_dir: str, prefix: str = "frame") -> list:
    """
    Save a list of [C, H, W] float tensors (values 0–1) as PNG files.

    Returns list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        arr = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def frames_to_mp4(frames: list, save_path: str, fps: float = 10.0) -> str:
    """
    Write a list of [C, H, W] float tensors (values 0–1) to an MP4 file.

    Uses imageio with the ffmpeg backend (pip install imageio[ffmpeg]).
    Falls back to writing a GIF if ffmpeg is unavailable.

    Returns the path of the saved file.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required for video export. "
                          "Run: pip install imageio[ffmpeg]")

    arrays = [
        (f.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        for f in frames
    ]

    # Try MP4 via ffmpeg; fall back to GIF
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
        print(f"  ffmpeg unavailable — saved GIF instead: {gif_path}")
        return gif_path

    return save_path


def save_comparison_grid(pred_frames: list, gt_frames: list,
                          save_dir: str, prefix: str = "cmp") -> list:
    """
    Save side-by-side pred | gt comparison images.

    pred_frames, gt_frames: list of [C, H, W] float tensors.
    Returns list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, (pred, gt) in enumerate(zip(pred_frames, gt_frames)):
        p = (pred.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        g = (gt.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        # Add a 4-px white separator
        sep = np.ones((p.shape[0], 4, 3), dtype=np.uint8) * 255
        grid = np.concatenate([p, sep, g], axis=1)
        path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
        Image.fromarray(grid).save(path)
        paths.append(path)
    return paths
