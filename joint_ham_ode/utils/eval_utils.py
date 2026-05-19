# [REF: ODE-GS/evaluate_extrapolation.py]
# [REF: RigGS/metrics.py]
import numpy as np
import torch
import torch.nn.functional as F

try:
    import lpips as _lpips_lib
    _lpips_fn = None

    def _get_lpips():
        global _lpips_fn
        if _lpips_fn is None:
            _lpips_fn = _lpips_lib.LPIPS(net="alex").cuda()
        return _lpips_fn

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Joint-level metrics
# ---------------------------------------------------------------------------

def compute_joint_mae(theta_pred: torch.Tensor,
                      theta_gt: torch.Tensor) -> dict:
    """
    Geodesic quaternion distance in degrees.
    Handles the double-cover ambiguity: q ≡ -q.

    theta_pred, theta_gt: [T, N_j, 4]
    """
    dot = (theta_pred * theta_gt).sum(dim=-1).abs().clamp(-1.0, 1.0)  # [T, N_j]
    angle_rad = 2.0 * torch.acos(dot)                                   # [T, N_j]
    angle_deg = torch.rad2deg(angle_rad)

    return {
        "mae_degrees":   angle_deg.mean().item(),
        "mae_per_joint": angle_deg.mean(dim=0).cpu(),   # [N_j]
    }


# ---------------------------------------------------------------------------
# Rendering metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """pred, gt: [C, H, W] or [B, C, H, W], values in [0, 1]."""
    mse = F.mse_loss(pred, gt)
    if mse.item() == 0.0:
        return float("inf")
    return (-10.0 * torch.log10(mse)).item()


def compute_rendering_metrics(pred_frames: list,
                               gt_frames: list) -> dict:
    """
    pred_frames, gt_frames: list of [C, H, W] tensors in [0, 1].
    [REF: ODE-GS/evaluate_extrapolation.py - metrics computation]
    """
    psnrs, lpips_vals = [], []

    lpips_fn = _get_lpips() if LPIPS_AVAILABLE else None

    for pred, gt in zip(pred_frames, gt_frames):
        psnrs.append(compute_psnr(pred, gt))

        if lpips_fn is not None:
            # LPIPS expects [B, C, H, W] in [-1, 1]
            p = (pred.unsqueeze(0) * 2 - 1).clamp(-1, 1)
            g = (gt.unsqueeze(0) * 2 - 1).clamp(-1, 1)
            with torch.no_grad():
                lp = lpips_fn(p, g).item()
            lpips_vals.append(lp)

    result = {"psnr": float(np.mean(psnrs))}
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    else:
        result["lpips"] = None
    return result


# ---------------------------------------------------------------------------
# Hamiltonian energy metrics
# ---------------------------------------------------------------------------

def compute_energy_metrics(model, q_traj: torch.Tensor,
                            p_traj: torch.Tensor) -> dict:
    """
    q_traj, p_traj: [T, N_j, 4]
    [REF: SE3HamDL/examples/pendulum/rollout_pend_SO3.py - H variance check]
    """
    with torch.no_grad():
        H_t = model.compute_energy(q_traj, p_traj)  # [T]

    delta_H = (H_t - H_t[0]).abs()

    return {
        "H_mean":       H_t.mean().item(),
        "H_std":        H_t.std().item(),
        "delta_H_mean": delta_H.mean().item(),
        "delta_H_max":  delta_H.max().item(),
        "H_t":          H_t.cpu(),
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(model, data: dict, model_type: str = "hamiltonian",
             skeleton=None, gaussians=None, cameras_interp=None,
             cameras_extrap=None, pipe=None, background=None) -> dict:
    """
    Run full evaluation: joint MAE on interp and extrap windows,
    optional rendering metrics, optional energy metrics.
    """
    from ..data.riggs_loader import prepare_data

    device = data["theta_train"].device
    N_j = data["N_j"]
    rot_dim = data["rot_dim"]
    T_train = data["T_train"]

    results = {}
    model.eval()

    with torch.no_grad():
        if model_type == "neural_ode":
            # --- interpolation: re-integrate over training window ---
            q_interp, _, _ = model(
                obs_traj=data["theta_train"],
                extrap_times=data["t_train"],
            )
            q_extrap, _, _ = model(
                obs_traj=data["theta_train"],
                extrap_times=data["t_extrap"],
            )
            q_gt_interp = data["theta_train"].reshape(T_train, N_j, rot_dim)
            q_gt_extrap = data["theta_extrap"].reshape(-1, N_j, rot_dim)

        elif model_type == "hamiltonian":
            q0 = data["theta_train"][0].reshape(N_j, rot_dim)
            p0 = data["dtheta_train"][0].reshape(N_j, rot_dim)

            q_interp, p_interp = model(q0, p0, data["t_train"])
            q_extrap, p_extrap = model(q0, p0, data["t_extrap"])

            q_gt_interp = data["theta_train"].reshape(T_train, N_j, rot_dim)
            q_gt_extrap = data["theta_extrap"].reshape(-1, N_j, rot_dim)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Joint MAE
    results["interp"] = compute_joint_mae(q_interp, q_gt_interp)
    results["extrap"] = compute_joint_mae(q_extrap, q_gt_extrap)

    # Energy metrics (Hamiltonian only)
    if model_type == "hamiltonian":
        results["energy_interp"] = compute_energy_metrics(model, q_interp, p_interp)
        results["energy_extrap"] = compute_energy_metrics(model, q_extrap, p_extrap)

    # Rendering metrics (optional)
    if skeleton is not None and gaussians is not None:
        from ..utils.lbs_utils import batch_render_trajectory
        if cameras_extrap:
            pred_frames = batch_render_trajectory(
                gaussians, skeleton, q_extrap, cameras_extrap, pipe, background
            )
            gt_frames = [cam.original_image.cuda() for cam in cameras_extrap]
            results["render_extrap"] = compute_rendering_metrics(pred_frames, gt_frames)

        if cameras_interp:
            pred_frames = batch_render_trajectory(
                gaussians, skeleton, q_interp, cameras_interp, pipe, background
            )
            gt_frames = [cam.original_image.cuda() for cam in cameras_interp]
            results["render_interp"] = compute_rendering_metrics(pred_frames, gt_frames)

    return results
