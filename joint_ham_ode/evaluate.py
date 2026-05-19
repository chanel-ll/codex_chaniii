# [REF: ODE-GS/evaluate_extrapolation.py]
import argparse
import json
import os

import torch
import yaml

from .models.neural_ode import JointNeuralODE
from .models.hamiltonian_ode import JointHamiltonianODE
from .data.riggs_loader import load_preextracted, load_npz_full, prepare_data
from .train import build_model
from .utils.eval_utils import evaluate, compute_joint_mae, compute_energy_metrics


def load_trained_model(checkpoint_path: str, device: str = "cuda"):
    """Load a checkpoint saved by train.py and reconstruct the model."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    data_meta = ckpt.get("data_meta", {})
    n_joints = data_meta.get("N_j")
    if n_joints is None:
        raise KeyError("Checkpoint missing data_meta['N_j']. Re-train with latest train.py.")

    model = build_model(config, n_joints).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config, data_meta


def _load_riggs(model_path: str, theta_path: str, device: str):
    """Load RigGS gaussians + skeleton + cameras from a trained checkpoint."""
    from .data.riggs_loader import load_riggs_checkpoint

    try:
        import argparse as _ap
        from arguments import ModelParams, PipelineParams, OptimizationParams
        import sys

        # Build minimal arg parser that RigGS expects
        parser = _ap.ArgumentParser()
        lp = ModelParams(parser)
        pp = PipelineParams(parser)
        op = OptimizationParams(parser)
        args = parser.parse_args(["--model_path", model_path,
                                  "--source_path", model_path])
        dataset = lp.extract(args)
        pipe    = pp.extract(args)
        opt     = op.extract(args)
    except Exception as e:
        raise ImportError(
            f"Could not load RigGS argument parsers: {e}\n"
            "Make sure this is run inside the RigGS conda environment."
        )

    gaussians, skeleton, scene = load_riggs_checkpoint(model_path, dataset, opt)
    gaussians = gaussians.to(device)
    skeleton  = skeleton.to(device).eval()

    cameras_all = sorted(scene.getTrainCameras(), key=lambda c: c.fid)

    import torch as _t
    background = _t.zeros(3, device=device)

    return gaussians, skeleton, cameras_all, pipe, background


def _predict_trajectories(model, data, model_type, device):
    """Run forward pass and return (q_interp, q_extrap, p_interp, p_extrap)."""
    N_j   = data["N_j"]
    rot_dim = data["rot_dim"]
    T_train = data["T_train"]

    model.eval()
    with torch.no_grad():
        if model_type == "neural_ode":
            q_interp, _, _ = model(obs_traj=data["theta_train"],
                                   extrap_times=data["t_train"])
            q_extrap, _, _ = model(obs_traj=data["theta_train"],
                                   extrap_times=data["t_extrap"])
            p_interp = p_extrap = None

        elif model_type == "hamiltonian_ode":
            q0 = data["theta_train"][0].reshape(N_j, rot_dim)
            p0 = data["dtheta_train"][0].reshape(N_j, rot_dim)
            q_interp, p_interp = model(q0, p0, data["t_train"])
            q_extrap, p_extrap = model(q0, p0, data["t_extrap"])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    return q_interp, q_extrap, p_interp, p_extrap


def _run_rendering(gaussians, skeleton, pipe, background,
                   q_interp, q_extrap,
                   cameras_all, T_train, output_dir, fps):
    """Render frames, compute PSNR/LPIPS, save PNGs + comparison grid + MP4."""
    from .utils.lbs_utils import (batch_render_trajectory,
                                   save_frames, save_comparison_grid, frames_to_mp4)
    from .utils.eval_utils import compute_rendering_metrics

    cameras_interp = cameras_all[:T_train]
    cameras_extrap = cameras_all[T_train:]

    render_results = {}

    for tag, q_traj, cameras in [("interp", q_interp, cameras_interp),
                                   ("extrap", q_extrap, cameras_extrap)]:
        if not cameras:
            continue

        print(f"  Rendering {tag} ({len(cameras)} frames)…")
        pred_frames = batch_render_trajectory(
            gaussians, skeleton, q_traj, cameras, pipe, background
        )
        gt_frames = [cam.original_image.to(q_traj.device).clamp(0, 1)
                     for cam in cameras]

        # Metrics
        metrics = compute_rendering_metrics(pred_frames, gt_frames)
        render_results[tag] = metrics
        print(f"    PSNR={metrics['psnr']:.2f} dB  "
              f"LPIPS={metrics['lpips']:.4f}" if metrics["lpips"] is not None
              else f"    PSNR={metrics['psnr']:.2f} dB")

        # Save individual frames
        frame_dir = os.path.join(output_dir, f"render_{tag}", "pred")
        save_frames(pred_frames, frame_dir, prefix="pred")

        gt_dir = os.path.join(output_dir, f"render_{tag}", "gt")
        save_frames(gt_frames, gt_dir, prefix="gt")

        # Save side-by-side comparison grid
        cmp_dir = os.path.join(output_dir, f"render_{tag}", "comparison")
        save_comparison_grid(pred_frames, gt_frames, cmp_dir, prefix="cmp")
        print(f"    Frames saved → {os.path.join(output_dir, f'render_{tag}')}/")

        # Save MP4 videos
        pred_mp4 = os.path.join(output_dir, f"render_{tag}", f"pred_{tag}.mp4")
        frames_to_mp4(pred_frames, pred_mp4, fps=fps)
        print(f"    Video saved  → {pred_mp4}")

        # comparison video: pred | separator | gt  (concat along width axis)
        cmp_mp4 = os.path.join(output_dir, f"render_{tag}", f"comparison_{tag}.mp4")
        sep_width = 4
        cmp_frames = []
        for p, g in zip(pred_frames, gt_frames):
            # p, g: [C, H, W]
            sep = torch.ones(p.shape[0], p.shape[1], sep_width, device=p.device)
            cmp_frames.append(torch.cat([p, sep, g], dim=2))
        frames_to_mp4(cmp_frames, cmp_mp4, fps=fps)
        print(f"    Comparison   → {cmp_mp4}")

    return render_results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ODE model on joint trajectories")
    p.add_argument("--checkpoint",  required=True, help="Path to model_final.pt")
    p.add_argument("--theta_path",  required=True, help="Path to joint_trajectory.npz / .pt / .npy")
    p.add_argument("--output_dir",  default=None,  help="Directory to save results (default: checkpoint dir)")
    p.add_argument("--device",      default="cuda")
    # Rendering options
    p.add_argument("--riggs_model_path", default=None,
                   help="RigGS output directory for rendering evaluation. "
                        "If omitted, only Joint MAE is computed.")
    p.add_argument("--render_fps",  type=float, default=10.0,
                   help="Frame rate for output MP4 videos (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    model, config, data_meta = load_trained_model(args.checkpoint, device)
    model_type = config["model_type"]
    print(f"Loaded {model_type} from {args.checkpoint}")

    # --- Load data ---
    timestamps = None
    if args.theta_path.endswith(".npz"):
        npz_data = load_npz_full(args.theta_path, device=device)
        theta = npz_data["theta"]
        timestamps = npz_data.get("timestamps")
    else:
        theta = load_preextracted(args.theta_path, device=device)
    data = prepare_data(theta, time_split=config["data"]["time_split"],
                        timestamps=timestamps)
    N_j, rot_dim = data["N_j"], data["rot_dim"]
    print(f"T_total={data['T_total']}  T_train={data['T_train']}  T_extrap={data['T_extrap']}")

    # --- Predict trajectories ---
    q_interp, q_extrap, p_interp, p_extrap = _predict_trajectories(
        model, data, model_type, device
    )

    # --- Joint MAE ---
    T_train = data["T_train"]
    q_gt_interp = data["theta_train"].reshape(T_train, N_j, rot_dim)
    q_gt_extrap = data["theta_extrap"].reshape(-1, N_j, rot_dim)

    results = {
        "interp": compute_joint_mae(q_interp, q_gt_interp),
        "extrap": compute_joint_mae(q_extrap, q_gt_extrap),
    }

    print("\n=== Joint MAE ===")
    print(f"  Interp:  {results['interp']['mae_degrees']:.4f} deg")
    print(f"  Extrap:  {results['extrap']['mae_degrees']:.4f} deg")

    # --- Energy metrics (Hamiltonian only) ---
    if model_type == "hamiltonian_ode" and p_extrap is not None:
        results["energy_interp"] = compute_energy_metrics(model, q_interp, p_interp)
        results["energy_extrap"] = compute_energy_metrics(model, q_extrap, p_extrap)
        e = results["energy_extrap"]
        print("\n=== Energy Conservation (extrap) ===")
        print(f"  ΔH_mean: {e['delta_H_mean']:.6f}")
        print(f"  ΔH_max:  {e['delta_H_max']:.6f}")
        print(f"  H_std:   {e['H_std']:.6f}")

    # --- Rendering evaluation ---
    if args.riggs_model_path is not None:
        print("\n=== Rendering Evaluation ===")
        try:
            gaussians, skeleton, cameras_all, pipe, background = _load_riggs(
                args.riggs_model_path, args.theta_path, device
            )
            render_results = _run_rendering(
                gaussians, skeleton, pipe, background,
                q_interp, q_extrap,
                cameras_all, T_train, output_dir, args.render_fps
            )
            for tag, m in render_results.items():
                results[f"render_{tag}"] = m
                print(f"  [{tag}]  PSNR={m['psnr']:.2f} dB"
                      + (f"  LPIPS={m['lpips']:.4f}" if m["lpips"] is not None else ""))
        except ImportError as exc:
            print(f"  Skipping rendering: {exc}")
    else:
        print("\n(Rendering evaluation skipped — pass --riggs_model_path to enable)")

    # --- Save JSON results ---
    out_path = os.path.join(output_dir, "eval_results.json")
    serialisable = {}
    for key, v in results.items():
        if isinstance(v, dict):
            serialisable[key] = {
                k2: float(v2) if hasattr(v2, "item") else
                    v2.tolist() if hasattr(v2, "tolist") else v2
                for k2, v2 in v.items()
                if k2 not in ("H_t", "mae_per_joint")
            }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # --- Save per-joint MAE tensors ---
    for split in ("interp", "extrap"):
        pj_path = os.path.join(output_dir, f"mae_per_joint_{split}.pt")
        torch.save(results[split]["mae_per_joint"], pj_path)

    # --- Save energy curve ---
    if "energy_extrap" in results and results["energy_extrap"].get("H_t") is not None:
        torch.save(results["energy_extrap"]["H_t"],
                   os.path.join(output_dir, "H_extrap.pt"))


if __name__ == "__main__":
    main()
