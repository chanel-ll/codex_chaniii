# [REF: ODE-GS/evaluate_extrapolation.py]
import argparse
import glob
import json
import os

import torch

from .data.riggs_loader import load_preextracted, load_npz_full, prepare_data
from .train import build_model
from .utils.eval_utils import compute_joint_mae, compute_energy_metrics, compute_rendering_metrics
from .utils.lbs_utils import save_frames, save_comparison_grid, frames_to_mp4


def load_trained_model(checkpoint_path: str, device: str = "cuda"):
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


def _predict_trajectories(model, data, model_type, device):
    N_j, rot_dim = data["N_j"], data["rot_dim"]
    model.eval()
    with torch.no_grad():
        if model_type == "neural_ode":
            q_interp, _, _ = model(obs_traj=data["theta_train"], extrap_times=data["t_train"])
            q_extrap, _, _ = model(obs_traj=data["theta_train"], extrap_times=data["t_extrap"])
            p_interp = p_extrap = None
        elif model_type == "hamiltonian_ode":
            q0 = data["theta_train"][0].reshape(N_j, rot_dim)
            p0 = data["dtheta_train"][0].reshape(N_j, rot_dim)
            # Integrate over the FULL time span [t_train[0] … t_extrap[-1]] from the
            # same initial conditions, then slice.  Using t_extrap directly would only
            # integrate for (t_extrap[-1]-t_extrap[0]) ≈ 0.2 time units instead of
            # the required ~1.0 units, giving completely wrong extrap predictions.
            T_train = data["T_train"]
            t_full = torch.cat([data["t_train"], data["t_extrap"]])
            q_full, p_full = model(q0, p0, t_full)
            q_interp, p_interp = q_full[:T_train], p_full[:T_train]
            q_extrap,  p_extrap  = q_full[T_train:], p_full[T_train:]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    return q_interp, q_extrap, p_interp, p_extrap


def _load_standalone_render_assets(args, device):
    """
    Load Gaussian PLY, skeleton, skinning weights, and cameras
    from RigGS output — no RigGS conda env required.

    Required args:
        --riggs_model_path : RigGS output root (skeleton_tree.npz, point_cloud/, skeleton/)
        --dataset_path     : D-NeRF dataset root (transforms_train.json + images)
        --image_size       : image resolution (default 800)
    """
    from .render.ply_loader import load_3dgs_ply
    from .render.skeleton_loader import load_skeleton_tree, load_lbs_weights
    from .render.camera_utils import load_cameras_from_transforms

    riggs_dir = args.riggs_model_path

    # point_cloud.ply — use latest iteration
    ply_candidates = sorted(glob.glob(
        os.path.join(riggs_dir, "point_cloud", "iteration_*", "point_cloud.ply")
    ))
    if not ply_candidates:
        raise FileNotFoundError(f"No point_cloud.ply found under {riggs_dir}/point_cloud/")
    gaussians = load_3dgs_ply(ply_candidates[-1], device=device)

    # skeleton_tree.npz — provides parents and template joint count
    skel = load_skeleton_tree(os.path.join(riggs_dir, "skeleton_tree.npz"), device=device)

    # skinning weights from skeleton/iteration_XXXX/
    # Also extracts trained joint positions and passes motion_mask from PLY fea_* attributes
    lbs = load_lbs_weights(
        os.path.join(riggs_dir, "skeleton"),
        n_joints=skel["joints"].shape[0],
        canonical_xyz=gaussians["xyz"].cpu(),
        motion_mask_from_ply=gaussians.get("motion_mask"),
        device=device,
    )

    # Use trained joint positions if available (more accurate than template after training)
    if lbs.get("trained_joints") is not None:
        skel["joints"] = lbs["trained_joints"]
        print(f"  Using trained joint positions from skeleton.pth")

    # cameras from transforms_train.json
    cameras_all = None
    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        json_path = os.path.join(dataset_path, "transforms_train.json")
        if os.path.exists(json_path):
            img_size = getattr(args, "image_size", 800)
            cameras_all = load_cameras_from_transforms(
                json_path,
                image_root=dataset_path,
                width=img_size, height=img_size,
                device=device,
            )
            print(f"  {len(cameras_all)} cameras loaded from {json_path}")
        else:
            print(f"  Warning: {json_path} not found — GT images unavailable")

    bg_val = getattr(args, "background", 0.0)
    background = torch.ones(3, device=device) * float(bg_val)

    return gaussians, skel, lbs, cameras_all, background


def _run_rendering(gaussians, skel, lbs, cameras_all, background,
                    q_interp, q_extrap, T_train, output_dir, fps, device):
    from .render.gaussian_renderer import render_trajectory
    from .render.camera_utils import load_gt_image

    results = {}
    for tag, q_traj in [("interp", q_interp), ("extrap", q_extrap)]:
        if cameras_all is None:
            print(f"  Skipping {tag} rendering (no cameras)")
            continue

        cameras = cameras_all[:T_train] if tag == "interp" else cameras_all[T_train:]
        if not cameras:
            continue

        T = q_traj.shape[0]
        if len(cameras) != T:
            idxs = [int(i * len(cameras) / T) for i in range(T)]
            cameras = [cameras[i] for i in idxs]

        print(f"  Rendering {tag} ({len(cameras)} frames)…")
        pred_frames = render_trajectory(
            gaussians, q_traj,
            skel["joints"], skel["parents"],
            lbs["lbs_weights"], cameras,
            motion_mask=lbs["motion_mask"],
            background=background,
        )

        gt_frames = [load_gt_image(cam, device=device) for cam in cameras]
        gt_frames = [g for g in gt_frames if g is not None]
        has_gt = len(gt_frames) == len(pred_frames)

        if has_gt:
            metrics = compute_rendering_metrics(pred_frames, gt_frames)
            results[tag] = metrics
            lbl = f"PSNR={metrics['psnr']:.2f} dB"
            if metrics["lpips"] is not None:
                lbl += f"  LPIPS={metrics['lpips']:.4f}"
            print(f"    {lbl}")
        else:
            results[tag] = {"psnr": None, "lpips": None}
            print(f"    GT not available — metrics skipped")

        frame_dir = os.path.join(output_dir, f"render_{tag}")
        save_frames(pred_frames, os.path.join(frame_dir, "pred"), prefix="pred")
        if has_gt:
            save_frames(gt_frames, os.path.join(frame_dir, "gt"), prefix="gt")
            save_comparison_grid(pred_frames, gt_frames,
                                  os.path.join(frame_dir, "comparison"), prefix="cmp")

        pred_mp4 = os.path.join(frame_dir, f"pred_{tag}.mp4")
        frames_to_mp4(pred_frames, pred_mp4, fps=fps)
        print(f"    Video → {pred_mp4}")

        if has_gt:
            sep = 4
            cmp_frames = [
                torch.cat([p, torch.ones(p.shape[0], p.shape[1], sep, device=p.device), g], dim=2)
                for p, g in zip(pred_frames, gt_frames)
            ]
            frames_to_mp4(cmp_frames, os.path.join(frame_dir, f"comparison_{tag}.mp4"), fps=fps)

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ODE model on joint trajectories")
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--theta_path",       required=True)
    p.add_argument("--output_dir",       default=None)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--riggs_model_path", default=None,
                   help="RigGS output root (point_cloud/, skeleton/, skeleton_tree.npz)")
    p.add_argument("--dataset_path",     default=None,
                   help="D-NeRF dataset root (transforms_train.json + images)")
    p.add_argument("--image_size",       type=int,   default=800)
    p.add_argument("--background",       type=float, default=0.0,
                   help="Background brightness 0.0=black 1.0=white")
    p.add_argument("--render_fps",       type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    model, config, data_meta = load_trained_model(args.checkpoint, device)
    model_type = config["model_type"]
    print(f"Loaded {model_type} from {args.checkpoint}")

    timestamps = None
    if args.theta_path.endswith(".npz"):
        npz_data = load_npz_full(args.theta_path, device=device)
        theta = npz_data["theta"]
        timestamps = npz_data.get("timestamps")
    else:
        theta = load_preextracted(args.theta_path, device=device)
    data = prepare_data(theta, time_split=config["data"]["time_split"], timestamps=timestamps)
    N_j, rot_dim, T_train = data["N_j"], data["rot_dim"], data["T_train"]
    print(f"T_total={data['T_total']}  T_train={T_train}  T_extrap={data['T_extrap']}")

    q_interp, q_extrap, p_interp, p_extrap = _predict_trajectories(model, data, model_type, device)

    q_gt_interp = data["theta_train"].reshape(T_train, N_j, rot_dim)
    q_gt_extrap = data["theta_extrap"].reshape(-1, N_j, rot_dim)
    results = {
        "interp": compute_joint_mae(q_interp, q_gt_interp),
        "extrap": compute_joint_mae(q_extrap, q_gt_extrap),
    }
    print("\n=== Joint MAE ===")
    print(f"  Interp:  {results['interp']['mae_degrees']:.4f} deg")
    print(f"  Extrap:  {results['extrap']['mae_degrees']:.4f} deg")

    if model_type == "hamiltonian_ode" and p_extrap is not None:
        results["energy_interp"] = compute_energy_metrics(model, q_interp, p_interp)
        results["energy_extrap"] = compute_energy_metrics(model, q_extrap, p_extrap)
        e = results["energy_extrap"]
        print("\n=== Energy Conservation (extrap) ===")
        print(f"  ΔH_mean: {e['delta_H_mean']:.6f}  ΔH_max: {e['delta_H_max']:.6f}")

    if args.riggs_model_path is not None:
        print("\n=== Rendering Evaluation (standalone) ===")
        try:
            gaussians, skel, lbs, cameras_all, background = _load_standalone_render_assets(
                args, device
            )
            render_results = _run_rendering(
                gaussians, skel, lbs, cameras_all, background,
                q_interp, q_extrap, T_train, output_dir, args.render_fps, device,
            )
            for tag, m in render_results.items():
                results[f"render_{tag}"] = m
        except (ImportError, FileNotFoundError) as exc:
            print(f"  Skipped: {exc}")
    else:
        print("\n(Rendering skipped — pass --riggs_model_path to enable)")

    # Save JSON
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
    print(f"\nResults → {out_path}")

    for split in ("interp", "extrap"):
        torch.save(results[split]["mae_per_joint"],
                   os.path.join(output_dir, f"mae_per_joint_{split}.pt"))

    if "energy_extrap" in results and results["energy_extrap"].get("H_t") is not None:
        torch.save(results["energy_extrap"]["H_t"],
                   os.path.join(output_dir, "H_extrap.pt"))


if __name__ == "__main__":
    main()
