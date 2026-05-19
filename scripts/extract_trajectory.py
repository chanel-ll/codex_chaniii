"""
Extract joint rotation trajectory from a trained RigGS checkpoint.
Must be run inside the RigGS conda environment.

Usage:
    python scripts/extract_trajectory.py \\
        --model_path /path/to/riggs_output \\
        --output_dir /path/to/save \\
        --device cuda

Outputs:
    <output_dir>/theta.pt     — float32 tensor [T, N_j, 4]
    <output_dir>/meta.json    — trajectory metadata
"""
# [REF: RigGS/train_rig.py - TrainRig.__init__ and deform_gaussians()]
import argparse
import json
import os
import sys

import torch

# Allow running as a top-level script without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from joint_ham_ode.data.riggs_loader import (
    load_riggs_checkpoint,
    extract_joint_trajectory,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to RigGS output directory (contains skeleton_tree.npz)")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save theta.pt and meta.json")
    p.add_argument("--device", default="cuda")
    # RigGS dataset/opt args — pass the same values used during training
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--hyper_dim", type=int, default=0)
    p.add_argument("--gs_with_motion_mask", action="store_true")
    p.add_argument("--use_isotropic_gs", action="store_true")
    p.add_argument("--is_blender", action="store_true")
    p.add_argument("--skeleton_weight_knn", type=int, default=4)
    p.add_argument("--fps", type=float, default=30.0,
                   help="Frame rate of the input video (for metadata)")
    return p.parse_args()


class _SimpleNamespace:
    pass


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build lightweight arg objects that mimic RigGS Namespace
    dataset_args = _SimpleNamespace()
    dataset_args.sh_degree = args.sh_degree
    dataset_args.hyper_dim = args.hyper_dim
    dataset_args.gs_with_motion_mask = args.gs_with_motion_mask
    dataset_args.use_isotropic_gs = args.use_isotropic_gs
    dataset_args.is_blender = args.is_blender

    opt_args = _SimpleNamespace()
    opt_args.skeleton_weight_knn = args.skeleton_weight_knn

    print(f"Loading RigGS checkpoint from: {args.model_path}")
    gaussians, skeleton, scene = load_riggs_checkpoint(
        args.model_path, dataset_args, opt_args
    )

    print("Extracting joint trajectory...")
    theta, d_nodes = extract_joint_trajectory(gaussians, skeleton, scene)

    T, N_j, rot_dim = theta.shape
    print(f"  theta shape: {list(theta.shape)}")
    print(f"  d_nodes shape: {list(d_nodes.shape)}")

    # Save
    theta_path = os.path.join(args.output_dir, "theta.pt")
    torch.save(theta.cpu().float(), theta_path)
    print(f"Saved theta → {theta_path}")

    d_nodes_path = os.path.join(args.output_dir, "d_nodes.pt")
    torch.save(d_nodes.cpu().float(), d_nodes_path)
    print(f"Saved d_nodes → {d_nodes_path}")

    meta = {
        "T": T,
        "N_j": N_j,
        "rot_dim": rot_dim,
        "fps": args.fps,
        "dt": 1.0 / T,
        "model_path": os.path.abspath(args.model_path),
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
