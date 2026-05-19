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


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ODE model on joint trajectories")
    p.add_argument("--checkpoint", required=True, help="Path to model_final.pt")
    p.add_argument("--theta_path", required=True, help="Path to theta.pt / theta.npz")
    p.add_argument("--output_dir", default=None, help="Directory to save results (default: checkpoint dir)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)

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

    # --- Evaluate ---
    results = evaluate(model, data, model_type=model_type)

    # Pretty-print
    print("\n=== Joint MAE ===")
    print(f"  Interp:  {results['interp']['mae_degrees']:.4f} deg")
    print(f"  Extrap:  {results['extrap']['mae_degrees']:.4f} deg")

    if "energy_extrap" in results:
        e = results["energy_extrap"]
        print("\n=== Energy Conservation (extrap) ===")
        print(f"  ΔH_mean: {e['delta_H_mean']:.6f}")
        print(f"  ΔH_max:  {e['delta_H_max']:.6f}")
        print(f"  H_std:   {e['H_std']:.6f}")

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_results.json")
    serialisable = {}
    for split, v in results.items():
        if isinstance(v, dict):
            serialisable[split] = {
                k2: float(v2) if hasattr(v2, "item") else
                    v2.tolist() if hasattr(v2, "tolist") else v2
                for k2, v2 in v.items()
                if k2 != "H_t" and k2 != "mae_per_joint"
            }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save per-joint MAE
    for split in ("interp", "extrap"):
        per_j = results[split]["mae_per_joint"]
        pj_path = os.path.join(output_dir, f"mae_per_joint_{split}.pt")
        torch.save(per_j, pj_path)

    # Optionally save energy curve
    if "energy_extrap" in results and results["energy_extrap"].get("H_t") is not None:
        torch.save(results["energy_extrap"]["H_t"],
                   os.path.join(output_dir, "H_extrap.pt"))


if __name__ == "__main__":
    main()
