# [REF: ODE-GS/train_extrapolation.py]
# [REF: SE3HamDL/examples/pendulum/train_pend_SO3.py]
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import yaml

from .models.neural_ode import JointNeuralODE
from .models.hamiltonian_ode import JointHamiltonianODE
from .data.riggs_loader import load_preextracted, prepare_data


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: dict, n_joints: int) -> nn.Module:
    """Instantiate the model specified in config."""
    rot_dim = config["data"]["rot_dim"]
    m = config["model"]

    if config["model_type"] == "neural_ode":
        return JointNeuralODE(
            n_joints=n_joints, rot_dim=rot_dim,
            latent_dim=m["latent_dim"],
            d_model=m["d_model"], nhead=m["nhead"],
            num_enc_layers=m["num_enc_layers"],
            ode_nhidden=m["ode_nhidden"], ode_layers=m["ode_layers"],
            decoder_nhidden=m["decoder_nhidden"],
            solver=config["ode"]["method"],
            rtol=config["ode"]["rtol"], atol=config["ode"]["atol"],
            kl_beta=m["kl_beta"],
        )
    elif config["model_type"] == "hamiltonian_ode":
        return JointHamiltonianODE(
            n_joints=n_joints, rot_dim=rot_dim,
            hidden_dim=m["hidden_dim"], n_layers=m["n_layers"],
            solver=config["ode"]["method"],
            rtol=config["ode"]["rtol"], atol=config["ode"]["atol"],
            use_adjoint=m["use_adjoint"],
        )
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")


def build_optimizer_and_scheduler(model: nn.Module, config: dict):
    tr = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"])

    sched_type = tr.get("scheduler", "cosine_annealing")
    if sched_type == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tr["epochs"], eta_min=tr.get("min_lr", 1e-6)
        )
    elif sched_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=100,
            min_lr=tr.get("min_lr", 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_neural_ode(model: JointNeuralODE, data: dict,
                     config: dict, output_dir: str) -> dict:
    """
    [REF: ODE-GS/train_extrapolation.py - train_epoch()]
    Train the Transformer-Latent ODE on joint rotation trajectories.
    """
    tr = config["training"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    N_j, rot_dim = data["N_j"], data["rot_dim"]
    history = {"loss": [], "L_recon": [], "L_kl": []}

    for epoch in range(tr["epochs"]):
        model.train()
        optimizer.zero_grad()

        q_pred, mu, logvar = model(
            obs_traj=data["theta_train"],      # [T_train, N_j*4]
            extrap_times=data["t_extrap"],
        )
        q_gt = data["theta_extrap"].reshape(-1, N_j, rot_dim)
        metrics = model.compute_loss(q_pred, q_gt, mu, logvar)

        metrics["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad_norm"])
        optimizer.step()

        sched_type = tr.get("scheduler", "cosine_annealing")
        if sched_type == "cosine_annealing":
            scheduler.step()
        elif sched_type == "reduce_on_plateau":
            scheduler.step(metrics["loss"].item())

        for k in history:
            history[k].append(metrics[k] if k == "loss" else metrics[k])

        if epoch % tr["log_every"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:5d}/{tr['epochs']}] "
                f"loss={metrics['loss']:.6f}  "
                f"recon={metrics['L_recon']:.6f}  "
                f"kl={metrics['L_kl']:.6f}  "
                f"lr={lr:.2e}"
            )

        if (epoch + 1) % tr["save_every"] == 0:
            _save_checkpoint(model, optimizer, scheduler, epoch, config, history, output_dir)

    return history


def train_hamiltonian_ode(model: JointHamiltonianODE, data: dict,
                           config: dict, output_dir: str) -> dict:
    """
    [REF: SE3HamDL/examples/pendulum/train_pend_SO3.py - train loop]
    """
    tr = config["training"]
    lambda_energy = tr.get("lambda_energy", 0.01)
    energy_subsample = tr.get("energy_subsample", 5)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    N_j, rot_dim = data["N_j"], data["rot_dim"]
    history = {"loss": [], "L_recon": [], "L_energy": [], "H_mean": []}

    q0 = data["theta_train"][0].reshape(N_j, rot_dim)
    p0 = data["dtheta_train"][0].reshape(N_j, rot_dim)
    q_gt = data["theta_train"].reshape(data["T_train"], N_j, rot_dim)

    for epoch in range(tr["epochs"]):
        model.train()
        optimizer.zero_grad()

        q_pred, p_pred = model(q0, p0, data["t_train"])
        metrics = model.compute_loss(
            q_pred, p_pred, q_gt,
            lambda_energy=lambda_energy,
            energy_subsample=energy_subsample,
        )

        metrics["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad_norm"])
        optimizer.step()

        sched_type = tr.get("scheduler", "cosine_annealing")
        if sched_type == "cosine_annealing":
            scheduler.step()
        elif sched_type == "reduce_on_plateau":
            scheduler.step(metrics["loss"].item())

        for k in history:
            history[k].append(metrics[k] if k == "loss" else metrics[k])

        if epoch % tr["log_every"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:5d}/{tr['epochs']}] "
                f"loss={metrics['loss']:.6f}  "
                f"recon={metrics['L_recon']:.6f}  "
                f"energy={metrics['L_energy']:.6f}  "
                f"H={metrics['H_mean']:.4f}  "
                f"lr={lr:.2e}"
            )

        if (epoch + 1) % tr["save_every"] == 0:
            _save_checkpoint(model, optimizer, scheduler, epoch, config, history, output_dir)

    return history


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, scheduler, epoch, config, history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ckpt_epoch{epoch + 1}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history,
    }, path)
    print(f"  → checkpoint saved: {path}")


def save_final(model, config, history, data_meta, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
        "data_meta": data_meta,
    }, path)
    print(f"Final model saved: {path}")

    hist_path = os.path.join(output_dir, "history.json")
    serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(hist_path, "w") as f:
        json.dump(serialisable, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Neural ODE or Hamiltonian ODE on joint trajectories")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--theta_path", default=None,
                   help="Path to pre-extracted theta.pt / theta.npz")
    p.add_argument("--model_path", default=None,
                   help="Path to RigGS checkpoint directory (requires RigGS env)")
    p.add_argument("--output_dir", default="output", help="Directory for checkpoints")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"

    # --- Load data ---
    if args.theta_path is not None:
        theta = load_preextracted(args.theta_path, device=device)
    elif args.model_path is not None:
        raise NotImplementedError(
            "Direct RigGS loading from train.py not yet implemented. "
            "Run scripts/extract_trajectory.py first to produce theta.pt."
        )
    else:
        raise ValueError("Provide either --theta_path or --model_path.")

    data = prepare_data(theta, time_split=config["data"]["time_split"])
    n_joints = data["N_j"]
    print(f"Loaded trajectory: T={data['T_total']}, N_j={n_joints}, "
          f"T_train={data['T_train']}, T_extrap={data['T_extrap']}")

    # --- Build model ---
    model = build_model(config, n_joints).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}  |  params: {total_params:,}")

    # --- Train ---
    t0 = time.time()
    if config["model_type"] == "neural_ode":
        history = train_neural_ode(model, data, config, args.output_dir)
    else:
        history = train_hamiltonian_ode(model, data, config, args.output_dir)

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed / 60:.1f} min")

    data_meta = {
        "T_total": data["T_total"], "T_train": data["T_train"],
        "T_extrap": data["T_extrap"], "N_j": n_joints,
        "rot_dim": data["rot_dim"], "time_split": config["data"]["time_split"],
    }
    save_final(model, config, history, data_meta, args.output_dir)


if __name__ == "__main__":
    main()
