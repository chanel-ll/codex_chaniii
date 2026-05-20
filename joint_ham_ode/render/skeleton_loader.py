"""Load skeleton structure and LBS skinning weights from RigGS output."""

import os
import re
import glob
import numpy as np
import torch
import torch.nn.functional as F


def load_skeleton_tree(npz_path: str, device: str = "cpu") -> dict:
    """
    Load joint structure from skeleton_tree.npz.

    Returns dict:
        joints         [N_j, 3]   float32  rest-pose joint positions (world space)
        parents        [N_j]      int64    parent index per joint (root: parent==0)
    """
    data = np.load(npz_path)
    joints  = torch.from_numpy(data["nodes"]).float().to(device)
    parents = torch.from_numpy(data["parents"]).long().to(device)
    print(f"Skeleton: {joints.shape[0]} joints loaded from {npz_path}")
    return {"joints": joints, "parents": parents}


# ---------------------------------------------------------------------------
# Skinning weight MLP helpers
# ---------------------------------------------------------------------------

def _positional_encoding(xyz: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding used by NeRF-style MLPs.
    xyz: [N, 3]  →  [N, 3*(2*n_freqs + 1)]
    """
    parts = [xyz]
    for k in range(n_freqs):
        freq = 2.0 ** k
        parts.append(torch.sin(freq * xyz))
        parts.append(torch.cos(freq * xyz))
    return torch.cat(parts, dim=-1)


def _prepare_mlp_input(xyz: torch.Tensor, in_dim: int) -> torch.Tensor:
    """
    Match xyz to the expected MLP input dimension.

    RigGS commonly uses sinusoidal positional encoding before the MLP:
      raw xyz  → in_dim = 3
      L=4 PE   → in_dim = 3*(2*4+1) = 27
      L=6 PE   → in_dim = 3*(2*6+1) = 39
      L=10 PE  → in_dim = 3*(2*10+1) = 63
    """
    if in_dim == 3:
        return xyz

    # Try to match known positional encoding sizes
    for n_freqs in (4, 6, 8, 10, 12):
        expected = 3 * (2 * n_freqs + 1)
        if expected == in_dim:
            return _positional_encoding(xyz, n_freqs)

    # Fallback: pad or truncate
    if xyz.shape[-1] >= in_dim:
        return xyz[:, :in_dim]
    return F.pad(xyz, (0, in_dim - xyz.shape[-1]))




def _run_skinning_mlp(state: dict, canonical_xyz: torch.Tensor,
                       n_joints: int) -> torch.Tensor:
    """
    Run the skinning_weight_mlp to obtain per-Gaussian skinning weights.

    RigGS uses a NeRF-style MLP with skip connections: at certain layers the
    original positional-encoded input is concatenated back to the hidden state.
    We detect this automatically by comparing each layer's expected input dim
    against the actual current hidden dim.

    canonical_xyz: [N, 3]  canonical Gaussian positions
    Returns: [N, n_joints] float32 (sum-to-1 per Gaussian, softmax applied)
    """
    prefix = "skinning_weight_mlp"
    sub = {k[len(prefix) + 1:]: v for k, v in state.items()
           if k.startswith(prefix + ".")}

    if not sub:
        raise KeyError(
            "skinning_weight_mlp not found in state dict. "
            f"Available top-level prefixes: {_top_prefixes(state)}"
        )

    indices = sorted({
        int(m.group(1))
        for k in sub
        if (m := re.match(r"linear\.(\d+)\.weight", k))
    })

    in_dim = sub[f"linear.{indices[0]}.weight"].shape[1]
    x_in = _prepare_mlp_input(canonical_xyz.cpu(), in_dim)  # [N, in_dim]
    print(f"  skinning_weight_mlp: input_dim={in_dim}  "
          f"({'raw xyz' if in_dim == 3 else f'PE (L={(in_dim//3-1)//2})'})")

    h = x_in
    with torch.no_grad():
        for i in indices:
            w = sub[f"linear.{i}.weight"]
            b = sub[f"linear.{i}.bias"]
            # Skip connection: concat original input when dim mismatch
            if w.shape[1] != h.shape[-1]:
                h = torch.cat([h, x_in], dim=-1)
            h = F.relu(F.linear(h, w, b))

        w_out = sub["weight_predict.weight"]
        b_out = sub["weight_predict.bias"]
        if w_out.shape[1] != h.shape[-1]:
            h = torch.cat([h, x_in], dim=-1)
        logits = F.linear(h, w_out, b_out)           # [N, n_joints]

    if logits.shape[1] != n_joints:
        raise ValueError(
            f"MLP output dim {logits.shape[1]} != n_joints {n_joints}."
        )

    return torch.softmax(logits, dim=-1)             # [N, n_joints]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lbs_weights(skeleton_dir: str, n_joints: int,
                      canonical_xyz: torch.Tensor = None,
                      device: str = "cpu") -> dict:
    """
    Compute per-Gaussian skinning weights from the RigGS skeleton checkpoint.

    RigGS does NOT store skinning weights as a precomputed tensor.
    Instead, it stores a skinning_weight_mlp whose weights are in the state dict.
    This function reconstructs that MLP and runs a forward pass with the
    canonical Gaussian positions to produce [N_gauss, N_joints] weights.

    Args:
        skeleton_dir:  Path to <riggs_output>/skeleton/ directory
        n_joints:      Number of skeleton joints (must match MLP output dim)
        canonical_xyz: [N, 3] canonical Gaussian positions.
                       If None, attempts to load gs__xyz from the state dict.
        device:        Target device for output tensors

    Returns dict:
        lbs_weights   [N_gauss, N_j]  float32  per-Gaussian skinning weights (softmax)
        motion_mask   [N_gauss]       bool     all True (RigGS uses the MLP for all Gaussians)
    """
    # --- Find latest iteration directory ---
    iter_dirs = sorted(glob.glob(os.path.join(skeleton_dir, "iteration_*")))
    if not iter_dirs:
        raise FileNotFoundError(f"No iteration_* directories found in {skeleton_dir}")
    latest = iter_dirs[-1]

    # --- Load state dict ---
    ckpt_files = (glob.glob(os.path.join(latest, "*.pth")) +
                  glob.glob(os.path.join(latest, "*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth/.pt files found in {latest}")

    state = None
    for f in sorted(ckpt_files):
        try:
            state = torch.load(f, map_location="cpu")
            print(f"  Loaded state dict from {f}")
            break
        except Exception:
            continue
    if state is None:
        raise RuntimeError(f"Could not load any checkpoint from {latest}")

    # --- Canonical Gaussian positions (needed as MLP input) ---
    if canonical_xyz is None:
        # RigGS saves Gaussian params inside the skeleton state dict under gs__* keys
        for key in ("gs__xyz", "gs_xyz", "xyz"):
            if key in state:
                canonical_xyz = state[key].float()
                print(f"  Using canonical xyz from state dict key '{key}' "
                      f"({canonical_xyz.shape[0]} Gaussians)")
                break
        if canonical_xyz is None:
            raise ValueError(
                "canonical_xyz not provided and 'gs__xyz' not found in state dict. "
                "Pass gaussians['xyz'] explicitly to load_lbs_weights()."
            )

    # --- Compute skinning weights via MLP ---
    print(f"  Running skinning_weight_mlp for {canonical_xyz.shape[0]} Gaussians…")
    lbs_weights = _run_skinning_mlp(state, canonical_xyz, n_joints)

    # --- Motion mask (all Gaussians participate) ---
    motion_mask = torch.ones(lbs_weights.shape[0], dtype=torch.bool)

    print(f"  LBS weights: {lbs_weights.shape}")
    return {
        "lbs_weights": lbs_weights.to(device),
        "motion_mask": motion_mask.to(device),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _top_prefixes(d: dict) -> list:
    """Return unique top-level key prefixes (before first dot)."""
    return sorted({k.split(".")[0] for k in d})
