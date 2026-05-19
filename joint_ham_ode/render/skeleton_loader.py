"""Load skeleton structure and LBS skinning weights from RigGS output."""

import os
import glob
import numpy as np
import torch


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


def load_lbs_weights(skeleton_dir: str, n_joints: int, device: str = "cpu") -> dict:
    """
    Load per-Gaussian skinning weights from the skeleton output directory.

    RigGS saves skeleton model state dicts under skeleton/iteration_XXXX/.
    This function:
      1. Finds the latest iteration directory.
      2. Loads the state dict (.pth / .pt file).
      3. Extracts the weight tensor by trying common key names.

    Returns dict:
        lbs_weights    [N_gauss, N_j]   float32   (sum-to-1 per Gaussian, after softmax/norm)
        motion_mask    [N_gauss]        bool      True = participates in deformation
                                                  (all-True if not found in state dict)
    """
    # --- Find latest iteration ---
    iter_dirs = sorted(glob.glob(os.path.join(skeleton_dir, "iteration_*")))
    if not iter_dirs:
        raise FileNotFoundError(f"No iteration_* directories found in {skeleton_dir}")
    latest = iter_dirs[-1]

    # --- Find state dict file ---
    ckpt_files = glob.glob(os.path.join(latest, "*.pth")) + \
                 glob.glob(os.path.join(latest, "*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth/.pt files found in {latest}")

    # Try each file until we find one with weight keys
    state = None
    for f in ckpt_files:
        try:
            state = torch.load(f, map_location="cpu")
            break
        except Exception:
            continue
    if state is None:
        raise RuntimeError(f"Could not load any checkpoint from {latest}")

    # --- Extract LBS weights ---
    WEIGHT_KEYS = [
        "lbs_weights", "skinning_weights", "skin_weights",
        "W", "blend_weights", "weights",
    ]
    weights_raw = None
    for key in WEIGHT_KEYS:
        if key in state:
            weights_raw = state[key]
            break
    # Also search nested dicts (e.g. state["model"] or state["state_dict"])
    if weights_raw is None:
        for top_val in state.values():
            if isinstance(top_val, dict):
                for key in WEIGHT_KEYS:
                    if key in top_val:
                        weights_raw = top_val[key]
                        break
            if weights_raw is not None:
                break

    if weights_raw is None:
        raise KeyError(
            f"Could not find skinning weight tensor in {latest}.\n"
            f"Available keys: {_flat_keys(state)}\n"
            f"Please check the RigGS skeleton model and set the correct key."
        )

    weights_raw = weights_raw.float()
    if weights_raw.dim() == 2 and weights_raw.shape[1] == n_joints:
        lbs_weights = torch.softmax(weights_raw, dim=-1)   # normalise
    elif weights_raw.dim() == 2 and weights_raw.shape[0] == n_joints:
        lbs_weights = torch.softmax(weights_raw.T, dim=-1)
    else:
        raise ValueError(
            f"Unexpected weight shape {list(weights_raw.shape)} for {n_joints} joints."
        )

    # --- Extract motion mask (optional) ---
    motion_mask = None
    for key in ("motion_mask", "mask", "deform_mask"):
        if key in state:
            motion_mask = state[key].bool()
            break
    if motion_mask is None:
        motion_mask = torch.ones(lbs_weights.shape[0], dtype=torch.bool)

    print(f"LBS weights: {lbs_weights.shape}  motion_mask: {motion_mask.sum().item()} / {len(motion_mask)}")
    return {
        "lbs_weights":  lbs_weights.to(device),
        "motion_mask":  motion_mask.to(device),
    }


def _flat_keys(d: dict, prefix: str = "") -> list:
    keys = []
    for k, v in d.items():
        full = f"{prefix}{k}"
        keys.append(full)
        if isinstance(v, dict):
            keys.extend(_flat_keys(v, prefix=full + "."))
    return keys
