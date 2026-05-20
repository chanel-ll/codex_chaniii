"""Load canonical Gaussian parameters from a 3DGS-format PLY file."""

import numpy as np
import torch
from plyfile import PlyData


def load_3dgs_ply(path: str, device: str = "cpu") -> dict:
    """
    Load canonical Gaussian parameters from a 3DGS-format PLY file.

    Returns dict with keys:
        xyz        [N, 3]      float32  Gaussian centers
        features   [N, K, 3]  float32  SH coefficients (DC + rest, all degrees)
        opacity    [N]         float32  logit opacity (before sigmoid)
        scaling    [N, 3]      float32  log scale (before exp)
        rotation   [N, 4]      float32  unit quaternion (w,x,y,z)
        sh_degree  int         max SH degree inferred from feature count
    """
    ply = PlyData.read(path)
    v = ply["vertex"]

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)  # [N, 3]
    N = xyz.shape[0]

    # DC SH: stored as f_dc_0, f_dc_1, f_dc_2
    # In 3DGS save_ply: features_dc transposed [N,1,3] → [N,3,1] then flattened → [N,3]
    # So f_dc_0=r, f_dc_1=g, f_dc_2=b of the DC component
    f_dc = np.zeros((N, 3, 1))
    f_dc[:, 0, 0] = np.asarray(v["f_dc_0"])
    f_dc[:, 1, 0] = np.asarray(v["f_dc_1"])
    f_dc[:, 2, 0] = np.asarray(v["f_dc_2"])
    f_dc = f_dc.transpose(0, 2, 1)  # [N, 1, 3]

    # Rest SH: stored as f_rest_0..f_rest_{3*K_rest-1}
    # In 3DGS save_ply: features_rest [N,K_rest,3] → transposed [N,3,K_rest] → flattened [N,3*K_rest]
    # So order: r0,r1,...,rK, g0,...,gK, b0,...,bK
    rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    if rest_names:
        f_rest_flat = np.stack([np.asarray(v[n]) for n in rest_names], axis=1)  # [N, 3*K_rest]
        K_rest = len(rest_names) // 3
        f_rest = f_rest_flat.reshape(N, 3, K_rest).transpose(0, 2, 1)  # [N, K_rest, 3]
    else:
        f_rest = np.zeros((N, 0, 3))

    features = np.concatenate([f_dc, f_rest], axis=1)  # [N, K, 3]
    K = features.shape[1]
    # sh_degree: K = (sh_degree+1)^2
    sh_degree = int(round(K ** 0.5)) - 1

    opacity = np.asarray(v["opacity"])  # [N]

    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    scaling = np.stack([np.asarray(v[n]) for n in scale_names], axis=1)  # [N, 3]

    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot")],
        key=lambda n: int(n.split("_")[-1]),
    )
    rotation = np.stack([np.asarray(v[n]) for n in rot_names], axis=1)  # [N, 4] (w,x,y,z)

    # fea_* attributes: RigGS stores skinning features + motion_mask as last channel
    fea_names = sorted(
        [p.name for p in v.properties if p.name.startswith("fea_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    if fea_names:
        fea = np.stack([np.asarray(v[n]) for n in fea_names], axis=1)  # [N, fea_dim]
        motion_mask = 1.0 / (1.0 + np.exp(-fea[:, -1]))                # sigmoid of last channel
        print(f"  fea_* attributes: {len(fea_names)} dims → motion_mask loaded")
    else:
        motion_mask = np.ones(N, dtype=np.float32)
        print(f"  No fea_* attributes found — motion_mask set to all-ones")

    print(f"Loaded {N} Gaussians (sh_degree={sh_degree}) from {path}")
    return {
        "xyz":         torch.from_numpy(xyz).float().to(device),
        "features":    torch.from_numpy(features).float().to(device),
        "opacity":     torch.from_numpy(opacity).float().to(device),
        "scaling":     torch.from_numpy(scaling).float().to(device),
        "rotation":    torch.from_numpy(rotation).float().to(device),
        "sh_degree":   sh_degree,
        "motion_mask": torch.from_numpy(motion_mask.astype(np.float32)).to(device),  # [N]
    }
