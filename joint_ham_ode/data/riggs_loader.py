# [REF: RigGS/train_rig.py - TrainRig class]
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scene import Scene, GaussianModel
    from scene.skeleton_model import SkeletonModel
    from utils.system_utils import searchForMaxIteration
    RIGGS_AVAILABLE = True
except ImportError:
    RIGGS_AVAILABLE = False


def load_riggs_checkpoint(model_path: str, dataset_args, opt_args):
    """
    Load a trained RigGS checkpoint.
    [REF: RigGS/train_rig.py - TrainRig.__init__()]
    """
    if not RIGGS_AVAILABLE:
        raise ImportError(
            "RigGS modules not found. Run this from inside the RigGS environment "
            "or use load_preextracted() with a pre-saved theta.pt file."
        )

    gaussians = GaussianModel(
        dataset_args.sh_degree,
        fea_dim=dataset_args.hyper_dim,
        with_motion_mask=dataset_args.gs_with_motion_mask,
        use_isotropic_gs=dataset_args.use_isotropic_gs,
    )

    load_iteration = searchForMaxIteration(
        os.path.join(model_path, "point_cloud")
    )
    scene = Scene(dataset_args, gaussians, load_iteration=load_iteration)

    skeleton_tree_path = os.path.join(model_path, "skeleton_tree.npz")
    skeleton_tree = np.load(skeleton_tree_path)

    joints = torch.from_numpy(skeleton_tree["nodes"]).float().cuda()
    parent_indices = torch.from_numpy(skeleton_tree["parents"]).long().cuda()

    skeleton = SkeletonModel(
        K=opt_args.skeleton_weight_knn,
        joints=joints,
        parent_indices=parent_indices,
        is_blender=dataset_args.is_blender,
        skinning=True,
        hyper_dim=dataset_args.hyper_dim,
    )
    skeleton.load_weights(model_path, iteration=load_iteration)
    skeleton = skeleton.cuda().eval()

    return gaussians, skeleton, scene


def extract_joint_trajectory(gaussians, skeleton, scene):
    """
    Extract per-frame joint rotation (quaternion) from a trained RigGS skeleton.
    [REF: RigGS/train_rig.py - TrainRig.deform_gaussians()]

    Returns:
        theta:   [T, N_j, 4]  local rotation quaternions per frame
        d_nodes: [T, N_j, 3]  deformed joint positions per frame
    """
    sorted_cams = sorted(scene.getTrainCameras(), key=lambda x: x.fid)

    theta_list, d_nodes_list = [], []

    with torch.no_grad():
        for cam in sorted_cams:
            time_input = skeleton.deform.expand_time(cam.fid)
            d_values = skeleton.step(
                gaussians.get_xyz.detach(),
                time_input,
                motion_mask=gaussians.motion_mask,
            )
            # .clone() is critical — skeleton state is mutated in-place
            theta_list.append(d_values["local_rotation"].detach().clone())
            d_nodes_list.append(d_values["d_nodes"].detach().clone())

    theta = torch.stack(theta_list)     # [T, N_j, 4]
    d_nodes = torch.stack(d_nodes_list)  # [T, N_j, 3]

    # Ensure unit quaternions
    theta = F.normalize(theta, dim=-1)

    return theta, d_nodes


def _central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Central finite difference along dim-0."""
    dx = torch.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def prepare_data(theta: torch.Tensor, time_split: float = 0.8) -> dict:
    """
    Split trajectory into train (obs) and extrapolation windows.
    Default 8:2 split matches ODE-GS D-NeRF benchmark.
    [REF: ODE-GS/train_extrapolation.py - time_split concept]

    Args:
        theta:      [T, N_j, 4]  quaternion trajectory
        time_split: fraction used for training (default 0.8)

    Returns dict with tensors on the same device as theta.
    """
    T, N_j, rot_dim = theta.shape
    T_train = int(T * time_split)
    T_extrap = T - T_train

    theta_flat = theta.reshape(T, -1)          # [T, N_j*4]
    theta_train = theta_flat[:T_train]          # [T_train, N_j*4]
    theta_extrap = theta_flat[T_train:]         # [T_extrap, N_j*4]

    dt = 1.0 / T
    t_all = torch.linspace(0.0, (T - 1) * dt, T, device=theta.device)
    t_train = t_all[:T_train]
    t_extrap = t_all[T_train:]

    # Angular velocity via central differences — used as p0 for Hamiltonian ODE
    # [REF: SE3HamDL/examples/pendulum/train_pend_SO3.py - velocity init]
    dtheta_train = _central_diff(theta_train, dt)

    return {
        "theta_train":  theta_train,   # [T_train, N_j*4]
        "theta_extrap": theta_extrap,  # [T_extrap, N_j*4]
        "dtheta_train": dtheta_train,  # [T_train, N_j*4]
        "t_train":      t_train,       # [T_train]
        "t_extrap":     t_extrap,      # [T_extrap]
        "T_train":      T_train,
        "T_extrap":     T_extrap,
        "T_total":      T,
        "dt":           dt,
        "N_j":          N_j,
        "rot_dim":      rot_dim,
        "state_dim":    N_j * rot_dim,
    }


def load_preextracted(path: str, device: str = "cuda") -> torch.Tensor:
    """
    Load pre-extracted theta saved by scripts/extract_trajectory.py.
    Supports .pt and .npz formats.
    """
    if path.endswith(".pt"):
        theta = torch.load(path, map_location=device)
    elif path.endswith(".npz"):
        arr = np.load(path)
        theta = torch.from_numpy(arr["theta"]).float().to(device)
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .pt or .npz.")

    if theta.dtype != torch.float32:
        theta = theta.float()

    theta = F.normalize(theta, dim=-1)
    return theta


def load_meta(meta_path: str) -> dict:
    with open(meta_path) as f:
        return json.load(f)
