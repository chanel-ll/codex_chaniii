"""Forward Kinematics (FK) and Linear Blend Skinning (LBS) — pure PyTorch."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation matrices.
    q: [..., 4] (w, x, y, z)
    Returns: [..., 3, 3]
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    B = q.shape[:-1]
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*B, 3, 3)
    return R


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to unit quaternions (w, x, y, z).
    R: [..., 3, 3]
    Returns: [..., 4]
    """
    # Using Shepperd's method (numerically stable)
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(n, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2  # s = 4w
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & ~(trace > 0)
    s2 = torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

    # Case 3: R[1,1] > R[2,2]
    mask3 = (R[:, 1, 1] > R[:, 2, 2]) & ~(trace > 0) & ~mask2
    s3 = torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

    # Case 4: otherwise
    mask4 = ~(trace > 0) & ~mask2 & ~mask3
    s4 = torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-10)) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    return F.normalize(q, dim=-1).reshape(*batch, 4)


# ---------------------------------------------------------------------------
# FK + LBS
# ---------------------------------------------------------------------------

def forward_kinematics(rest_joints: torch.Tensor,
                        parents: torch.Tensor,
                        local_quats: torch.Tensor) -> torch.Tensor:
    """
    Compute per-joint world transforms given local joint rotations.

    rest_joints:  [N_j, 3]   rest-pose joint positions in world space
    parents:      [N_j]      parent index (root: parents[0] == 0)
    local_quats:  [N_j, 4]   (w,x,y,z) local rotation per joint

    Returns: world_transforms [N_j, 4, 4]
    """
    N_j = rest_joints.shape[0]
    device = rest_joints.device
    dtype = rest_joints.dtype

    R = quat_to_matrix(local_quats)  # [N_j, 3, 3]

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N_j, 1, 1)

    for j in range(N_j):
        parent = int(parents[j].item())
        offset = rest_joints[j] - rest_joints[parent]  # relative to parent (root: relative to itself = 0)
        if j == 0:
            offset = rest_joints[j]  # root: absolute position

        L = torch.eye(4, device=device, dtype=dtype)
        L[:3, :3] = R[j]
        L[:3,  3] = offset

        T[j] = T[parent] @ L if j != 0 else L

    return T  # [N_j, 4, 4]


def compute_skinning_transforms(world_T: torch.Tensor,
                                  rest_joints: torch.Tensor) -> torch.Tensor:
    """
    Skinning transform: S_j = G_j @ inv(T_rest_j)
    where T_rest_j = Translation(rest_joints[j]).

    world_T:     [N_j, 4, 4]  global joint transforms from FK
    rest_joints: [N_j, 3]     rest-pose positions

    Returns: [N_j, 4, 4]
    """
    N_j = rest_joints.shape[0]
    device = rest_joints.device
    dtype = rest_joints.dtype

    T_rest_inv = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N_j, 1, 1)
    T_rest_inv[:, :3, 3] = -rest_joints  # inv of pure translation

    return torch.bmm(world_T, T_rest_inv)  # [N_j, 4, 4]


def apply_lbs(canonical_xyz: torch.Tensor,
               skinning_T: torch.Tensor,
               lbs_weights: torch.Tensor,
               motion_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Linear Blend Skinning for Gaussian center positions.

    canonical_xyz: [N, 3]
    skinning_T:    [N_j, 4, 4]
    lbs_weights:   [N, N_j]
    motion_mask:   [N] bool (True = deformable). If None, all deformed.

    Returns: deformed_xyz [N, 3]
    """
    N = canonical_xyz.shape[0]
    device = canonical_xyz.device
    dtype = canonical_xyz.dtype

    ones = torch.ones(N, 1, device=device, dtype=dtype)
    xyz_h = torch.cat([canonical_xyz, ones], dim=1)  # [N, 4]

    # [N_j, 4, 4] x [4, N] → [N_j, 4, N] → [N, N_j, 4]
    deformed_per_joint = torch.einsum("jab,nb->nja", skinning_T, xyz_h)  # [N, N_j, 4]

    # Weighted blend: [N, N_j, 1] * [N, N_j, 4] → [N, 4]
    blended = (lbs_weights.unsqueeze(-1) * deformed_per_joint).sum(dim=1)  # [N, 4]
    deformed = blended[:, :3]

    if motion_mask is not None:
        deformed = torch.where(motion_mask.unsqueeze(-1), deformed, canonical_xyz)

    return deformed


def apply_lbs_rotation(canonical_rot: torch.Tensor,
                        skinning_T: torch.Tensor,
                        lbs_weights: torch.Tensor,
                        motion_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Apply LBS to Gaussian orientations via blended rotation matrices.

    canonical_rot: [N, 4]  (w,x,y,z) unit quaternions
    skinning_T:    [N_j, 4, 4]
    lbs_weights:   [N, N_j]
    motion_mask:   [N] bool

    Returns: deformed_rot [N, 4] unit quaternions
    """
    R_gs = quat_to_matrix(canonical_rot)          # [N, 3, 3]
    R_joints = skinning_T[:, :3, :3]              # [N_j, 3, 3]

    # Blend joint rotation matrices weighted per Gaussian: [N, 3, 3]
    R_blended = torch.einsum("nj,jab->nab", lbs_weights, R_joints)

    # Apply blended joint rotation to canonical Gaussian rotation
    R_deformed = torch.bmm(R_blended, R_gs)       # [N, 3, 3]
    q_deformed = matrix_to_quat(R_deformed)        # [N, 4]

    if motion_mask is not None:
        q_deformed = torch.where(motion_mask.unsqueeze(-1), q_deformed, canonical_rot)

    return q_deformed


def deform_gaussians(gaussians: dict,
                      local_quats: torch.Tensor,
                      rest_joints: torch.Tensor,
                      parents: torch.Tensor,
                      lbs_weights: torch.Tensor,
                      motion_mask: torch.Tensor = None) -> dict:
    """
    Full deformation pipeline: FK → skinning transforms → LBS.

    gaussians:   dict from ply_loader.load_3dgs_ply()
    local_quats: [N_j, 4]  predicted joint rotations for one frame
    rest_joints: [N_j, 3]
    parents:     [N_j]
    lbs_weights: [N_gauss, N_j]
    motion_mask: [N_gauss] bool (optional)

    Returns: deformed gaussians dict with updated xyz and rotation.
    """
    world_T    = forward_kinematics(rest_joints, parents, local_quats)
    skinning_T = compute_skinning_transforms(world_T, rest_joints)

    deformed = dict(gaussians)  # shallow copy
    deformed["xyz"]      = apply_lbs(gaussians["xyz"], skinning_T, lbs_weights, motion_mask)
    deformed["rotation"] = apply_lbs_rotation(gaussians["rotation"], skinning_T, lbs_weights, motion_mask)
    return deformed
