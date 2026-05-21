"""Forward Kinematics (FK) and Linear Blend Skinning (LBS) — RigGS-compatible."""

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
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(n, 4, device=R.device, dtype=R.dtype)

    s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s

    mask2 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & ~(trace > 0)
    s2 = torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

    mask3 = (R[:, 1, 1] > R[:, 2, 2]) & ~(trace > 0) & ~mask2
    s3 = torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

    mask4 = ~(trace > 0) & ~mask2 & ~mask3
    s4 = torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-10)) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    return F.normalize(q, dim=-1).reshape(*batch, 4)


# ---------------------------------------------------------------------------
# RigGS-compatible FK + LBS
# [REF: RigGS/skeleton_utils/skeleton_warp.py - chain_product_transform / deform_by_pose]
# ---------------------------------------------------------------------------

def _transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Build batch of 4x4 homogeneous transforms.
    R: [N, 3, 3], t: [N, 3, 1]
    Returns: [N, 4, 4]
    """
    top = torch.cat([R, t], dim=-1)                              # [N, 3, 4]
    bot = torch.zeros(R.shape[0], 1, 4, device=R.device, dtype=R.dtype)
    bot[:, 0, 3] = 1.0
    return torch.cat([top, bot], dim=1)                          # [N, 4, 4]


def riggs_chain_fk(local_quats: torch.Tensor,
                    nodes: torch.Tensor,
                    parents: torch.Tensor):
    """
    RigGS-compatible FK: each joint rotates around its parent's rest position.
    [REF: RigGS/skeleton_utils/skeleton_warp.py - chain_product_transform]

    Key difference from standard FK: the pivot for joint j is the PARENT joint's
    world position, not the joint's own position.
    local_trans_j = p_parent - R_j @ p_parent  →  T_j rotates x around p_parent.

    local_quats: [N_j, 4]  (w,x,y,z)
    nodes:       [N_j, 3]  rest-pose world positions
    parents:     [N_j]     parent index (root: parents[0]==0)

    Returns:
        transforms   [N_j, 4, 4]  global joint transforms
        posed_joints [N_j, 3]     deformed joint world positions
    """
    N_j = nodes.shape[0]
    R = quat_to_matrix(local_quats)          # [N_j, 3, 3]
    joints = nodes.unsqueeze(-1)              # [N_j, 3, 1]

    # Root's virtual parent = itself
    vp = parents.clone()
    vp[0] = 0

    # local_trans = (I - R_j) @ p_parent  ←  rotate around parent position
    RJ = torch.bmm(R, joints[vp])            # [N_j, 3, 1]
    local_trans = joints[vp] - RJ            # [N_j, 3, 1]

    T_local = _transform_mat(R, local_trans) # [N_j, 4, 4]

    # Chain: G_j = G_parent(j) @ T_local_j
    chain = [T_local[0]]
    for i in range(1, N_j):
        chain.append(chain[int(parents[i].item())] @ T_local[i])
    transforms = torch.stack(chain)          # [N_j, 4, 4]

    # Posed joint positions: G_j @ [p_j; 1]
    jh = F.pad(joints, [0, 0, 0, 1], value=1)          # [N_j, 4, 1]
    posed_joints = torch.bmm(transforms, jh).squeeze(-1)[:, :3]  # [N_j, 3]

    return transforms, posed_joints


def apply_lbs_riggs(xyz: torch.Tensor,
                     transforms: torch.Tensor,
                     lbs_weights: torch.Tensor,
                     motion_mask: torch.Tensor = None) -> torch.Tensor:
    """
    RigGS-compatible LBS for Gaussian positions.
    [REF: RigGS/skeleton_utils/skeleton_warp.py - deform_by_pose, Ax computation]

    Formula: deformed_j = R_global_j @ x + t_global_j
             blended    = sum_j( w_j * deformed_j )
    Equivalent to rotating x around parent joint position (not the joint itself).

    xyz:        [N, 3]
    transforms: [N_j, 4, 4]
    lbs_weights:[N, N_j]
    motion_mask:[N] float (optional)

    Returns: deformed xyz [N, 3]
    """
    R_g = transforms[:, :3, :3]   # [N_j, 3, 3]
    t_g = transforms[:, :3, 3]    # [N_j, 3]

    # Per-joint deformed positions: [N, N_j, 3]
    Ax = torch.einsum('jab,nb->nja', R_g, xyz) + t_g.unsqueeze(0)

    # Weighted blend
    Ax_avg = (Ax * lbs_weights.unsqueeze(-1)).sum(dim=1)  # [N, 3]

    d_xyz = Ax_avg - xyz
    if motion_mask is not None:
        d_xyz = d_xyz * motion_mask.float().unsqueeze(-1)

    return xyz + d_xyz


def apply_lbs_rotation_riggs(canonical_rot: torch.Tensor,
                               transforms: torch.Tensor,
                               lbs_weights: torch.Tensor,
                               motion_mask: torch.Tensor = None) -> torch.Tensor:
    """
    RigGS-compatible rotation blending for Gaussians.
    [REF: RigGS/skeleton_utils/skeleton_warp.py - deform_by_pose, rotation blending]

    Blends global joint rotation quaternions weighted by lbs_weights,
    then composes with canonical Gaussian orientation.

    canonical_rot: [N, 4]  (w,x,y,z)
    transforms:    [N_j, 4, 4]
    lbs_weights:   [N, N_j]
    motion_mask:   [N] float (optional, 0=static)

    Returns: [N, 4] unit quaternions
    """
    R_g = transforms[:, :3, :3]           # [N_j, 3, 3]
    node_rot = matrix_to_quat(R_g)         # [N_j, 4]

    # Weighted sum of joint quaternions
    blended = torch.einsum('nj,jk->nk', lbs_weights, node_rot)  # [N, 4]
    blended = F.normalize(blended, dim=-1)

    if motion_mask is not None:
        mask = motion_mask.float().unsqueeze(-1)          # [N, 1]
        identity = torch.tensor([1., 0., 0., 0.],
                                 device=blended.device, dtype=blended.dtype)
        blended = mask * blended + (1.0 - mask) * identity
        blended = F.normalize(blended, dim=-1)

    # Compose: blended_rotation @ canonical_rotation
    R_blend = quat_to_matrix(blended)                     # [N, 3, 3]
    R_canon = quat_to_matrix(canonical_rot)               # [N, 3, 3]
    R_deformed = torch.bmm(R_blend, R_canon)              # [N, 3, 3]
    return matrix_to_quat(R_deformed)                     # [N, 4]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deform_gaussians(gaussians: dict,
                      local_quats: torch.Tensor,
                      rest_joints: torch.Tensor,
                      parents: torch.Tensor,
                      lbs_weights: torch.Tensor,
                      motion_mask: torch.Tensor = None,
                      skip_rotation: bool = False,
                      global_trans: torch.Tensor = None) -> dict:
    """
    Full RigGS-compatible deformation pipeline: FK → LBS.

    gaussians:    dict from ply_loader.load_3dgs_ply()
    local_quats:  [N_j, 4]  predicted joint rotations for one frame
    rest_joints:  [N_j, 3]
    parents:      [N_j]
    lbs_weights:  [N_gauss, N_j]
    motion_mask:  [N_gauss] float (optional, 0=static Gaussian)
    skip_rotation: if True, skip Gaussian orientation update (use for isotropic GS)
    global_trans: [3] or [1, 3]  global scene translation (from RigGS PoseMLP)

    Returns: deformed gaussians dict with updated xyz (and rotation if not skipped).
    """
    transforms, _ = riggs_chain_fk(local_quats, rest_joints, parents)

    deformed = dict(gaussians)
    deformed["xyz"] = apply_lbs_riggs(
        gaussians["xyz"], transforms, lbs_weights, motion_mask
    )

    if global_trans is not None:
        gt = global_trans.to(deformed["xyz"].device).reshape(1, 3)
        deformed["xyz"] = deformed["xyz"] + gt

    if not skip_rotation:
        deformed["rotation"] = apply_lbs_rotation_riggs(
            gaussians["rotation"], transforms, lbs_weights, motion_mask
        )

    return deformed
