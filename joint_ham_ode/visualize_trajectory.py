"""
Visualize joint trajectory from a trained ODE model.

Renders the skeleton motion (joint positions + bone connections) using
matplotlib 3D animation — no Gaussian rasterizer required.

Supports side-by-side comparison of two checkpoints (e.g. neural_ode vs
hamiltonian_ode), and optionally overlays the ground-truth trajectory.

Usage examples
--------------
# Single model — interp + extrap in one animation
python -m joint_ham_ode.visualize_trajectory \
    --checkpoint output/standup/neural_ode/model_final.pt \
    --theta_path data/standup/joint_trajectory.npz

# Two models side-by-side
python -m joint_ham_ode.visualize_trajectory \
    --checkpoint output/standup/neural_ode/model_final.pt \
    --checkpoint2 output/standup/hamiltonian_ode/model_final.pt \
    --theta_path data/standup/joint_trajectory.npz \
    --label "Neural ODE" --label2 "Hamiltonian ODE"

# Save video (requires ffmpeg)
python -m joint_ham_ode.visualize_trajectory \
    --checkpoint output/standup/neural_ode/model_final.pt \
    --theta_path data/standup/joint_trajectory.npz \
    --save_video output/standup/neural_ode/traj_vis.mp4

# Use skeleton_tree.npz for rest joint positions and parents
python -m joint_ham_ode.visualize_trajectory \
    --checkpoint output/standup/neural_ode/model_final.pt \
    --theta_path data/standup/joint_trajectory.npz \
    --skeleton_path riggs_output/skeleton_tree.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")           # non-interactive; switch to TkAgg/Qt5Agg for live window
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from .train import build_model
from .data.riggs_loader import load_preextracted, load_npz_full, prepare_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    data_meta = ckpt.get("data_meta", {})
    n_joints = data_meta.get("N_j")
    if n_joints is None:
        raise KeyError("Checkpoint missing data_meta['N_j']. Re-train with latest train.py.")
    model = build_model(config, n_joints).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, n_joints


# ---------------------------------------------------------------------------
# Trajectory prediction
# ---------------------------------------------------------------------------

def predict(model, config, data, device):
    """Return (q_interp [T_train, N_j, 4], q_extrap [T_extrap, N_j, 4])."""
    model_type = config["model_type"]
    N_j    = data["N_j"]
    rot_dim = data["rot_dim"]
    T_train = data["T_train"]

    model.eval()
    with torch.no_grad():
        if model_type == "neural_ode":
            q_interp, _, _ = model(
                obs_traj=data["theta_train"],
                extrap_times=data["t_train"],
            )
            q_extrap, _, _ = model(
                obs_traj=data["theta_train"],
                extrap_times=data["t_extrap"],
            )

        elif model_type == "hamiltonian_ode":
            q0 = data["theta_train"][0].reshape(N_j, rot_dim)
            p0 = data["dtheta_train"][0].reshape(N_j, rot_dim)
            t_full = torch.cat([data["t_train"], data["t_extrap"]])
            q_full, _ = model(q0, p0, t_full)
            q_interp = q_full[:T_train]
            q_extrap  = q_full[T_train:]

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    return q_interp, q_extrap   # both [T, N_j, 4]


# ---------------------------------------------------------------------------
# FK: quaternion → world joint positions
# ---------------------------------------------------------------------------

def _quat_to_matrix(q):
    """[..., 4] (w,x,y,z) → [..., 3, 3]"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    B = q.shape[:-1]
    R = torch.stack([
        1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
        2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x),
        2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y),
    ], dim=-1).reshape(*B, 3, 3)
    return R


def fk_positions(rest_joints, parents, quats):
    """
    rest_joints: [N_j, 3]
    parents:     [N_j] int
    quats:       [T, N_j, 4]
    Returns:     [T, N_j, 3]  world joint positions
    """
    T, N_j, _ = quats.shape
    device = quats.device
    R_all = _quat_to_matrix(quats)  # [T, N_j, 3, 3]

    T_mats = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(T, N_j, 1, 1)

    for j in range(N_j):
        p = int(parents[j].item())
        if j == 0:
            offset = rest_joints[j]
        else:
            offset = rest_joints[j] - rest_joints[p]

        L = torch.eye(4, device=device).unsqueeze(0).repeat(T, 1, 1)   # [T, 4, 4]
        L[:, :3, :3] = R_all[:, j]
        L[:, :3,  3] = offset.unsqueeze(0)

        if j == 0:
            T_mats[:, j] = L
        else:
            T_mats[:, j] = torch.bmm(T_mats[:, p], L)

    positions = T_mats[:, :, :3, 3]   # [T, N_j, 3]
    return positions.cpu().numpy()


# ---------------------------------------------------------------------------
# Skeleton loading (optional)
# ---------------------------------------------------------------------------

def load_skeleton(skeleton_path: str, device: str):
    """Load rest joints and parent indices from skeleton_tree.npz."""
    npz = np.load(skeleton_path)
    rest_joints = torch.from_numpy(npz["nodes"]).float().to(device)
    parents = torch.from_numpy(npz["parents"]).long().to(device)
    print(f"  Skeleton: {rest_joints.shape[0]} joints from {skeleton_path}")
    return rest_joints, parents


def _make_default_skeleton(N_j, device):
    """Dummy rest joints in a vertical line when no skeleton file is provided."""
    rest_joints = torch.zeros(N_j, 3, device=device)
    rest_joints[:, 1] = torch.linspace(0.0, 1.0, N_j)
    parents = torch.zeros(N_j, dtype=torch.long, device=device)
    for j in range(1, N_j):
        parents[j] = j - 1
    return rest_joints, parents


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------

BONE_COLOR    = {"pred1": "#2196F3", "pred2": "#FF5722", "gt": "#4CAF50"}
JOINT_COLOR   = {"pred1": "#1565C0", "pred2": "#BF360C", "gt": "#1B5E20"}
REGION_ALPHA  = 0.08   # shading for extrap region


def _draw_skeleton_3d(ax, pos, parents, color, alpha=1.0, linewidth=1.5, zorder=2):
    """Draw joints + bones on a 3D axis. pos: [N_j, 3]."""
    N_j = pos.shape[0]
    scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=color, s=20, alpha=alpha, zorder=zorder)
    lines = []
    for j in range(1, N_j):
        p = int(parents[j])
        xs = [pos[j, 0], pos[p, 0]]
        ys = [pos[j, 1], pos[p, 1]]
        zs = [pos[j, 2], pos[p, 2]]
        ln, = ax.plot(xs, ys, zs, color=color, alpha=alpha,
                      linewidth=linewidth, zorder=zorder)
        lines.append(ln)
    return scat, lines


def _update_skeleton_3d(scat, lines, pos, parents):
    """Update existing scatter + line artists in place."""
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    bone_idx = 0
    for j in range(1, pos.shape[0]):
        p = int(parents[j])
        xs = [pos[j, 0], pos[p, 0]]
        ys = [pos[j, 1], pos[p, 1]]
        zs = [pos[j, 2], pos[p, 2]]
        lines[bone_idx].set_data_3d(xs, ys, zs)
        bone_idx += 1


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    # --- Load model(s) -------------------------------------------------------
    model1, config1, n_joints = load_model(args.checkpoint, device)
    label1 = args.label or config1["model_type"].replace("_", " ").title()

    model2, config2, label2 = None, None, None
    if args.checkpoint2:
        model2, config2, _ = load_model(args.checkpoint2, device)
        label2 = args.label2 or config2["model_type"].replace("_", " ").title()

    # --- Load trajectory data -------------------------------------------------
    timestamps = None
    if args.theta_path.endswith(".npz"):
        npz_data = load_npz_full(args.theta_path, device=device)
        theta     = npz_data["theta"]
        timestamps = npz_data.get("timestamps")
        # Use parent indices from npz if available
        parents_from_npz = npz_data.get("parent_indices")
    else:
        theta = load_preextracted(args.theta_path, device=device)
        parents_from_npz = None

    data = prepare_data(theta, time_split=config1["data"]["time_split"],
                        timestamps=timestamps)
    T_train  = data["T_train"]
    T_extrap = data["T_extrap"]
    T_total  = data["T_total"]
    N_j      = data["N_j"]
    rot_dim  = data["rot_dim"]

    # --- Load rest skeleton ---------------------------------------------------
    if args.skeleton_path:
        rest_joints, parents = load_skeleton(args.skeleton_path, device)
    elif parents_from_npz is not None:
        # npz provides parent indices; derive rest joints from GT positions if available
        parents = parents_from_npz
        jp = npz_data.get("joint_position")
        if jp is not None:
            rest_joints = jp[0]   # first frame as rest pose
            print(f"  rest_joints from npz joint_position[0]: {rest_joints.shape}")
        else:
            rest_joints, _ = _make_default_skeleton(N_j, device)
            print("  No joint_position in npz — using dummy chain rest pose")
    else:
        rest_joints, parents = _make_default_skeleton(N_j, device)
        print(f"  No skeleton file — using dummy {N_j}-joint chain")

    if rest_joints.shape[0] != N_j:
        print(f"  Warning: skeleton N_j={rest_joints.shape[0]} != trajectory N_j={N_j}")
        rest_joints = rest_joints[:N_j]
        parents = parents[:N_j]

    # --- Predict trajectories ------------------------------------------------
    print("Predicting trajectories...")
    q_interp1, q_extrap1 = predict(model1, config1, data, device)
    q_full1 = torch.cat([q_interp1, q_extrap1], dim=0)   # [T_total, N_j, 4]

    q_full2 = None
    if model2 is not None:
        q_interp2, q_extrap2 = predict(model2, config2, data, device)
        q_full2 = torch.cat([q_interp2, q_extrap2], dim=0)

    # Ground truth
    q_gt = theta   # [T, N_j, 4]

    # --- FK to world positions -----------------------------------------------
    print("Computing FK positions...")
    pos_pred1 = fk_positions(rest_joints, parents, q_full1)   # [T, N_j, 3]
    pos_gt    = fk_positions(rest_joints, parents, q_gt)

    pos_pred2 = None
    if q_full2 is not None:
        pos_pred2 = fk_positions(rest_joints, parents, q_full2)

    parents_np = parents.cpu().numpy().astype(int)

    # --- Scene bounds --------------------------------------------------------
    all_pos = [pos_pred1, pos_gt]
    if pos_pred2 is not None:
        all_pos.append(pos_pred2)
    all_xyz = np.concatenate(all_pos, axis=0)   # [N, N_j, 3]
    xyz_flat = all_xyz.reshape(-1, 3)
    mn, mx = xyz_flat.min(0), xyz_flat.max(0)
    center = (mn + mx) / 2
    half   = (mx - mn).max() / 2 * 1.15 + 1e-3

    def _set_axes(ax):
        ax.set_xlim(center[0]-half, center[0]+half)
        ax.set_ylim(center[1]-half, center[1]+half)
        ax.set_zlim(center[2]-half, center[2]+half)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # --- Figure layout -------------------------------------------------------
    n_cols = 2 if pos_pred2 is not None else 1
    fig = plt.figure(figsize=(7*n_cols + 1, 7))
    axes = []
    for col in range(n_cols):
        ax = fig.add_subplot(1, n_cols, col+1, projection="3d")
        axes.append(ax)

    # Titles
    axes[0].set_title(label1, fontsize=11, pad=8)
    if len(axes) > 1:
        axes[1].set_title(label2, fontsize=11, pad=8)

    for ax in axes:
        _set_axes(ax)
        ax.view_init(elev=args.elev, azim=args.azim)

    # --- Initial artists -----------------------------------------------------
    def _init_ax(ax, pos_pred, label_pred, color_key):
        s_gt, l_gt = _draw_skeleton_3d(ax, pos_gt[0], parents_np,
                                        BONE_COLOR["gt"], alpha=0.5,
                                        linewidth=1.2, zorder=1)
        s_p, l_p   = _draw_skeleton_3d(ax, pos_pred[0], parents_np,
                                        BONE_COLOR[color_key], alpha=1.0,
                                        linewidth=2.0, zorder=3)

        # Legend proxies
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color=BONE_COLOR[color_key], lw=2, label=label_pred),
            Line2D([0], [0], color=BONE_COLOR["gt"], lw=1.5, alpha=0.5, label="Ground Truth"),
        ]
        ax.legend(handles=legend_elems, loc="upper left", fontsize=8)
        return s_gt, l_gt, s_p, l_p

    arts1 = _init_ax(axes[0], pos_pred1, label1, "pred1")
    arts2 = _init_ax(axes[1], pos_pred2, label2, "pred2") if pos_pred2 is not None else None

    # Frame counter text
    frame_txt = fig.text(0.5, 0.01, "", ha="center", fontsize=10, color="#333")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # --- Animation update ----------------------------------------------------
    def update(frame):
        t = frame % T_total
        is_extrap = t >= T_train
        region = "EXTRAP" if is_extrap else "INTERP"
        color  = "#E53935" if is_extrap else "#1565C0"
        frame_txt.set_text(
            f"Frame {t+1}/{T_total}  [{region}]  "
            f"(train: 0–{T_train-1}  extrap: {T_train}–{T_total-1})"
        )
        frame_txt.set_color(color)

        s_gt1, l_gt1, s_p1, l_p1 = arts1
        _update_skeleton_3d(s_gt1, l_gt1, pos_gt[t], parents_np)
        _update_skeleton_3d(s_p1,  l_p1,  pos_pred1[t], parents_np)

        if arts2 is not None:
            s_gt2, l_gt2, s_p2, l_p2 = arts2
            _update_skeleton_3d(s_gt2, l_gt2, pos_gt[t], parents_np)
            _update_skeleton_3d(s_p2,  l_p2,  pos_pred2[t], parents_np)

        return []

    interval_ms = int(1000 / args.fps)
    ani = animation.FuncAnimation(fig, update, frames=T_total,
                                   interval=interval_ms, blit=False)

    # --- Output ---------------------------------------------------------------
    if args.save_video:
        out_path = args.save_video
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=1200)
        ani.save(out_path, writer=writer)
        print(f"Saved → {out_path}")

    if args.save_gif:
        out_path = args.save_gif
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        ani.save(out_path, writer="pillow", fps=args.fps)
        print(f"Saved → {out_path}")

    if not args.save_video and not args.save_gif:
        # Try interactive display; fall back to saving a single PNG per-frame grid
        try:
            matplotlib.use("TkAgg")
            plt.show()
        except Exception:
            png_path = (args.checkpoint.replace(".pt", "") + "_skeleton_vis.png")
            _save_frame_grid(fig, update, T_total, png_path, args.fps)

    plt.close(fig)


def _save_frame_grid(fig, update_fn, T_total, out_png, fps):
    """Save a grid of evenly-spaced frames as a static PNG."""
    n_cols = 8
    n_frames = min(T_total, 24)
    idxs = [int(i * T_total / n_frames) for i in range(n_frames)]
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig2, axes2 = plt.subplots(n_rows, n_cols,
                                figsize=(n_cols * 2, n_rows * 2))
    axes2 = axes2.flatten()

    for plot_i, frame_i in enumerate(idxs):
        update_fn(frame_i)
        # Render current figure to image buffer
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = buf.reshape(h, w, 3)
        axes2[plot_i].imshow(img)
        axes2[plot_i].axis("off")
        axes2[plot_i].set_title(f"t={frame_i}", fontsize=6)

    for ax in axes2[len(idxs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    print(f"Frame grid → {out_png}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Skeleton motion visualizer for Neural ODE / Hamiltonian ODE trajectories"
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--theta_path",   required=True,
                   help="Joint trajectory file (.npz or .pt)")
    p.add_argument("--checkpoint2",  default=None,
                   help="Second checkpoint for side-by-side comparison")
    p.add_argument("--label",        default=None,
                   help="Display name for --checkpoint model")
    p.add_argument("--label2",       default=None,
                   help="Display name for --checkpoint2 model")
    p.add_argument("--skeleton_path", default=None,
                   help="skeleton_tree.npz from RigGS output (rest joints + parents)")
    p.add_argument("--save_video",   default=None,
                   help="Save animation as MP4 (requires ffmpeg)")
    p.add_argument("--save_gif",     default=None,
                   help="Save animation as GIF (requires pillow)")
    p.add_argument("--fps",          type=float, default=10.0,
                   help="Playback frame rate (default 10)")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--elev",         type=float, default=15.0,
                   help="Camera elevation angle in degrees (default 15)")
    p.add_argument("--azim",         type=float, default=-60.0,
                   help="Camera azimuth angle in degrees (default -60)")
    return p.parse_args()


def main():
    args = parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
