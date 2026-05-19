# [REF: SE3HamDL/se3hamneuralode/se3ham_ode.py]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint


class PotentialEnergyNet(nn.Module):
    """
    V_psi(q): joint rotations → scalar potential energy.
    [REF: SE3HamDL/se3hamneuralode/se3ham_ode.py - V_net]
    """

    def __init__(self, n_joints: int, rot_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        q_dim = n_joints * rot_dim
        layers = [nn.Linear(q_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [..., q_dim]  →  [...] (scalar per sample)
        return self.net(q).squeeze(-1)


class HamiltonianDynamicsFunc(nn.Module):
    """
    H(q,p) = ½||p||² + V_ψ(q)   [M = I, Phase 1]
    dq/dt =  p
    dp/dt = -∂V/∂q

    [REF: SE3HamDL/se3hamneuralode/se3ham_ode.py - SE3HamNODE.forward()]
    """

    def __init__(self, potential_net: PotentialEnergyNet,
                 n_joints: int, rot_dim: int = 4):
        super().__init__()
        self.V = potential_net
        self.q_dim = n_joints * rot_dim

    def forward(self, t, z: torch.Tensor) -> torch.Tensor:
        """
        z: [..., 2*q_dim]  — concatenated [q | p]
        Returns dz/dt of same shape.
        """
        q = z[..., :self.q_dim]
        p = z[..., self.q_dim:]

        # Gradient of V w.r.t. q
        # torch.enable_grad() handles the case where the outer context is no_grad
        with torch.enable_grad():
            q_in = q.detach().requires_grad_(True)
            V_val = self.V(q_in).sum()
            dV_dq = torch.autograd.grad(
                V_val, q_in,
                create_graph=self.training
            )[0]

        dq_dt = p
        dp_dt = -dV_dq
        return torch.cat([dq_dt, dp_dt], dim=-1)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        H(q,p) per sample.
        q, p: [..., q_dim]  →  [...] scalar
        [REF: SE3HamDL/examples/pendulum/rollout_pend_SO3.py]
        """
        T_kin = 0.5 * (p ** 2).sum(dim=-1)
        V_val = self.V(q)
        return T_kin + V_val


def _sign_fix_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """MSE with quaternion double-cover correction."""
    dot = (pred * gt).sum(dim=-1, keepdim=True)
    pred_aligned = torch.where(dot < 0, -pred, pred)
    return F.mse_loss(pred_aligned, gt)


class JointHamiltonianODE(nn.Module):
    """
    Hamiltonian Neural ODE for joint rotation dynamics.
    State: z = [q, p] where q = joint quaternions, p = generalised momenta.

    [REF: SE3HamDL/se3hamneuralode/se3ham_ode.py - SE3HamNODE class]
    """

    def __init__(self, n_joints: int, rot_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 3,
                 solver: str = 'rk4', rtol: float = 1e-4, atol: float = 1e-6,
                 use_adjoint: bool = True):
        super().__init__()
        self.n_joints = n_joints
        self.rot_dim = rot_dim
        self.q_dim = n_joints * rot_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        V_net = PotentialEnergyNet(n_joints, rot_dim, hidden_dim, n_layers)
        self.func = HamiltonianDynamicsFunc(V_net, n_joints, rot_dim)

    def forward(self, q0: torch.Tensor, p0: torch.Tensor,
                t_span: torch.Tensor):
        """
        q0, p0: [N_j, 4]   initial joint positions and momenta
        t_span: [T]         time points (must be monotonically increasing)

        Returns:
            q_traj: [T, N_j, 4]  predicted quaternions (normalised)
            p_traj: [T, N_j, 4]  predicted momenta
        """
        q0_flat = q0.reshape(self.q_dim)
        p0_flat = p0.reshape(self.q_dim)
        z0 = torch.cat([q0_flat, p0_flat]).unsqueeze(0)  # [1, 2*q_dim]

        _odeint = odeint_adjoint if self.use_adjoint else odeint
        z_traj = _odeint(
            self.func, z0, t_span,
            method=self.solver, rtol=self.rtol, atol=self.atol
        )  # [T, 1, 2*q_dim]
        z_traj = z_traj.squeeze(1)   # [T, 2*q_dim]

        q_flat = z_traj[:, :self.q_dim]
        p_flat = z_traj[:, self.q_dim:]

        T = t_span.shape[0]
        q_traj = F.normalize(q_flat.reshape(T, self.n_joints, self.rot_dim), dim=-1)
        p_traj = p_flat.reshape(T, self.n_joints, self.rot_dim)

        return q_traj, p_traj

    def compute_energy(self, q_traj: torch.Tensor,
                       p_traj: torch.Tensor) -> torch.Tensor:
        """
        q_traj, p_traj: [T, N_j, 4]
        Returns H: [T]
        """
        T = q_traj.shape[0]
        q_flat = q_traj.reshape(T, self.q_dim)
        p_flat = p_traj.reshape(T, self.q_dim)
        return self.func.hamiltonian(q_flat, p_flat)

    def compute_loss(self, q_pred: torch.Tensor, p_pred: torch.Tensor,
                     q_gt: torch.Tensor,
                     lambda_energy: float = 0.01,
                     energy_subsample: int = 5) -> dict:
        """
        q_pred, q_gt: [T, N_j, 4]
        p_pred:       [T, N_j, 4]
        """
        L_recon = _sign_fix_mse(q_pred, q_gt)

        # Energy conservation: Var(H) → 0
        idx = slice(None, None, energy_subsample)
        H_t = self.compute_energy(q_pred[idx], p_pred[idx])
        L_energy = H_t.var()

        loss = L_recon + lambda_energy * L_energy

        return {
            'loss': loss,
            'L_recon': L_recon.item(),
            'L_energy': L_energy.item(),
            'H_mean': H_t.mean().item(),
            'H_std': H_t.std().item(),
        }
