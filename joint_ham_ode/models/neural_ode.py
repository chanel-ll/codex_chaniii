# [REF: ODE-GS/scene/extrapolation_ode_model.py]
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class LatentODEFunc(nn.Module):
    """
    dz/dt = f_phi(z)  — autonomous MLP in latent space.
    [REF: ODE-GS/scene/extrapolation_ode_model.py - LatentODEfunc]
    """

    def __init__(self, latent_dim: int = 64, nhidden: int = 256,
                 num_layers: int = 3, use_tanh: bool = True):
        super().__init__()
        activation = nn.Tanh() if use_tanh else nn.ReLU(inplace=True)

        layers = [nn.Linear(latent_dim, nhidden), activation]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(nhidden, nhidden), activation]
        layers.append(nn.Linear(nhidden, latent_dim))
        self.net = nn.Sequential(*layers)

        # weight initialisation following ODE-GS convention
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if use_tanh:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        # t is unused (autonomous system)
        return self.net(z)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal PE over a variable-length time sequence."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [T, d_model]
        return x + self.pe[:x.size(0)]


class JointODEEncoder(nn.Module):
    """
    Transformer encoder: observed joint trajectory → latent z0.
    [REF: ODE-GS/scene/extrapolation_ode_model.py - TransformerLatentODEWrapper encoder]

    Produces mu and logvar for VAE reparameterisation.
    """

    def __init__(self, obs_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, latent_dim: int = 64):
        super().__init__()
        self.value_embedding = nn.Linear(obs_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=False, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.z0_proj = nn.Linear(d_model, latent_dim * 2)  # mu + logvar

    def forward(self, obs_traj: torch.Tensor):
        """
        obs_traj: [T_obs, obs_dim]
        Returns z0 [latent_dim], mu [latent_dim], logvar [latent_dim]
        """
        x = self.value_embedding(obs_traj)   # [T_obs, d_model]
        x = self.pos_enc(x)                  # [T_obs, d_model]
        x = x.unsqueeze(1)                   # [T_obs, 1, d_model] — batch_first=False
        h = self.transformer(x)              # [T_obs, 1, d_model]
        h_last = h[-1, 0, :]                 # last timestep [d_model]
        params = self.z0_proj(h_last)        # [latent_dim * 2]
        mu, logvar = params.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z0 = mu + eps * (0.5 * logvar).exp()
        return z0, mu, logvar


class JointODEDecoder(nn.Module):
    """
    Latent z → joint rotation state.
    [REF: ODE-GS/scene/extrapolation_ode_model.py - decoder MLP]
    """

    def __init__(self, latent_dim: int = 64, obs_dim: int = 1,
                 nhidden: int = 128, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(latent_dim, nhidden), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(nhidden, nhidden), nn.Tanh()]
        layers.append(nn.Linear(nhidden, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [T, latent_dim] → [T, obs_dim]
        return self.net(z)


def _sign_fix_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """MSE with quaternion double-cover correction: q ≡ -q."""
    # pred, gt: [..., 4]
    dot = (pred * gt).sum(dim=-1, keepdim=True)  # [..., 1]
    pred_aligned = torch.where(dot < 0, -pred, pred)
    return F.mse_loss(pred_aligned, gt)


class JointNeuralODE(nn.Module):
    """
    Full Neural ODE model following ODE-GS architecture, applied to joint rotations.

    Pipeline:
        obs_traj  →  Transformer encoder  →  z0
        z0        →  LatentODEFunc + odeint  →  z_traj
        z_traj    →  Decoder  →  q_pred (quaternions)

    [REF: ODE-GS/scene/extrapolation_ode_model.py - TransformerLatentODEWrapper]
    """

    def __init__(self, n_joints: int, rot_dim: int = 4,
                 latent_dim: int = 64, d_model: int = 128,
                 nhead: int = 8, num_enc_layers: int = 4,
                 ode_nhidden: int = 256, ode_layers: int = 3,
                 decoder_nhidden: int = 128,
                 solver: str = 'rk4', rtol: float = 1e-4, atol: float = 1e-6,
                 kl_beta: float = 1e-3):
        super().__init__()
        self.n_joints = n_joints
        self.rot_dim = rot_dim
        self.state_dim = n_joints * rot_dim
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.kl_beta = kl_beta

        self.encoder = JointODEEncoder(
            obs_dim=self.state_dim, d_model=d_model,
            nhead=nhead, num_layers=num_enc_layers, latent_dim=latent_dim
        )
        self.func = LatentODEFunc(
            latent_dim=latent_dim, nhidden=ode_nhidden, num_layers=ode_layers
        )
        self.decoder = JointODEDecoder(
            latent_dim=latent_dim, obs_dim=self.state_dim,
            nhidden=decoder_nhidden, num_layers=3
        )

    def forward(self, obs_traj: torch.Tensor, extrap_times: torch.Tensor):
        """
        obs_traj:     [T_obs, N_j*4]  observed joint rotations (flattened)
        extrap_times: [T_ext]          time points to predict

        Returns:
            q_pred:  [T_ext, N_j, 4]  predicted quaternions (normalised)
            mu:      [latent_dim]
            logvar:  [latent_dim]
        """
        z0, mu, logvar = self.encoder(obs_traj)   # [latent_dim]

        z0_batch = z0.unsqueeze(0)                # [1, latent_dim] for odeint
        z_traj = odeint(
            self.func, z0_batch, extrap_times,
            method=self.solver, rtol=self.rtol, atol=self.atol
        )  # [T_ext, 1, latent_dim]
        z_traj = z_traj.squeeze(1)               # [T_ext, latent_dim]

        q_flat = self.decoder(z_traj)            # [T_ext, N_j*4]
        q_pred = q_flat.reshape(-1, self.n_joints, self.rot_dim)
        q_pred = F.normalize(q_pred, dim=-1)

        return q_pred, mu, logvar

    def compute_loss(self, q_pred: torch.Tensor, q_gt: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = None) -> dict:
        """
        q_pred, q_gt: [T, N_j, 4]
        """
        if beta is None:
            beta = self.kl_beta

        L_recon = _sign_fix_mse(q_pred, q_gt)
        L_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = L_recon + beta * L_kl

        return {
            'loss': loss,
            'L_recon': L_recon.item(),
            'L_kl': L_kl.item(),
        }
