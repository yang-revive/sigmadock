"""Flow matching on ligand atom coordinates (Å), with the pocket held fixed.

Path:      x_t = (1 - t) x_0 + t x_1,   x_0 ~ N(pocket centre, prior_std² I),  t ∈ [0, 1] (1 = data)
Model:     predicts x̂_1 = E[x_1 | x_t]; the velocity is v = (x̂_1 - x_t) / (1 - t)
Sampling:  Euler steps x ← x + Δt · v from t = 0 to 1 (the last step lands exactly on x̂_1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pocketdit.model import PocketDiT


@dataclass
class FlowConfig:
    prior_std: float = 4.0  # Å, isotropic Gaussian prior around the pocket centre
    center_noise: float = 1.0  # Å, jitter of the prior centre during training
    dist_loss_weight: float = 1.0  # auxiliary loss on short intra-ligand distances
    dist_loss_cutoff: float = 3.0  # Å, pairs this close in the true pose (bonds and angles)
    dist_loss_t_min: float = 0.75  # only near the data end, where x̂_1 should be sharp
    num_steps: int = 20  # default number of sampling steps


def sample_prior(batch: dict, std: float, center_noise: float = 0.0) -> torch.Tensor:
    center = batch["ref_center"]
    if center_noise > 0:
        center = center + center_noise * torch.randn_like(center)
    x0 = center[:, None] + std * torch.randn_like(batch["lig_pos"])
    return x0 * batch["lig_mask"].unsqueeze(-1)


def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    """Pairwise distances that stay differentiable on the diagonal."""
    return ((x[:, :, None] - x[:, None]).pow(2).sum(-1) + 1e-8).sqrt()


def flow_matching_loss(model: PocketDiT, batch: dict, cfg: FlowConfig) -> tuple[torch.Tensor, dict[str, float]]:
    x1, mask = batch["lig_pos"], batch["lig_mask"]
    m = mask.float()
    t = torch.rand(x1.shape[0], device=x1.device)
    x0 = sample_prior(batch, cfg.prior_std, cfg.center_noise)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    x1_hat = model(batch, xt, t).float()

    n_atoms = m.sum(1)
    coord = (((x1_hat - x1) ** 2).sum(-1) * m).sum(1) / n_atoms  # per-complex MSD (Å²)

    d_true = _pairwise_dist(x1)
    eye = torch.eye(x1.shape[1], dtype=torch.bool, device=x1.device)
    pairs = (d_true < cfg.dist_loss_cutoff) & mask[:, :, None] & mask[:, None, :] & ~eye
    d_err = (_pairwise_dist(x1_hat) - d_true) ** 2
    dist = (d_err * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp(min=1)
    dist = dist * (t > cfg.dist_loss_t_min)

    loss = (coord + cfg.dist_loss_weight * dist).mean()
    metrics = {
        "loss": loss.item(),
        "coord_mse": coord.mean().item(),
        "dist_mse": dist.sum().item() / max(int((t > cfg.dist_loss_t_min).sum()), 1),
    }
    return loss, metrics


@torch.no_grad()
def sample(model: PocketDiT, batch: dict, cfg: FlowConfig, num_steps: int | None = None) -> torch.Tensor:
    """Generate ligand poses [B, N, 3] (Å). The pocket is encoded once and reused at every step."""
    steps = num_steps or cfg.num_steps
    x = sample_prior(batch, cfg.prior_std)
    h_p = model.encode_pocket(batch)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    for i in range(steps):
        t = ts[i].expand(x.shape[0])
        x1_hat = model(batch, x, t, h_p=h_p).float()
        x = x + (ts[i + 1] - ts[i]) / (1 - ts[i]) * (x1_hat - x)
    return x * batch["lig_mask"].unsqueeze(-1)


def rmsd(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-complex RMSD without alignment or symmetry correction (use RDKit for the latter)."""
    m = mask.float()
    return ((((x - y) ** 2).sum(-1) * m).sum(1) / m.sum(1)).sqrt()
