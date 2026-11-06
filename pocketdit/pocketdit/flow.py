"""Flow matching on ligand atom coordinates (Å), with the pocket held fixed.

Path:      x_t = (1 - t) x_0 + t x_1,   x_0 ~ N(pocket centre, prior_std² I),  t ∈ [0, 1] (1 = data)
Model:     predicts x̂_1 = E[x_1 | x_t]; the velocity is v = (x̂_1 - x_t) / (1 - t)
Sampling:  Euler steps x ← x + Δt · v from t = 0 to 1 (the last step lands exactly on x̂_1).

The model is translation-equivariant by construction (the trunk sees only distances, the head only
relative vectors), so absolute coordinates never reach it and the pocket centre enters through this
prior alone. That makes the centre entirely a question of how x_0 is drawn: what has to match
between training and inference is the *distribution of the offset* between the prior centre and the
true ligand centroid. Hence ``pocket_com`` rather than the ligand's centroid, and the same jitter on
both sides.

Fragment mode (``FlowConfig.fragments``) is the same flow on a different state: one rigid
transform ``(R_f, T_f)`` per fragment instead of 3N coordinates, with every fragment's internal
geometry fixed to the template's.

Path:      T_t = (1 - t) T_0 + t T_1,   R_t = exp(t log(R_1 R_0^T)) R_0   (the geodesic)
Prior:     T_0 ~ N(pocket centre, prior_std² I) per fragment,   R_0 ~ uniform on SO(3)
Target:    T_1 the fitted centroid; R_1 the tied solution nearest R_0, gauge pinned against R_0
Model:     predicts (R̂_1, T̂_1); the velocity is the geodesic towards it, divided by (1 - t)
Loss:      |T̂_1 - T_1|² + rot_loss_weight * d(R̂_1, R_1), chordal or geodesic

"Nearest R_0" is ``rigid``'s ``reference=R_t``: R_t lies on a shortest path from R_0 to R_1, and a
target nearest one end of such a path is nearest every point on it (``select_targets``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pocketdit import so3
from pocketdit.model import PocketDiT, assemble, fragment_onehot
from pocketdit.rigid import GAUGE_FULL


@dataclass
class FlowConfig:
    prior_std: float = 4.0  # Å, isotropic Gaussian prior around the pocket centre
    center_noise: float = 1.0  # Å, jitter of the prior centre during training
    sample_center_noise: float = 0.25  # Å, the same jitter at sampling time
    dist_loss_weight: float = 1.0  # auxiliary loss on short intra-ligand distances
    dist_loss_cutoff: float = 3.0  # Å, pairs this close in the true pose (bonds and angles)
    dist_loss_t_min: float = 0.75  # only near the data end, where x̂_1 should be sharp
    num_steps: int = 20  # default number of sampling steps
    # Fragment mode: the state is one rigid transform per fragment (needs a --fragments cache).
    fragments: bool = False
    rot_loss: str = "chordal"  # "chordal" |R̂ - R|²_F = 8 sin²(θ/2), or "geodesic" θ²
    rot_loss_weight: float = 1.0  # chordal ≈ 2θ², so 1.0 weighs a turn like ~1.7 Å of radius of gyration


def sample_prior(batch: dict, std: float, center_noise: float = 0.0) -> torch.Tensor:
    """x_0: an isotropic cloud of independent atoms around the (jittered) pocket centre.

    The centre is ``pocket_com``, a function of the protein and the reference ligand -- not the
    true ligand's centroid, which in re-docking training would be the centroid of the answer.
    The jitter widens the offset the model is trained to correct, so that a pocket centre a few Å
    off at inference is still in distribution. Worth keeping in mind when tuning it: x_0's own
    centroid already concentrates to within ``std / sqrt(n_atoms)`` per axis (0.8 Å for a 26-atom
    ligand at std 4), so the jitter is what actually sets how much translation there is to learn.
    """
    center = batch["pocket_com"]
    if center_noise > 0:
        center = center + center_noise * torch.randn_like(center)
    x0 = center[:, None] + std * torch.randn_like(batch["lig_pos"])
    return x0 * batch["lig_mask"].unsqueeze(-1)


def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    """Pairwise distances that stay differentiable on the diagonal."""
    return ((x[:, :, None] - x[:, None]).pow(2).sum(-1) + 1e-8).sqrt()


def flow_matching_loss(model: PocketDiT, batch: dict, cfg: FlowConfig) -> tuple[torch.Tensor, dict[str, float]]:
    if cfg.fragments:
        return fragment_loss(model, batch, cfg)
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
        "centroid_err": centroid_error(x1_hat, x1, mask).mean().item(),
    }
    return loss, metrics


@torch.no_grad()
def sample(model: PocketDiT, batch: dict, cfg: FlowConfig, num_steps: int | None = None) -> torch.Tensor:
    """Generate ligand poses [B, N, 3] (Å). The pocket is encoded once and reused at every step."""
    if cfg.fragments:
        return sample_fragments(model, batch, cfg, num_steps)
    steps = num_steps or cfg.num_steps
    x = sample_prior(batch, cfg.prior_std, cfg.sample_center_noise)
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


def centroid_error(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-complex distance between two poses' centroids (Å): the translation the model got wrong.

    Worth logging next to the RMSD, because it is the one number that exposes a leaking prior. If
    the prior centre is derived from the true ligand, this collapses towards zero on re-docking
    while the total RMSD still looks respectable -- and then rises sharply in cross-docking, where
    the centre comes from a different ligand. A re-docking value far below the cross-docking one
    means the model is reading the answer out of the prior instead of out of the pocket.
    """
    m = mask.unsqueeze(-1).float()
    n = m.sum(1)
    return (((x * m).sum(1) / n) - ((y * m).sum(1) / n)).norm(dim=-1)


# --------------------------------------------------------------------------------------
# Fragment mode
# --------------------------------------------------------------------------------------
def sample_fragment_prior(batch: dict, std: float, center_noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """(R_0, T_0): uniform rotations, centroids an isotropic cloud around the (jittered) pocket centre.

    The same centre and jitter as ``sample_prior``, now per fragment rather than per atom. Padded
    fragments sit at the identity and the origin. With ~5 fragments rather than ~25 atoms the
    prior's own centroid is spread over ``std / sqrt(F)``, about twice as far.
    """
    mask = batch["frag_mask"]
    center = batch["pocket_com"]
    if center_noise > 0:
        center = center + center_noise * torch.randn_like(center)
    trans = (center[:, None] + std * torch.randn(*mask.shape, 3, device=mask.device)) * mask.unsqueeze(-1)
    rot = so3.random_rotation(*mask.shape, device=mask.device)
    eye = torch.eye(3, device=mask.device).expand_as(rot)
    return torch.where(mask[..., None, None], rot, eye), trans


@torch.no_grad()
def select_targets(batch: dict, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each fragment's target rotation nearest ``reference`` [B, F, 3, 3], and which solution it is.

    The batched ``rigid.fragment_targets(reference=...)``: every tied solution has its gauge pinned
    against the reference, and the nearest one wins, ties to the first. Nearest is largest
    ``tr(reference^T R)``, the geodesic distance being monotone in it.
    """
    cand = batch["frag_rot"]  # [B, F, K, 3, 3]
    ref = reference.unsqueeze(2).expand_as(cand)
    pinned = so3.pin_gauge(cand, batch["frag_axis"].unsqueeze(2), batch["frag_free"].unsqueeze(2), ref)
    score = so3.trace(ref.transpose(-1, -2) @ pinned).masked_fill(~batch["frag_rot_mask"], float("-inf"))
    choice = score.argmax(-1)  # [B, F]
    rot = torch.take_along_dim(pinned, choice[..., None, None, None], dim=2).squeeze(2)
    return rot, choice


def _fragment_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over each complex's fragments where ``mask`` [B, F] is set: [B] (0 where none is)."""
    m = mask.float()
    return (value * m).sum(1) / m.sum(1).clamp(min=1)


def fragment_loss(model: PocketDiT, batch: dict, cfg: FlowConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Translation + rotation flow matching on one rigid transform per fragment.

    Every term is a per-complex mean over its fragments, then a mean over the batch, so a ligand
    with many fragments does not outweigh one with few. Single-atom fragments carry no rotation
    and are left out of the rotation term. The distance loss is kept but only over pairs in
    *different* fragments: inside a fragment every distance is the template's by construction, so
    what is left is the joints -- cut bonds and the angles across them, which nothing else holds
    at bonding distance. Its targets are the assembled target pose, so it never pulls away from the
    reachable optimum.
    """
    fmask, mask = batch["frag_mask"], batch["lig_mask"]
    B = fmask.shape[0]
    t = torch.rand(B, device=fmask.device)
    rot0, trans0 = sample_fragment_prior(batch, cfg.prior_std, cfg.center_noise)
    rot1, _ = select_targets(batch, rot0)
    trans1 = batch["frag_trans"]
    rot_t = so3.geodesic(rot0, rot1, t[:, None].expand_as(fmask))
    trans_t = (1 - t)[:, None, None] * trans0 + t[:, None, None] * trans1
    rot_hat, trans_hat = model.forward_fragments(batch, rot_t, trans_t, t)

    trans = _fragment_mean(((trans_hat - trans1) ** 2).sum(-1), fmask)
    rot_err = so3.chordal(rot_hat, rot1) if cfg.rot_loss == "chordal" else so3.geodesic_sq(rot_hat, rot1)
    rot = _fragment_mean(rot_err, fmask & (batch["frag_free"] != GAUGE_FULL))

    x1, x1_hat = assemble(batch, rot1, trans1), assemble(batch, rot_hat, trans_hat)
    m = mask.float()
    coord = (((x1_hat - x1) ** 2).sum(-1) * m).sum(1) / m.sum(1)
    d_true = _pairwise_dist(x1)
    idx = batch["frag_idx"]
    pairs = (d_true < cfg.dist_loss_cutoff) & mask[:, :, None] & mask[:, None, :] & (idx[:, :, None] != idx[:, None, :])
    d_err = (_pairwise_dist(x1_hat) - d_true) ** 2
    dist = (d_err * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp(min=1)
    dist = dist * (t > cfg.dist_loss_t_min)

    loss = (trans + cfg.rot_loss_weight * rot + cfg.dist_loss_weight * dist).mean()
    metrics = {
        "loss": loss.item(),
        "trans_mse": trans.mean().item(),
        "rot_loss": rot.mean().item(),
        "coord_mse": coord.mean().item(),
        "dist_mse": dist.sum().item() / max(int((t > cfg.dist_loss_t_min).sum()), 1),
        "centroid_err": centroid_error(x1_hat, x1, mask).mean().item(),
    }
    return loss, metrics


@torch.no_grad()
def sample_fragments(
    model: PocketDiT, batch: dict, cfg: FlowConfig, num_steps: int | None = None, return_state: bool = False
):
    """Generate ligand poses [B, N, 3] (Å) by Euler steps on one rigid transform per fragment.

    Each step moves the centroids a fraction ``dt / (1 - t)`` of the way to T̂_1 and turns each
    fragment the same fraction of the shortest way to R̂_1, so the last step lands exactly on the
    prediction. ``return_state`` also returns the final ``(R, T)``.
    """
    steps = num_steps or cfg.num_steps
    rot, trans = sample_fragment_prior(batch, cfg.prior_std, cfg.sample_center_noise)
    h_p = model.encode_pocket(batch)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=trans.device)
    for i in range(steps):
        t = ts[i].expand(trans.shape[0])
        rot_hat, trans_hat = model.forward_fragments(batch, rot, trans, t, h_p=h_p)
        frac = (ts[i + 1] - ts[i]) / (1 - ts[i])
        trans = trans + frac * (trans_hat.float() - trans)
        rot = so3.exp(frac * so3.log(rot_hat.float() @ rot.transpose(-1, -2))) @ rot
    x = assemble(batch, rot, trans)
    return (x, rot, trans) if return_state else x


def pose_rmsd(x: torch.Tensor, batch: dict) -> torch.Tensor:
    """Per-complex RMSD [B] of a fragment-mode pose against the true one, labelling-corrected.

    A pose's atom i is where the template put atom i, but a tied solution may have fitted it onto
    an equivalent true atom instead -- the far ortho carbon of a ring flipped by 180°. Each
    fragment is therefore scored under whichever of its tied labellings (``frag_pairing``) fits
    best; fragments choose independently, so the per-fragment minima add up to the best overall.
    Without this a correctly placed but relabelled ring reads as ~2 Å off on four atoms.
    """
    true, pairing = batch["lig_pos"], batch["frag_pairing"]  # [B, N, 3], [B, K, N]
    b = torch.arange(true.shape[0], device=true.device)[:, None, None]
    err = ((x.unsqueeze(1) - true[b, pairing]) ** 2).sum(-1)  # [B, K, N]
    sse = torch.einsum("bfn,bkn->bfk", fragment_onehot(batch, err.dtype), err)
    sse = sse.masked_fill(~batch["frag_rot_mask"], float("inf")).min(-1).values
    sse = torch.where(batch["frag_mask"], sse, torch.zeros_like(sse))
    return (sse.sum(1) / batch["lig_mask"].sum(1)).sqrt()
