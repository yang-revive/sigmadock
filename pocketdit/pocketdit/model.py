"""Pocket-conditioned geometric DiT.

    pocket atoms ──► PocketEncoder (static, run once per complex) ──► h_p
                                                                      │
    ligand atoms + x_t ──► atom embedding ──► [DiTBlock × depth] ──► GeometricHead ──► x̂_1
                                               │ self-attn  (bias: d_ij(x_t), bond, graph distance)
                                               │ cross-attn (bias: d_ik(x_t, pocket))
                                               │ MLP
                                               └ adaLN-Zero on the flow time t

Symmetry: the trunk only sees distances and chirality volumes (SE(3)-invariant) and the head
outputs a weighted sum of relative vectors, so x̂_1 is SE(3)-equivariant by construction.
No rotation augmentation is needed.

Fragment mode (``forward_fragments``) runs the very same trunk and head on the pose assembled
from one rigid transform per fragment, and turns the per-atom displacements into a force and a
torque per fragment -- the rigid motion that best explains them (``rigid_motion``). The predicted
transform is ``(exp(omega) R_t, T_t + v)``, equivariant for the same reason as x̂_1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pocketdit import so3
from pocketdit.chem import LIG_FEATURE_SIZES, NUM_BOND_TYPES, NUM_SPD, POCKET_FEATURE_SIZES
from pocketdit.rigid import GAUGE_AXIS, GAUGE_FULL

INERTIA_DAMPING = 0.05  # Å², added to every fragment's inertia before it is inverted, see rigid_motion


@dataclass
class ModelConfig:
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    pocket_depth: int = 2
    pair_dim: int = 64
    head_hidden: int = 64  # hidden size of the geometric output head
    num_rbf: int = 32
    lig_rbf_max: float = 20.0  # Å
    cross_rbf_max: float = 24.0  # Å


# --------------------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------------------
class RBF(nn.Module):
    def __init__(self, num: int, d_max: float) -> None:
        super().__init__()
        self.register_buffer("mu", torch.linspace(0.0, d_max, num), persistent=False)
        self.coeff = -0.5 / (d_max / num) ** 2

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.coeff * (d.unsqueeze(-1) - self.mu) ** 2)


class TimeEmbed(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = 1000.0 * t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class Mlp(nn.Sequential):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))


class Mlp2(nn.Sequential):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))


class Attention(nn.Module):
    """Multi-head attention with an additive per-head bias, via ``F.scaled_dot_product_attention``."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, Nq, D = x.shape
        q = self.q(x).view(B, Nq, self.heads, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out(o.transpose(1, 2).reshape(B, Nq, D))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def head_bias(proj: nn.Linear, pair: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
    """[B, Q, K, P] pair features -> [B, H, Q, K] attention bias with padded keys at -inf."""
    return proj(pair).permute(0, 3, 1, 2).masked_fill(key_pad, float("-inf"))


def chirality_feature(x: torch.Tensor, nbrs: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """sign * normalised signed volume of each stereocentre at pose x: [B, N]. > 0 = correct."""
    b = torch.arange(x.shape[0], device=x.device)[:, None, None]
    v = x[b, nbrs] - x.unsqueeze(2)  # [B, N, 3, 3]
    vol = (v[..., 0, :] * torch.linalg.cross(v[..., 1, :], v[..., 2, :])).sum(-1)
    return sign * vol / (v.norm(dim=-1).prod(dim=-1) + 1e-6)


# --------------------------------------------------------------------------------------
# Pocket encoder (static: does not depend on t or the ligand pose)
# --------------------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h)
        h = h + self.attn(x, x, bias)
        return h + self.mlp(self.norm2(h))


class PocketEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, num_rbf: int = 16) -> None:
        super().__init__()
        self.embed = nn.ModuleList(nn.Embedding(n, cfg.dim) for n in POCKET_FEATURE_SIZES)
        self.rbf = RBF(num_rbf, cfg.cross_rbf_max)
        self.bias = nn.ModuleList(nn.Linear(num_rbf, cfg.heads) for _ in range(cfg.pocket_depth))
        self.blocks = nn.ModuleList(EncoderBlock(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.pocket_depth))
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, feat: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = sum(emb(feat[..., i]) for i, emb in enumerate(self.embed))
        rbf = self.rbf(torch.cdist(pos, pos))  # [B, M, M, R]
        key_pad = ~mask[:, None, None, :]
        for blk, proj in zip(self.blocks, self.bias, strict=True):
            h = blk(h, head_bias(proj, rbf, key_pad))
        return self.norm(h)


# --------------------------------------------------------------------------------------
# Ligand trunk
# --------------------------------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, pair_dim: int) -> None:
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.self_attn = Attention(dim, heads)
        self.cross_attn = Attention(dim, heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ll_bias = nn.Linear(pair_dim, heads)
        self.lp_bias = nn.Linear(pair_dim, heads)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, D] ligand tokens
        c: torch.Tensor,  # [B, D] time conditioning
        h_p: torch.Tensor,  # [B, M, D] pocket tokens
        z_ll: torch.Tensor,  # [B, N, N, P]
        z_lp: torch.Tensor,  # [B, N, M, P]
        lig_pad: torch.Tensor,  # [B, 1, 1, N] True = padding
        poc_pad: torch.Tensor,  # [B, 1, 1, M]
    ) -> torch.Tensor:
        s1, a1, g1, s2, a2, g2, s3, a3, g3 = self.ada(c).chunk(9, dim=-1)
        h = modulate(self.norm1(x), s1, a1)
        x = x + g1.unsqueeze(1) * self.self_attn(h, h, head_bias(self.ll_bias, z_ll, lig_pad))
        h = modulate(self.norm2(x), s2, a2)
        x = x + g2.unsqueeze(1) * self.cross_attn(h, h_p, head_bias(self.lp_bias, z_lp, poc_pad))
        return x + g3.unsqueeze(1) * self.mlp(modulate(self.norm3(x), s3, a3))


class GeometricHead(nn.Module):
    """Equivariant update  Δx_i = mean_j w_ij r_ij + mean_k w_ik r_ik  (ligand j, pocket k).

    w are invariant scalars from token and pair features; r are soft unit vectors x_i - x_j.
    """

    def __init__(self, dim: int, pair_dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.q_ll, self.k_ll, self.z_ll = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(pair_dim, hidden)
        self.q_lp, self.k_lp, self.z_lp = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(pair_dim, hidden)
        self.w_ll = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.w_lp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for layer in (self.ada[-1], self.w_ll[-1], self.w_lp[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _soft_unit(v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm(dim=-1, keepdim=True) + 1.0)

    def forward(self, h, c, h_p, z_ll, z_lp, x, x_p, lig_mask, poc_mask) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift, scale)
        w_ll = self.w_ll(self.q_ll(h)[:, :, None] * self.k_ll(h)[:, None] + self.z_ll(z_ll)).squeeze(-1)
        w_lp = self.w_lp(self.q_lp(h)[:, :, None] * self.k_lp(h_p)[:, None] + self.z_lp(z_lp)).squeeze(-1)
        w_ll = w_ll.float() * lig_mask[:, None, :]
        w_lp = w_lp.float() * poc_mask[:, None, :]
        r_ll = self._soft_unit(x[:, :, None] - x[:, None])  # [B, N, N, 3]
        r_lp = self._soft_unit(x[:, :, None] - x_p[:, None])  # [B, N, M, 3]
        n_l = lig_mask.sum(1).clamp(min=1)[:, None, None]
        n_p = poc_mask.sum(1).clamp(min=1)[:, None, None]
        return (w_ll.unsqueeze(-1) * r_ll).sum(2) / n_l + (w_lp.unsqueeze(-1) * r_lp).sum(2) / n_p


# --------------------------------------------------------------------------------------
# Fragment mode: one rigid transform per fragment
# --------------------------------------------------------------------------------------
def fragment_onehot(batch: dict, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[B, F, N], 1 where atom n belongs to fragment f (padding atoms belong to none)."""
    num = batch["frag_free"].shape[1]
    onehot = batch["frag_idx"][:, None, :] == torch.arange(num, device=batch["frag_idx"].device)[None, :, None]
    return (onehot & batch["lig_mask"][:, None, :]).to(dtype)


def _per_atom(batch: dict, per_fragment: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B, F, ...] tensor to the atoms: [B, N, ...]."""
    b = torch.arange(per_fragment.shape[0], device=per_fragment.device)[:, None]
    return per_fragment[b, batch["frag_idx"]]


def assemble(batch: dict, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Atom coordinates [B, N, 3] of the pose with fragment f at ``(rot[f], trans[f])``.

    Atom i goes to ``R_f @ local_i + T_f``: the template fragment, rotated about its own centroid,
    with the centroid put at ``T_f``. Internal geometry is the template's whatever the transforms.
    """
    local = batch["frag_local"].to(rot.dtype)
    x = (_per_atom(batch, rot) @ local.unsqueeze(-1)).squeeze(-1) + _per_atom(batch, trans)
    return x * batch["lig_mask"].unsqueeze(-1)


def rigid_motion(
    delta: torch.Tensor, x: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, batch: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Force and torque per fragment: the rigid motion ``(omega, v)`` closest to the displacements.

    Least squares over ``v + omega x r_i ≈ Δx_i`` (r_i relative to the fragment centroid, which is
    ``trans``) gives ``v = mean Δx_i``, the force, and ``I omega = mean r_i x Δx_i``, the torque, with
    ``I = mean(|r|² 1 - r r^T)`` the fragment's inertia. Solving with the inertia rather than taking
    the torque as it is matters: a ring's in-plane and out-of-plane moments differ by two, so a raw
    torque would turn it at different rates for the same displacements depending on the axis.

    ``INERTIA_DAMPING`` keeps the solve finite where a moment is close to zero, and how small it can
    be is set by the smallest real moment, which is not where one would guess: a bent three-atom
    fragment -- the most common size -- turns about the axis through its two outer atoms with a
    moment of only 0.09-0.18 Å² on the examples (two-atom fragments: 0.43-0.56; rings: 0.6 and up).
    Damping trades fidelity on that axis against the gain of the map, and both ends cost. At
    0.05 Å² a perfectly rigid displacement field comes back shrunk by at most 36% about the weak
    axis and by 5% in the median (0.1: 53% / 9%). At 0.02 (18% / 2%) the gain is high enough that
    one of five single-complex overfits stalled at 2.1 Å, where 0.05 and 0.1 all reached 0.3-0.8 Å.

    The gauge is removed explicitly. A collinear fragment's torque has no component along its own
    axis in exact arithmetic, and the target never asks for one (it is pinned to the nearest member
    of the gauge circle, whose relative rotation is perpendicular to the axis); projecting it out
    keeps a nearly collinear fragment from amplifying rounding through its tiny axial moment. A
    single atom has no rotation at all.
    """
    onehot = fragment_onehot(batch, delta.dtype)
    count = onehot.sum(-1).clamp(min=1)[..., None]  # [B, F, 1]
    force = onehot @ delta / count
    r = x - _per_atom(batch, trans)
    torque = onehot @ torch.linalg.cross(r, delta) / count
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    outer = (r * r).sum(-1)[..., None, None] * eye - r.unsqueeze(-1) * r.unsqueeze(-2)  # [B, N, 3, 3]
    inertia = torch.einsum("bfn,bnij->bfij", onehot, outer) / count[..., None]
    omega = torch.linalg.solve(inertia + INERTIA_DAMPING * eye, torque.unsqueeze(-1)).squeeze(-1)

    kind = batch["frag_free"]
    axis = (rot @ batch["frag_axis"].to(rot.dtype).unsqueeze(-1)).squeeze(-1)  # the free axis, world frame
    omega = torch.where((kind == GAUGE_AXIS)[..., None], omega - (omega * axis).sum(-1, keepdim=True) * axis, omega)
    omega = torch.where((kind == GAUGE_FULL)[..., None], torch.zeros_like(omega), omega)
    return omega, force


class PocketDiT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pocket_encoder = PocketEncoder(cfg)
        self.lig_embed = nn.ModuleList(nn.Embedding(n, cfg.dim) for n in LIG_FEATURE_SIZES)
        self.chiral_embed = nn.Sequential(nn.Linear(1, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.time_embed = TimeEmbed(cfg.dim)
        # pair features
        self.ll_rbf = RBF(cfg.num_rbf, cfg.lig_rbf_max)
        self.lp_rbf = RBF(cfg.num_rbf, cfg.cross_rbf_max)
        self.ll_dist = Mlp2(cfg.num_rbf, cfg.pair_dim)
        self.lp_dist = Mlp2(cfg.num_rbf, cfg.pair_dim)
        self.bond_embed = nn.Embedding(NUM_BOND_TYPES, cfg.pair_dim)
        self.spd_embed = nn.Embedding(NUM_SPD, cfg.pair_dim)
        self.blocks = nn.ModuleList(DiTBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.pair_dim) for _ in range(cfg.depth))
        self.head = GeometricHead(cfg.dim, cfg.pair_dim, cfg.head_hidden)

    def encode_pocket(self, batch: dict) -> torch.Tensor:
        return self.pocket_encoder(batch["poc_feat"], batch["poc_pos"], batch["poc_mask"])

    def forward(self, batch: dict, x_t: torch.Tensor, t: torch.Tensor, h_p: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the clean ligand pose x̂_1 [B, N, 3] (Å) from the noisy pose x_t at time t [B]."""
        return (x_t + self.displacement(batch, x_t, t, h_p)) * batch["lig_mask"].unsqueeze(-1)

    def forward_fragments(
        self, batch: dict, rot: torch.Tensor, trans: torch.Tensor, t: torch.Tensor, h_p: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the clean transform of every fragment from the current ones at time t [B].

        ``rot`` [B, F, 3, 3] and ``trans`` [B, F, 3] (Å) place each fragment's template, rotated about
        its centroid and with the centroid put at ``trans``. Returns ``(R̂_1, T̂_1)`` in the same form.
        The trunk and head are the all-atom ones, run on the assembled pose; only the read-out of
        their per-atom displacements differs (``rigid_motion``). Zero-initialised, it returns its
        input, like ``forward``.
        """
        x_t = assemble(batch, rot, trans)
        delta = self.displacement(batch, x_t, t, h_p)
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            rot, trans = rot.float(), trans.float()
            omega, force = rigid_motion(delta.float(), x_t.float(), rot, trans, batch)
            return so3.exp(omega) @ rot, trans + force

    def displacement(
        self, batch: dict, x_t: torch.Tensor, t: torch.Tensor, h_p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The head's per-atom equivariant displacement Δx [B, N, 3] at pose x_t and time t [B]."""
        if h_p is None:
            h_p = self.encode_pocket(batch)
        lig_mask, poc_mask, x_p = batch["lig_mask"], batch["poc_mask"], batch["poc_pos"]

        h = sum(emb(batch["lig_feat"][..., i]) for i, emb in enumerate(self.lig_embed))
        chi = chirality_feature(x_t, batch["chiral_nbrs"], batch["chiral_sign"])
        h = h + self.chiral_embed(chi.unsqueeze(-1))
        c = self.time_embed(t)

        z_ll = self.ll_dist(self.ll_rbf(torch.cdist(x_t, x_t)))
        z_ll = z_ll + self.bond_embed(batch["lig_bond"]) + self.spd_embed(batch["lig_spd"])
        z_lp = self.lp_dist(self.lp_rbf(torch.cdist(x_t, x_p)))

        lig_pad, poc_pad = ~lig_mask[:, None, None, :], ~poc_mask[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, c, h_p, z_ll, z_lp, lig_pad, poc_pad)

        return self.head(h, c, h_p, z_ll, z_lp, x_t, x_p, lig_mask, poc_mask)
