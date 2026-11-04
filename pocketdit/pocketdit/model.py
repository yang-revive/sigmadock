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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pocketdit.chem import LIG_FEATURE_SIZES, NUM_BOND_TYPES, NUM_SPD, POCKET_FEATURE_SIZES


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

        delta = self.head(h, c, h_p, z_ll, z_lp, x_t, x_p, lig_mask, poc_mask)
        return (x_t + delta) * lig_mask.unsqueeze(-1)
