"""Rotations in torch, batched over any leading dimensions.

The fragment flow moves one rotation per fragment, so it needs the exponential and logarithm of
SO(3), a uniform prior and the geodesic between two rotations -- all batched, and all usable inside
the training step. ``rigid`` has the numpy counterparts used to build the targets; the conventions
are the same: column rotations ``R @ x``, which is ``x @ R.T`` on row vectors, and rotation vectors
``omega = angle * axis``.

Two places need care. ``exp`` sits in the model's forward pass (the predicted rotation is
``exp(omega) @ R_t``), so it has to be smooth with a finite gradient at ``omega = 0``, which is
exactly where a zero-initialised head starts. ``log`` is applied to the relative rotation between a
uniform prior draw and a target, and uniform rotations put a lot of mass near pi -- the density of
the angle is proportional to ``1 - cos(angle)`` -- where ``sin(angle)`` vanishes and the usual
``angle / sin(angle) * vee(R - R.T) / 2`` is all rounding error. There the axis is read off the
symmetric part instead.
"""

from __future__ import annotations

import torch

from pocketdit.rigid import GAUGE_AXIS, GAUGE_FULL

_SMALL = 1e-6  # angle² below which the Taylor expansions are used
_NEAR_PI = 0.5  # sin(angle) below which, with cos(angle) < 0, log reads the axis off the symmetric part


def skew(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] with ``skew(a) @ b == cross(a, b)``."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).unflatten(-1, (3, 3))


def vee(m: torch.Tensor) -> torch.Tensor:
    """Inverse of ``skew`` on the antisymmetric part: [..., 3, 3] -> [..., 3]."""
    return torch.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1) / 2


def trace(m: torch.Tensor) -> torch.Tensor:
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation vector [..., 3] -> rotation [..., 3, 3], smooth through zero."""
    theta2 = (omega * omega).sum(-1)[..., None, None]
    small = theta2 < _SMALL
    safe2 = torch.where(small, torch.ones_like(theta2), theta2)
    safe = safe2.sqrt()
    a = torch.where(small, 1 - theta2 / 6, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(safe)) / safe2)
    k = skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye + a * k + b * (k @ k)


def angle(rot: torch.Tensor) -> torch.Tensor:
    """Geodesic distance from the identity, in radians: [..., 3, 3] -> [...]."""
    cos = ((trace(rot) - 1) / 2).clamp(-1.0, 1.0)
    sin = vee(rot).norm(dim=-1)
    return torch.atan2(sin, cos)


def log(rot: torch.Tensor) -> torch.Tensor:
    """Rotation [..., 3, 3] -> rotation vector [..., 3] with angle in [0, pi]. Not for gradients."""
    cos = ((trace(rot) - 1) / 2).clamp(-1.0, 1.0)
    v = vee(rot)  # sin(angle) * axis
    sin = v.norm(dim=-1)
    theta = torch.atan2(sin, cos)
    scale = torch.where(sin < 1e-6, 1 + theta * theta / 6, theta / sin.clamp(min=1e-12))
    regular = v * scale[..., None]

    # Near pi: (R + R.T) / 2 = cos I + (1 - cos) n n.T, so the largest-diagonal column of the
    # symmetric remainder is n up to scale. Its sign comes from v, which still points along n while
    # sin > 0; at exactly pi either sign is the same rotation.
    eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
    sym = (rot + rot.transpose(-1, -2)) / 2 - cos[..., None, None] * eye
    col = sym.diagonal(dim1=-2, dim2=-1).argmax(-1)
    n = torch.take_along_dim(sym, col[..., None, None].expand(*col.shape, 3, 1), dim=-1).squeeze(-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    n = torch.where(((n * v).sum(-1, keepdim=True) < 0), -n, n)
    near_pi = (sin < _NEAR_PI) & (cos < 0)
    return torch.where(near_pi[..., None], n * theta[..., None], regular)


def random_rotation(*shape: int, device=None, dtype=torch.float32, generator=None) -> torch.Tensor:
    """Uniform rotations [*shape, 3, 3], from normalised Gaussian quaternions."""
    q = torch.randn(*shape, 4, device=device, dtype=dtype, generator=generator)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))  # fmt: skip


def geodesic(r0: torch.Tensor, r1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The point a fraction ``t`` [...] of the way along the shortest rotation from ``r0`` to ``r1``.

    In the world frame, ``exp(t log(r1 r0^T)) r0``; the angle to ``r0`` grows linearly in ``t``.
    """
    return exp(t[..., None] * log(r1 @ r0.transpose(-1, -2))) @ r0


def chordal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance ``|a - b|²`` = ``8 sin²(angle / 2)``: 2 angle² when small, 8 at most."""
    return ((a - b) ** 2).sum((-1, -2))


def geodesic_sq(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Squared geodesic distance in radians², with a finite gradient where the two coincide."""
    rel = a.transpose(-1, -2) @ b
    cos = ((trace(rel) - 1) / 2).clamp(-1.0, 1.0)
    sin = ((vee(rel) ** 2).sum(-1) + eps).sqrt()
    return torch.atan2(sin, cos) ** 2


def pin_gauge(rot: torch.Tensor, axis: torch.Tensor, kind: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Batched ``rigid.pin_gauge``: each rotation's gauge-family member nearest ``reference``.

    ``axis`` [..., 3] is the free axis in the template frame and ``kind`` [...] the gauge kind from
    ``rigid.gauge_of``; ``rot`` and ``reference`` are [..., 3, 3]. A collinear fragment's family is
    ``rot @ R_axis(phi)``, nearest at ``phi = atan2(tr(Q K), -tr(Q K²))`` with ``Q = reference^T rot``
    and ``K = skew(axis)``; a single atom takes the reference outright; anything else is unchanged.
    """
    k = skew(axis)
    q = reference.transpose(-1, -2) @ rot
    phi = torch.atan2(trace(q @ k), -trace(q @ k @ k))
    pinned = rot @ exp(axis * phi[..., None])
    out = torch.where((kind == GAUGE_AXIS)[..., None, None], pinned, rot)
    return torch.where((kind == GAUGE_FULL)[..., None, None], reference, out)
