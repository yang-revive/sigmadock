import math

import torch

from conftest import tiny_config
from pocketdit.data import collate
from pocketdit.model import PocketDiT


def random_rotation(seed: int) -> torch.Tensor:
    q = torch.nn.functional.normalize(torch.randn(4, generator=torch.Generator().manual_seed(seed)), dim=0)
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def noisy_pose(batch: dict, t: float = 0.5) -> torch.Tensor:
    torch.manual_seed(1)
    noise = batch["pocket_com"][:, None] + 4.0 * torch.randn_like(batch["lig_pos"])
    return ((1 - t) * noise + t * batch["lig_pos"]) * batch["lig_mask"][..., None]


def test_zero_init_returns_input(items):
    model = PocketDiT(tiny_config()).eval()
    batch = collate(items)
    x_t = noisy_pose(batch)
    with torch.no_grad():
        assert torch.allclose(model(batch, x_t, torch.full((2,), 0.3)), x_t)


def test_se3_equivariance(items, random_model):
    batch = collate(items)
    x_t, t = noisy_pose(batch), torch.tensor([0.2, 0.7])
    R, shift = random_rotation(0), torch.tensor([3.0, -7.0, 11.0])
    moved = dict(batch, poc_pos=batch["poc_pos"] @ R.T + shift, pocket_com=batch["pocket_com"] @ R.T + shift)
    with torch.no_grad():
        out = random_model(batch, x_t, t)
        out_moved = random_model(moved, x_t @ R.T + shift, t)
    mask = batch["lig_mask"]
    assert out.abs().max() > 1.0  # the randomised model really moves atoms
    assert torch.allclose(out_moved[mask], out[mask] @ R.T + shift, atol=1e-3)


def test_reflection_changes_output_for_chiral_ligand(items, random_model):
    batch = collate(items[:1])  # has stereocentres
    x_t, t = noisy_pose(batch), torch.tensor([0.5])
    mirrored = dict(batch, poc_pos=-batch["poc_pos"])
    with torch.no_grad():
        out = random_model(batch, x_t, t)
        out_mirror = random_model(mirrored, -x_t, t)
    assert not torch.allclose(out_mirror, -out, atol=1e-2)  # chirality makes it SE(3), not E(3)


def test_padding_invariance(items, random_model):
    alone = collate(items[1:])  # the smaller complex
    padded = collate(items[::-1])  # smaller complex first, padded to the larger one
    n = int(alone["lig_mask"].sum())
    x_t = noisy_pose(padded)
    with torch.no_grad():
        out_alone = random_model(alone, x_t[:1, :n], torch.tensor([0.4]))
        out_padded = random_model(padded, x_t, torch.tensor([0.4, 0.9]))
    assert torch.allclose(out_alone[0], out_padded[0, :n], atol=1e-4)
    assert math.isfinite(out_padded.abs().sum().item())
