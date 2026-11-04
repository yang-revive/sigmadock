import pytest
import torch
from torch import nn

from conftest import DUMMY, load_item, tiny_config
from pocketdit.data import collate
from pocketdit.flow import FlowConfig, flow_matching_loss, rmsd, sample
from pocketdit.model import PocketDiT


class Oracle(nn.Module):
    """Always predicts the true pose: the sampler must then land exactly on it."""

    def encode_pocket(self, batch):
        return None

    def forward(self, batch, x_t, t, h_p=None):
        return batch["lig_pos"]


def test_sampler_integrates_to_prediction(items):
    batch = collate(items)
    x = sample(Oracle(), batch, FlowConfig(), num_steps=7)
    assert torch.allclose(x, batch["lig_pos"], atol=1e-4)


def test_loss_and_backward(items):
    torch.manual_seed(0)
    model = PocketDiT(tiny_config())
    loss, metrics = flow_matching_loss(model, collate(items), FlowConfig())
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert set(metrics) == {"loss", "coord_mse", "dist_mse"}


def test_overfit_single_complex():
    """A tiny model must learn to dock one complex from noise (sanity check of the whole stack)."""
    if not DUMMY.exists():
        pytest.skip("example data not found")
    torch.manual_seed(0)
    batch = collate([load_item("1U1C_BAU", pocket_cutoff=5.0)] * 8)
    model = PocketDiT(tiny_config())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cfg = FlowConfig()
    for _ in range(600):  # ~40 s on 4 CPU cores; medians 0.3-0.9 Å over seeds 0-2
        loss, _ = flow_matching_loss(model, batch, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    r = rmsd(sample(model, batch, cfg), batch["lig_pos"], batch["lig_mask"])
    assert r.median() < 1.5, r
