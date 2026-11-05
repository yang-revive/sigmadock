import pytest
import torch
from torch import nn

from conftest import DUMMY, load_item, tiny_config
from pocketdit.data import PocketCrop, collate
from pocketdit.flow import FlowConfig, centroid_error, flow_matching_loss, rmsd, sample, sample_prior
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
    assert set(metrics) == {"loss", "coord_mse", "dist_mse", "centroid_err"}


def test_prior_is_centred_on_the_pocket_not_the_ligand(items):
    """The prior must not hand the model the answer's centroid. See flow.sample_prior."""
    batch = collate(items * 400)
    torch.manual_seed(0)
    x0 = sample_prior(batch, std=FlowConfig().prior_std, center_noise=0.0)
    mask = batch["lig_mask"].unsqueeze(-1).float()
    mean = (x0 * mask).sum(1) / mask.sum(1)
    # Averaged over many draws the prior sits on pocket_com, which is not the ligand's centroid.
    for b in range(len(items)):
        draws = mean[b :: len(items)]
        assert torch.allclose(draws.mean(0), batch["pocket_com"][b], atol=0.2)
        assert (batch["pocket_com"][b] - batch["lig_pos"][b][batch["lig_mask"][b]].mean(0)).norm() > 0.5


def test_centre_noise_widens_the_offset_the_model_must_correct(items):
    batch = collate(items * 200)
    cfg = FlowConfig()
    torch.manual_seed(0)
    quiet = centroid_error(sample_prior(batch, cfg.prior_std, 0.0), batch["lig_pos"], batch["lig_mask"])
    noisy = centroid_error(sample_prior(batch, cfg.prior_std, cfg.center_noise), batch["lig_pos"], batch["lig_mask"])
    assert noisy.mean() > quiet.mean()
    # Sampling keeps the same jitter on, just a smaller one, so x_0 stays in distribution.
    assert 0 < cfg.sample_center_noise <= cfg.center_noise


def test_centroid_error_ignores_padding(items):
    batch = collate(items)
    x = batch["lig_pos"].clone()
    shift = torch.tensor([0.0, 3.0, 4.0])
    x[batch["lig_mask"]] += shift
    assert torch.allclose(centroid_error(x, batch["lig_pos"], batch["lig_mask"]), torch.full((2,), 5.0), atol=1e-4)
    x[~batch["lig_mask"]] += 100.0  # padded rows must not leak in
    assert torch.allclose(centroid_error(x, batch["lig_pos"], batch["lig_mask"]), torch.full((2,), 5.0), atol=1e-4)


def test_overfit_single_complex():
    """A tiny model must learn to dock one complex from noise (sanity check of the whole stack)."""
    if not DUMMY.exists():
        pytest.skip("example data not found")
    torch.manual_seed(0)
    batch = collate([load_item("1U1C_BAU", crop=PocketCrop(max_atoms=96, cutoff=5.0))] * 8)
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
