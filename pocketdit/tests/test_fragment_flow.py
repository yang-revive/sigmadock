"""Fragment mode: one rigid transform per fragment instead of 3N coordinates.

What is checked here is what makes the swap safe. The rotation maps are accurate where the flow
actually uses them (near pi, for a uniform prior; at zero, for a zero-initialised head). The batched
target the flow trains on is exactly ``rigid.fragment_targets`` with ``reference=R_t``. The read-out
from per-atom displacements to a force and torque keeps the model SE(3)-equivariant, keeps fragments
rigid, and never turns a fragment about the axis its gauge leaves free. And the whole stack still
learns to dock a complex from noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from conftest import DUMMY, load_item, tiny_config
from pocketdit import model as model_module
from pocketdit import so3
from pocketdit.chem import featurize_fragments, load_ligand
from pocketdit.data import PocketCrop, collate, fragment_template
from pocketdit.flow import (
    FlowConfig,
    flow_matching_loss,
    pose_rmsd,
    rmsd,
    sample,
    sample_fragment_prior,
    sample_fragments,
    select_targets,
)
from pocketdit.fragment import fragment
from pocketdit.model import PocketDiT, assemble, rigid_motion
from pocketdit.rigid import GAUGE_AXIS, GAUGE_FULL, GAUGE_NONE, embed_template, fragment_targets, gauge_of, pin_gauge

#: 1R1H_BIR: 7 fragments, two of them with a tied relabelling, four stereocentres.
#: 1U1C_BAU: a collinear two-atom fragment and a ring with a 180° flip. 1S3V_TQD: three gauge fragments.
NAMES = ("1R1H_BIR", "1U1C_BAU", "1S3V_TQD")


def fragment_case(name: str, crop: PocketCrop | None = None) -> tuple[dict, tuple]:
    """A cropped complex with ETKDG-template fragment targets, and what ``rigid`` needs to rebuild them."""
    item = load_item(name, crop=crop)
    mol = load_ligand(next((DUMMY / name).glob("*_ligand.sdf")))
    frag, template = fragment(mol), embed_template(mol)
    item.update(featurize_fragments(mol, frag, template, mol.GetConformer().GetPositions()))
    return item, (mol, frag, template)


@pytest.fixture(scope="module")
def cases() -> list[tuple[dict, tuple]]:
    if not DUMMY.exists():
        pytest.skip(f"example data not found at {DUMMY}")
    return [fragment_case(name) for name in NAMES]


@pytest.fixture(scope="module")
def batch(cases) -> dict:
    return collate([item for item, _ in cases])


def as_double(batch: dict) -> dict:
    return {k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in batch.items()}


def random_state(batch: dict, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return sample_fragment_prior(batch, std=4.0)


# --------------------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------------------
def test_exp_and_log_are_inverse_everywhere_the_flow_goes() -> None:
    """Uniform rotations put real mass near pi, where the textbook log is all rounding error."""
    torch.manual_seed(0)
    rot = so3.random_rotation(20000)
    assert (rot @ rot.transpose(-1, -2) - torch.eye(3)).abs().max() < 1e-5
    assert torch.linalg.det(rot).min() > 0.999
    assert (so3.exp(so3.log(rot)) - rot).abs().max() < 1e-5
    axis = torch.nn.functional.normalize(torch.randn(500, 3), dim=-1)
    for theta in (0.0, 1e-5, 0.3, 2.0, math.pi - 1e-2, math.pi - 1e-4, math.pi):
        rot = so3.exp(axis * theta)
        back = so3.log(rot)
        assert (back.norm(dim=-1) - theta).abs().max() < 1e-5, theta
        assert (so3.exp(back) - rot).abs().max() < 1e-5, theta


def test_exp_is_smooth_at_zero_where_the_head_starts() -> None:
    omega = torch.zeros(4, 3, requires_grad=True)
    so3.exp(omega).sum().backward()
    assert torch.isfinite(omega.grad).all()
    small = torch.tensor([[1e-4, -2e-4, 3e-4]], dtype=torch.float64)
    assert torch.allclose(so3.exp(small), torch.linalg.matrix_exp(so3.skew(small)), atol=1e-12)
    big = torch.tensor([[0.4, -1.2, 0.9]], dtype=torch.float64)
    assert torch.allclose(so3.exp(big), torch.linalg.matrix_exp(so3.skew(big)), atol=1e-12)


def test_the_geodesic_turns_at_constant_speed() -> None:
    torch.manual_seed(1)
    r0, r1 = so3.random_rotation(256, dtype=torch.float64), so3.random_rotation(256, dtype=torch.float64)
    total = so3.angle(r1 @ r0.transpose(-1, -2))
    for t in (0.0, 0.25, 0.7, 1.0):
        rt = so3.geodesic(r0, r1, torch.full((256,), t, dtype=torch.float64))
        assert torch.allclose(so3.angle(rt @ r0.transpose(-1, -2)), t * total, atol=1e-6)
        assert torch.allclose(so3.angle(r1 @ rt.transpose(-1, -2)), (1 - t) * total, atol=1e-6)


def test_the_rotation_losses_agree_near_zero_and_are_differentiable_there() -> None:
    torch.manual_seed(2)
    rot = so3.random_rotation(8, dtype=torch.float64)
    near = so3.exp(1e-3 * torch.randn(8, 3, dtype=torch.float64)) @ rot
    assert torch.allclose(so3.chordal(near, rot), 2 * so3.geodesic_sq(near, rot), rtol=1e-3)
    omega = torch.zeros(8, 3, dtype=torch.float64, requires_grad=True)
    (so3.geodesic_sq(so3.exp(omega) @ rot, rot).sum() + so3.chordal(so3.exp(omega) @ rot, rot).sum()).backward()
    assert torch.isfinite(omega.grad).all()


def test_batched_gauge_pinning_matches_rigid() -> None:
    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    templates = {
        GAUGE_AXIS: np.stack([-0.7 * axis, 0.7 * axis]),
        GAUGE_FULL: np.zeros((1, 3)),
        GAUGE_NONE: rng.normal(size=(5, 3)),
    }
    for expected, template in templates.items():
        kind, free = gauge_of(template - template.mean(0))
        assert kind == expected
        for seed in range(4):
            rot, ref = so3.random_rotation(2, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
            want, _ = pin_gauge(rot.numpy(), template - template.mean(0), ref.numpy())
            got = so3.pin_gauge(rot, torch.from_numpy(free), torch.tensor(kind), ref)
            assert np.allclose(got.numpy(), want, atol=1e-12)


# --------------------------------------------------------------------------------------
# The target the flow trains on
# --------------------------------------------------------------------------------------
def test_the_cached_target_is_the_aligned_template(cases, batch) -> None:
    """Against the identity, the batched choice is ``fragment_targets``' own default target, and
    the labelling-corrected RMSD of it is exactly the floor ``rigid`` reports."""
    eye = torch.eye(3, dtype=torch.float64).expand(*batch["frag_mask"].shape, 3, 3)
    b64 = as_double(batch)
    rot, _ = select_targets(b64, eye)
    x = assemble(b64, rot, b64["frag_trans"])
    floor = pose_rmsd(x, b64)
    for b, (_, (mol, frag, template)) in enumerate(cases):
        true = mol.GetConformer().GetPositions()
        tgt = fragment_targets(mol, frag, true, template)
        n = mol.GetNumAtoms()
        assert np.allclose(x[b, :n].numpy(), tgt.pos, atol=1e-5)
        assert abs(float(floor[b]) - tgt.total_rmsd(true)) < 1e-5


def test_the_batched_choice_is_rigid_with_reference_r_t(cases, batch) -> None:
    """Any reference, one per fragment: ``select_targets`` is ``fragment_targets(reference=...)``."""
    b64 = as_double(batch)
    for seed in range(3):
        ref = so3.random_rotation(
            *batch["frag_mask"].shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
        )
        rot, choice = select_targets(b64, ref)
        for b, (_, (mol, frag, template)) in enumerate(cases):
            num = frag.num_fragments
            tgt = fragment_targets(
                mol, frag, mol.GetConformer().GetPositions(), template, reference=ref[b, :num].numpy()
            )
            assert np.allclose(rot[b, :num].numpy(), tgt.rot, atol=1e-6)
            n = mol.GetNumAtoms()
            pairing = batch["frag_pairing"][
                b, choice[b, batch["frag_idx"][b]], torch.arange(batch["lig_mask"].shape[1])
            ]
            assert np.array_equal(pairing[:n].numpy(), tgt.pairing)


def test_the_target_nearest_the_prior_is_nearest_along_the_whole_path(batch) -> None:
    """Training picks the target against R_0 because R_t needs the target first. That is the same
    choice: R_t is on a shortest path from R_0 to R_1, so nothing equivalent is nearer to it."""
    rot0, _ = random_state(batch, seed=3)
    rot1, choice0 = select_targets(batch, rot0)
    for t in (0.1, 0.5, 0.9):
        rot_t = so3.geodesic(rot0, rot1, torch.full(batch["frag_mask"].shape, t))
        again, choice_t = select_targets(batch, rot_t)
        assert torch.equal(choice_t[batch["frag_mask"]], choice0[batch["frag_mask"]])
        assert torch.allclose(again, rot1, atol=1e-4)


def test_symmetric_fragments_really_have_a_choice(batch) -> None:
    """The examples include tied solutions; otherwise the tests above would not exercise them."""
    tied = batch["frag_rot_mask"].sum(-1)
    assert int(tied.max()) >= 2
    rot0, _ = random_state(batch, seed=4)
    _, choice = select_targets(batch, rot0)
    assert bool((choice > 0).any())  # random references do pick the non-first solution


def test_pose_rmsd_forgives_a_tied_relabelling(batch) -> None:
    """A ring placed by its flipped solution is in the right place with the wrong labels; the plain
    RMSD says it is far off, the labelling-corrected one that it is as good as the target."""
    flipped = torch.where(
        batch["frag_rot_mask"][..., 1, None, None], batch["frag_rot"][:, :, 1], batch["frag_rot"][:, :, 0]
    )
    eye = torch.eye(3).expand_as(flipped)
    target = assemble(batch, select_targets(batch, eye)[0], batch["frag_trans"])
    x = assemble(batch, flipped, batch["frag_trans"])
    has_tie = batch["frag_rot_mask"][..., 1].any(-1)
    assert has_tie.sum() >= 2
    plain = rmsd(x, batch["lig_pos"], batch["lig_mask"])
    corrected = pose_rmsd(x, batch)
    assert (plain[has_tie] > corrected[has_tie] + 0.3).all(), (plain, corrected)
    assert (corrected < pose_rmsd(target, batch) + 0.1).all()
    assert torch.allclose(plain[~has_tie], corrected[~has_tie])  # nothing to forgive without a tie


def test_fragment_template_options() -> None:
    if not DUMMY.exists():
        pytest.skip("example data not found")
    ligand = next((DUMMY / "1U1C_BAU").glob("*_ligand.sdf"))
    mol = load_ligand(ligand)
    true = mol.GetConformer().GetPositions()
    assert np.allclose(fragment_template(mol, ligand, "crystal"), true)
    start = fragment_template(mol, ligand, "*_ligand_start_conf.sdf")
    assert start.shape == true.shape and not np.allclose(start, true)
    with pytest.raises(ValueError, match="no template"):
        fragment_template(mol, ligand, "*_does_not_exist.sdf")


# --------------------------------------------------------------------------------------
# The model: force and torque from per-atom displacements
# --------------------------------------------------------------------------------------
def test_zero_init_returns_the_input_transforms(batch) -> None:
    model = PocketDiT(tiny_config()).eval()
    rot, trans = random_state(batch)
    with torch.no_grad():
        rot_hat, trans_hat = model.forward_fragments(batch, rot, trans, torch.full((3,), 0.3))
    assert torch.allclose(rot_hat, rot, atol=1e-6)
    assert torch.allclose(trans_hat, trans, atol=1e-6)


def test_fragment_mode_is_se3_equivariant(batch, random_model) -> None:
    rot, trans = random_state(batch)
    t = torch.tensor([0.2, 0.5, 0.8])
    q = so3.random_rotation(generator=torch.Generator().manual_seed(5))
    shift = torch.tensor([3.0, -7.0, 11.0])
    moved = dict(batch, poc_pos=batch["poc_pos"] @ q.T + shift, pocket_com=batch["pocket_com"] @ q.T + shift)
    with torch.no_grad():
        rot_hat, trans_hat = random_model.forward_fragments(batch, rot, trans, t)
        rot_m, trans_m = random_model.forward_fragments(moved, q @ rot, trans @ q.T + shift, t)
    m = batch["frag_mask"]
    assert (trans_hat - trans)[m].norm(dim=-1).max() > 0.5  # the randomised model really moves them
    assert so3.angle(rot_hat @ rot.transpose(-1, -2))[m].max() > 0.1
    assert torch.allclose(trans_m[m], trans_hat[m] @ q.T + shift, atol=1e-3)
    assert torch.allclose(rot_m[m], (q @ rot_hat)[m], atol=1e-4)


def test_predicted_fragments_stay_rigid(batch, random_model) -> None:
    rot, trans = random_state(batch)
    with torch.no_grad():
        rot_hat, trans_hat = random_model.forward_fragments(batch, rot, trans, torch.full((3,), 0.5))
    x = assemble(batch, rot_hat, trans_hat)
    template = batch["frag_local"]
    same = batch["frag_idx"][:, :, None] == batch["frag_idx"][:, None, :]
    pairs = same & batch["lig_mask"][:, :, None] & batch["lig_mask"][:, None, :]
    dist = lambda y: (y[:, :, None] - y[:, None]).norm(dim=-1)  # noqa: E731 - cdist's matmul path is ~0.01 Å off
    assert torch.allclose(dist(x)[pairs], dist(template)[pairs], atol=1e-4)


def test_fragment_padding_invariance(cases, random_model) -> None:
    alone = collate([cases[1][0]])  # the smaller ligand
    padded = collate([cases[1][0], cases[0][0]])  # padded to the larger one, in atoms, fragments and K
    rot, trans = random_state(padded)
    num = int(alone["frag_mask"].sum())
    with torch.no_grad():
        r_alone, t_alone = random_model.forward_fragments(alone, rot[:1, :num], trans[:1, :num], torch.tensor([0.4]))
        r_pad, t_pad = random_model.forward_fragments(padded, rot, trans, torch.tensor([0.4, 0.9]))
    assert torch.allclose(r_alone[0], r_pad[0, :num], atol=1e-5)
    assert torch.allclose(t_alone[0], t_pad[0, :num], atol=1e-4)
    assert torch.allclose(r_pad[0, num:], rot[0, num:])  # padded fragments are left alone


def test_a_rigid_displacement_field_comes_back_as_its_force_and_torque(batch, monkeypatch) -> None:
    """The read-out is the least-squares rigid motion: with no damping it is exact on any fragment
    that pins its rotation, and the damping only shrinks it -- most on the weak axis of a bent
    three-atom fragment, by at most 36% there."""
    b64 = as_double(batch)
    rot, trans = (s.double() for s in random_state(batch, seed=6))
    x = assemble(b64, rot, trans)
    torch.manual_seed(7)
    omega, v = 0.3 * torch.randn_like(trans), torch.randn_like(trans)
    r = x - trans[torch.arange(3)[:, None], batch["frag_idx"]]
    per_atom = lambda y: y[torch.arange(3)[:, None], batch["frag_idx"]]  # noqa: E731
    delta = per_atom(v) + torch.linalg.cross(per_atom(omega), r)
    rigid = batch["frag_mask"] & (batch["frag_free"] == GAUGE_NONE)

    got_omega, got_v = rigid_motion(delta, x, rot, trans, b64)
    assert torch.allclose(got_v[batch["frag_mask"]], v[batch["frag_mask"]], atol=1e-9)
    ratio = (got_omega * omega).sum(-1)[rigid] / (omega * omega).sum(-1)[rigid]
    assert ((ratio > 0.64) & (ratio <= 1.0)).all(), ratio
    assert ratio.median() > 0.9

    # Next to no damping (exactly none is singular on padded and single-atom fragments).
    monkeypatch.setattr(model_module, "INERTIA_DAMPING", 1e-12)
    exact, _ = rigid_motion(delta, x, rot, trans, b64)
    assert torch.allclose(exact[rigid], omega[rigid], atol=1e-8)


def test_the_gauge_is_never_turned(batch) -> None:
    """Whatever the displacements, a collinear fragment is not turned about its own axis, and a
    single atom is not turned at all -- the same convention the pinned target uses."""
    torch.manual_seed(8)
    rot, trans = random_state(batch, seed=8)
    x = assemble(batch, rot, trans)
    omega, _ = rigid_motion(torch.randn_like(x), x, rot, trans, batch)
    axis = (rot @ batch["frag_axis"].unsqueeze(-1)).squeeze(-1)
    gauge = batch["frag_mask"] & (batch["frag_free"] == GAUGE_AXIS)
    assert gauge.sum() >= 3
    assert (omega * axis).sum(-1)[gauge].abs().max() < 1e-5
    assert omega[gauge].norm(dim=-1).min() > 1e-3  # it still turns, just not about the free axis
    full = batch["frag_free"] == GAUGE_FULL
    assert (omega[full] == 0).all()


def test_the_pinned_target_needs_no_turn_about_the_free_axis(batch) -> None:
    """The other half of the gauge convention: the target's relative rotation to R_0 is already
    perpendicular to the axis, so a model that cannot turn about it can still reach it exactly."""
    rot0, _ = random_state(batch, seed=9)
    rot1, _ = select_targets(batch, rot0)
    omega = so3.log(rot1 @ rot0.transpose(-1, -2))
    axis = (rot0 @ batch["frag_axis"].unsqueeze(-1)).squeeze(-1)
    gauge = batch["frag_mask"] & (batch["frag_free"] == GAUGE_AXIS)
    assert (omega * axis).sum(-1)[gauge].abs().max() < 1e-4


# --------------------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------------------
class FragmentOracle(nn.Module):
    """Always predicts the identity-referenced target: the sampler must land exactly on it."""

    def __init__(self, rot: torch.Tensor, trans: torch.Tensor) -> None:
        super().__init__()
        self.rot, self.trans = rot, trans

    def encode_pocket(self, batch):
        return None

    def forward_fragments(self, batch, rot, trans, t, h_p=None):
        return self.rot, self.trans


def test_the_fragment_sampler_lands_on_the_prediction(batch) -> None:
    eye = torch.eye(3).expand_as(batch["frag_rot"][:, :, 0])
    rot1 = select_targets(batch, eye)[0]
    oracle = FragmentOracle(rot1, batch["frag_trans"])
    cfg = FlowConfig(fragments=True)
    x, rot, trans = sample_fragments(oracle, batch, cfg, num_steps=7, return_state=True)
    m = batch["frag_mask"]
    assert torch.allclose(rot[m], rot1[m], atol=1e-4)
    assert torch.allclose(trans[m], batch["frag_trans"][m], atol=1e-4)
    assert torch.allclose(x, assemble(batch, rot1, batch["frag_trans"]), atol=1e-4)
    # ``sample`` dispatches on the config, so training and docking need no fragment-specific call.
    assert torch.allclose(sample(oracle, batch, cfg, num_steps=3), x, atol=1e-4)


def test_the_fragment_prior_is_centred_on_the_pocket_and_uniform_in_rotation(cases) -> None:
    batch = collate([item for item, _ in cases] * 300)
    torch.manual_seed(10)
    rot, trans = sample_fragment_prior(batch, std=4.0)
    m = batch["frag_mask"]
    for b in range(len(cases)):
        draws = trans[b :: len(cases)][m[b :: len(cases)]]
        assert torch.allclose(draws.mean(0), batch["pocket_com"][b], atol=0.25)
    assert rot[m].mean(0).abs().max() < 0.05  # E[R] = 0 for the uniform distribution on SO(3)
    assert torch.equal(rot[~m], torch.eye(3).expand_as(rot[~m]))
    assert (trans[~m] == 0).all()


@pytest.mark.parametrize("rot_loss", ["chordal", "geodesic"])
def test_fragment_loss_and_backward(batch, rot_loss) -> None:
    torch.manual_seed(0)
    model = PocketDiT(tiny_config())
    loss, metrics = flow_matching_loss(model, batch, FlowConfig(fragments=True, rot_loss=rot_loss))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert set(metrics) == {"loss", "trans_mse", "rot_loss", "coord_mse", "dist_mse", "centroid_err"}


def test_overfit_single_complex_in_fragment_mode() -> None:
    """A tiny model must learn to dock one complex from noise, now as rigid fragments."""
    if not DUMMY.exists():
        pytest.skip("example data not found")
    torch.manual_seed(0)
    item, _ = fragment_case("1U1C_BAU", crop=PocketCrop(max_atoms=96, cutoff=5.0))
    batch = collate([item] * 8)
    model = PocketDiT(tiny_config())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cfg = FlowConfig(fragments=True)
    for _ in range(600):  # ~25 s on 4 CPU cores; medians 0.31-0.79 Å over seeds 0-4, against a 0.08 Å floor
        loss, _ = flow_matching_loss(model, batch, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    r = pose_rmsd(sample(model, batch, cfg), batch)
    assert r.median() < 1.0, r
