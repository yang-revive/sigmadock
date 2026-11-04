"""The pocket budget: whole residues, nearest first, and randomisation that keeps them whole."""

import torch

from conftest import load_uncropped
from pocketdit.data import PocketCrop, collate, crop_pocket


def residue_of_kept_atoms(full: dict, cropped: dict) -> torch.Tensor:
    """Residue rank of every atom of ``full`` that survived into ``cropped`` (-1 if dropped)."""
    same = (full["poc_pos"][:, None] == cropped["poc_pos"][None]).all(-1).any(1)
    return torch.where(same, full["poc_res_id"], torch.full_like(full["poc_res_id"], -1))


def check_whole_residues(full: dict, cropped: dict) -> torch.Tensor:
    """Assert no residue was split, and return the ranks of the residues that were kept."""
    kept_rank = residue_of_kept_atoms(full, cropped)
    kept = set()
    for r in range(len(full["poc_res_dmin"])):
        atoms = full["poc_res_id"] == r
        present = (kept_rank[atoms] == r).sum()
        assert present in (0, int(atoms.sum())), f"residue {r} split: {present}/{int(atoms.sum())} atoms"
        if present:
            kept.add(r)
    assert len(cropped["poc_pos"]) == sum(int((full["poc_res_id"] == r).sum()) for r in kept)
    return torch.tensor(sorted(kept))


def test_crop_keeps_whole_residues_nearest_first(items_uncropped):
    for full in items_uncropped:
        crop = PocketCrop()
        cropped = crop_pocket(full, crop)
        kept = check_whole_residues(full, cropped)
        dmin = full["poc_res_dmin"]
        dropped = torch.tensor([r for r in range(len(dmin)) if r not in set(kept.tolist())], dtype=torch.long)
        assert len(dropped) > 0, "the 8 Å shell should not fit in the budget"
        # Strict distance ranking: nothing dropped is closer than anything kept.
        assert dmin[kept].max() <= dmin[dropped].min()
        # The contact shell is never traded away for the far one.
        assert set(torch.nonzero(dmin < crop.core_cutoff).flatten().tolist()) <= set(kept.tolist())


def test_crop_respects_the_budget_as_a_cap_not_a_quota(items_uncropped):
    for full in items_uncropped:
        n_full = len(full["poc_pos"])
        assert len(crop_pocket(full, PocketCrop())["poc_pos"]) <= 256
        assert len(crop_pocket(full, PocketCrop(max_atoms=64))["poc_pos"]) <= 64
        # A budget wider than the cached shell leaves the pocket alone rather than padding it out.
        roomy = crop_pocket(full, PocketCrop(max_atoms=10_000, cutoff=99.0))
        assert len(roomy["poc_pos"]) == n_full
        # 8 Å really is more than the budget allows, i.e. the crop is doing something.
        assert n_full > 256


def test_crop_degrades_to_the_closest_contacts_when_the_core_overruns(items_uncropped):
    """A core bigger than the budget loses its farthest residues instead of the complex failing."""
    full = items_uncropped[0]
    cropped = crop_pocket(full, PocketCrop(max_atoms=40, core_cutoff=8.0))
    kept = check_whole_residues(full, cropped)
    assert 0 < len(cropped["poc_pos"]) <= 40
    assert kept.tolist() == list(range(len(kept)))  # still a nearest-first prefix


def test_crop_never_returns_an_empty_pocket(items_uncropped):
    for full in items_uncropped:
        starved = crop_pocket(full, PocketCrop(max_atoms=1, core_cutoff=0.0, cutoff=0.0))
        assert len(starved["poc_pos"]) > 0


def test_crop_drops_the_residue_bookkeeping(items_uncropped):
    cropped = crop_pocket(items_uncropped[0], PocketCrop())
    assert "poc_res_id" not in cropped and "poc_res_dmin" not in cropped
    assert collate([cropped])["poc_pos"].shape[1] == len(cropped["poc_pos"])


def test_training_jitter_varies_the_boundary_but_not_the_residues(items_uncropped):
    """The augmentation moves the pocket's edge; it never splits a residue or touches the core."""
    full = items_uncropped[0]
    crop = PocketCrop(cutoff=6.5, cutoff_jitter=1.5, shell_dropout=0.1)
    core = set(torch.nonzero(full["poc_res_dmin"] < crop.core_cutoff).flatten().tolist())
    torch.manual_seed(0)
    sizes = set()
    for _ in range(20):
        cropped = crop_pocket(full, crop, train=True)
        kept = check_whole_residues(full, cropped)
        assert core <= set(kept.tolist())  # dropout spares the contact shell
        assert len(cropped["poc_pos"]) <= crop.max_atoms
        sizes.add(len(cropped["poc_pos"]))
    assert len(sizes) > 5, sizes  # the boundary really does move between draws


def test_eval_crop_is_deterministic(items_uncropped):
    full = items_uncropped[0]
    crop = PocketCrop(cutoff_jitter=1.5, shell_dropout=0.5)
    torch.manual_seed(0)
    a = crop_pocket(full, crop)["poc_pos"]
    torch.manual_seed(1)
    b = crop_pocket(full, crop)["poc_pos"]
    assert torch.equal(a, b)


def test_budget_lands_near_the_sigmadock_cutoff():
    """4 Å core + fill to 256 should reproduce SigmaDock's own ~6.5 Å pocket."""
    effective = []
    for name in ("1R1H_BIR", "1U1C_BAU"):
        full = load_uncropped(name)
        kept = check_whole_residues(full, crop_pocket(full, PocketCrop()))
        effective.append(float(full["poc_res_dmin"][kept].max()))
    assert all(5.0 < e < 7.5 for e in effective), effective
