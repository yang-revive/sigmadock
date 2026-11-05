"""The pocket budget: whole residues, nearest first, and randomisation that keeps them whole."""

import pytest
import torch

from conftest import load_uncropped
from pocketdit.data import ComplexDataset, PocketCrop, collate, crop_pocket


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
        assert len(dropped) > 0, "the cached shell should not fit in the budget"
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
        # The cached shell really is more than the budget allows, i.e. the crop is doing something.
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


def test_crop_drops_the_selection_bookkeeping(items_uncropped):
    """These are selection inputs, not model inputs, so collate must never see them."""
    cropped = crop_pocket(items_uncropped[0], PocketCrop())
    for key in ("poc_res_id", "poc_res_dmin", "ref_pos", "poc_cache_cutoff"):
        assert key not in cropped
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


def kept_residues(full: dict, crop: PocketCrop, train: bool, draws: int = 1) -> list[set]:
    return [set(check_whole_residues(full, crop_pocket(full, crop, train=train)).tolist()) for _ in range(draws)]


def test_dir_noise_reselects_the_shell_but_keeps_the_core(items_uncropped):
    """A displaced reference must swap out shell residues while the contact core stays put."""
    full = items_uncropped[0]
    crop = PocketCrop(dir_noise=4.0, cutoff_jitter=0.0, shell_dropout=0.0)
    core = set(torch.nonzero(full["poc_res_dmin"] < crop.core_cutoff).flatten().tolist())
    base = kept_residues(full, crop, train=False)[0]
    torch.manual_seed(0)
    changed, sizes = [], []
    for kept in kept_residues(full, crop, train=True, draws=30):
        assert core <= kept, "the displacement must never drop a contact residue"
        changed.append(len(kept ^ base) / len(base))
        sizes.append(len(kept))
    # Substantial reselection, but the residue count stays in the same ballpark (budget intact).
    assert 0.1 < sum(changed) / len(changed) < 0.9, changed
    assert max(sizes) - min(sizes) < 0.5 * len(base), sizes


def test_dir_noise_moves_the_pocket_off_the_ligand(items_uncropped):
    """The point of the displacement: the pocket's centroid stops coinciding with the ligand's."""
    full = items_uncropped[0]
    lig_c = full["lig_pos"].mean(0)
    quiet = crop_pocket(full, PocketCrop(dir_noise=0.0, cutoff_jitter=0.0))["poc_pos"].mean(0)
    torch.manual_seed(0)
    crop = PocketCrop(dir_noise=4.0, cutoff_jitter=0.0)
    noisy = [crop_pocket(full, crop, train=True)["poc_pos"].mean(0) for _ in range(30)]
    off_quiet = (quiet - lig_c).norm()
    off_noisy = torch.stack([(c - lig_c).norm() for c in noisy]).mean()
    assert off_noisy > off_quiet, (off_noisy, off_quiet)


def test_dir_noise_is_off_outside_training(items_uncropped):
    full = items_uncropped[0]
    crop = PocketCrop(dir_noise=4.0, cutoff_jitter=0.0, shell_dropout=0.0)
    torch.manual_seed(0)
    a = crop_pocket(full, crop)["poc_pos"]
    torch.manual_seed(7)
    b = crop_pocket(full, crop)["poc_pos"]
    assert torch.equal(a, b)


def test_dataset_rejects_a_cache_too_narrow_for_the_crop(tmp_path, items_uncropped):
    """A displaced shell reaching past the cache radius would be clipped, so it must not pass."""
    torch.save(items_uncropped[0] | {"poc_cache_cutoff": torch.tensor(10.0)}, tmp_path / "a.pt")
    fits = PocketCrop(cutoff=6.5, cutoff_jitter=1.5, dir_noise=2.0)  # reaches exactly 10 Å
    assert fits.cache_cutoff_needed() == 10.0
    ComplexDataset(tmp_path, fits, train=True)
    with pytest.raises(ValueError, match="cached with pocket_cutoff"):
        ComplexDataset(tmp_path, PocketCrop(cutoff=6.5, cutoff_jitter=1.5, dir_noise=4.0), train=True)
    # Evaluation never displaces the reference, so the cache radius does not constrain it.
    ComplexDataset(tmp_path, PocketCrop(dir_noise=99.0), train=False)


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
