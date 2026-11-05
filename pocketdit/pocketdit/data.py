"""Preprocessing to a tensor cache, the dataset, and padded batching.

Each complex is parsed with RDKit exactly once (``python -m pocketdit.data ...``) and stored as a
``.pt`` file of tensors. Training then only does ``torch.load`` + padding.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pocketdit.chem import featurize_complex, load_ligand

# --------------------------------------------------------------------------------------
# Index of complexes: (protein.pdb, ligand.sdf, optional reference.sdf)
# --------------------------------------------------------------------------------------
Row = tuple[Path, Path, "Path | None"]


def index_from_folders(root: str | Path, protein_glob: str, ligand_glob: str, ref_glob: str = "") -> list[Row]:
    """One sub-folder per complex. Every ligand matching ``ligand_glob`` becomes one row."""
    rows: list[Row] = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        proteins = sorted(folder.glob(protein_glob))
        refs = sorted(folder.glob(ref_glob)) if ref_glob else []
        if len(proteins) != 1 or (ref_glob and len(refs) != 1):
            print(f"[skip] {folder}: expected one protein{' and one reference' if ref_glob else ''}")
            continue
        for ligand in sorted(folder.glob(ligand_glob)):
            rows.append((proteins[0], ligand, refs[0] if refs else None))
    return rows


def index_from_csv(path: str | Path) -> list[Row]:
    """CSV with columns ``protein,ligand[,ref]``; relative paths are relative to the CSV."""
    base = Path(path).parent
    rows: list[Row] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ref = r.get("ref") or None
            rows.append((base / r["protein"], base / r["ligand"], base / ref if ref else None))
    return rows


def complex_name(row: Row) -> str:
    return f"{row[1].parent.name}__{row[1].stem}"


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def _process(job: tuple[Row, Path, float, float, int]) -> str | None:
    (protein, ligand, ref), out_dir, cutoff, com_cutoff, max_atoms = job
    name = complex_name((protein, ligand, ref))
    try:
        mol = load_ligand(ligand)
        if mol.GetNumAtoms() > max_atoms:
            raise ValueError(f"{mol.GetNumAtoms()} heavy atoms > {max_atoms}")
        ref_pos = load_ligand(ref).GetConformer().GetPositions() if ref else mol.GetConformer().GetPositions()
        item = featurize_complex(protein, mol, ref_pos, pocket_cutoff=cutoff, com_cutoff=com_cutoff)
        item["name"] = name
        torch.save(item, out_dir / f"{name}.pt")
        return None
    except Exception as e:  # keep going and report every failure at the end
        return f"{name}: {e}"


def preprocess(
    rows: list[Row], out_dir: str | Path, pocket_cutoff: float, com_cutoff: float, max_atoms: int, workers: int
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(row, out_dir, pocket_cutoff, com_cutoff, max_atoms) for row in rows]
    if workers > 1:
        with Pool(workers) as pool:
            errors = [e for e in pool.imap_unordered(_process, jobs, chunksize=8) if e]
    else:
        errors = [e for e in map(_process, jobs) if e]
    (out_dir / "failures.txt").write_text("\n".join(errors))
    print(f"Cached {len(rows) - len(errors)}/{len(rows)} complexes in {out_dir} ({len(errors)} failures listed).")


# --------------------------------------------------------------------------------------
# Narrowing the cached pocket to an atom budget
# --------------------------------------------------------------------------------------
@dataclass
class PocketCrop:
    """How a cached pocket is narrowed to a budget of atoms.

    Whole residues only, ranked by their distance to the reference: residues closer than
    ``core_cutoff`` are the contact shell and are kept first, the rest fill the remaining budget
    nearest-first. A 4 Å core plus a fill to 256 atoms reaches an effective cutoff of 6.5 Å on the
    ten example complexes -- which is what SigmaDock itself trains with (6.0 Å + 0.5 Å of noise) --
    against ~400 atoms for the plain 8 Å shell. Besides the memory, a tighter pocket also helps the
    output head: it averages the pocket term over ``n_p`` atoms, so 400 mostly-distant atoms dilute
    the ~37 real contacts tenfold, and a count that swings from 126 to 600 across complexes makes
    that term's scale swing with it.

    Residues, never individual atoms. A half-residue is off-distribution for the atom-name and
    residue-type embeddings (a backbone N with no CA, three carbons of a phenyl ring), and ranking
    by distance beats sampling uniformly from the 4-8 Å band, where a 7.9 Å atom would be kept
    while a 4.1 Å one is dropped.

    ``max_atoms`` is a cap, not a quota: a small buried pocket stays small rather than being padded
    out with solvent-exposed residues. If the core alone overruns the cap its farthest residues go
    too, so a large ligand degrades to its closest contacts instead of being dropped outright.

    Randomisation goes on the *boundary*, not on individual atoms: the shell is re-selected against
    a rigidly displaced copy of the reference, the cutoff is jittered, and whole residues may be
    dropped -- but every residue that is kept is kept whole. Dropping random atoms would punch
    holes in the pocket wall, so the steric field the model learns from would differ between epochs
    for the same complex.

    ``dir_noise`` and ``rot_noise`` are the important ones. The shell selected against the
    untouched reference traces the ligand's outline at ~1 Å, which the model can read the answer
    off. So the shell is selected against a copy of the reference that is rotated about its own
    centroid by theta ~ U[0, rot_noise] and then displaced by |u| ~ U[0, dir_noise] in a uniformly
    random direction. Both are *maxima*, not fixed amounts, so the model sees pocket definitions
    ranging from near-perfect to poor; the mean perturbation is about half of each.

    They break different things, which is why both are here. Over the ten example complexes, 30
    draws each -- the share of selected residues that differ from the untouched selection, the
    share of the true 6.5 Å shell still selected, the distance from the pocket's centroid to the
    ligand's, and the principal-axis angle between pocket and ligand:

        untouched            237 atoms    0%    97%    2.21 Å    28 deg
        displace 2 Å         239 atoms   16%    91%    2.44 Å    30 deg
        displace 4 Å         242 atoms   30%    85%    3.03 Å    31 deg
        rotate 30 deg        240 atoms   15%    92%    2.11 Å    32 deg
        rotate 60 deg        243 atoms   26%    87%    2.05 Å    39 deg
        default 2 Å + 30 deg 242 atoms   24%    88%    2.29 Å    34 deg

    A displacement moves the pocket off the ligand (2.21 -> 3.03 Å) and on average barely turns it
    (28 -> 31 deg); a rotation turns it (28 -> 39 deg) and, being about the reference's own
    centroid, does not move it at all. The atom count stays at 237-243 throughout, so neither
    disturbs the budget. The axis figures are population averages with a wide spread -- a small,
    well-aligned ligand in a tight pocket can have its axis moved a long way by a displacement too.

    Both also stand in for cross-docking: at inference the shell comes from a reference ligand that
    sits a few Å and a few tens of degrees off the query's true pose, which is exactly what a
    rotated, displaced reference is. So this buys the cross-docking input distribution without
    needing cross-docked data -- and that is also what bounds ``rot_noise``. A reference ligand
    binds *along the groove*, so at inference the angle is small, not uniform. Two ligands sharing a
    groove differ by roughly 20-40 deg, hence the 30 deg default. Going further is not more
    realistic, it is off-distribution in the wrong direction, and it costs: a reference rotated by
    90 deg has 47% of its atoms buried within 2.5 Å of a protein heavy atom, so the shell around it
    is protein interior and a quarter of the genuine pocket lining is crowded out. (For the record,
    180 deg decorrelates *less* than 90 deg -- rotating an elongated ligand by 180 deg about a
    perpendicular axis roughly maps it onto itself.)

    The contact core is left untouched on purpose: those residues are the signal, and what they
    give away -- the ligand's gross position, blurred by a 4 Å dilation -- is the pocket location,
    which is a legitimate input to the task and is handed to the model as ``pocket_com`` anyway.
    """

    max_atoms: int = 256
    core_cutoff: float = 4.0  # Å, residues this close to the reference are kept first
    cutoff: float = 6.5  # Å, shell boundary; on the examples it and the budget bind about equally often
    dir_noise: float = 2.0  # Å, training only: displacement of the reference, |u| ~ U[0, dir_noise]
    rot_noise: float = 30.0  # degrees, training only: rotation about its centroid, theta ~ U[0, rot_noise]
    cutoff_jitter: float = 1.5  # training only: cutoff ~ U[cutoff - jitter, cutoff + jitter]
    shell_dropout: float = 0.0  # training only: chance of dropping an individual non-core residue

    def cache_cutoff_needed(self, ref_radius: float) -> float:
        """Cache radius this crop can reach into for a reference of this extent.

        A rotation about the centroid moves an atom ``ref_radius`` out by up to
        ``2 sin(theta/2) * ref_radius``, so unlike the other knobs this one scales with the
        reference. See ``ComplexDataset`` for why reaching past the cache is refused.
        """
        swing = 2 * math.sin(math.radians(self.rot_noise) / 2) * ref_radius
        return self.cutoff + self.cutoff_jitter + self.dir_noise + swing


DROPPED_BY_CROP = ("poc_res_id", "poc_res_dmin", "ref_pos", "ref_radius", "poc_cache_cutoff")


def _residue_distance(poc_pos: torch.Tensor, ref: torch.Tensor, res_id: torch.Tensor, n_res: int) -> torch.Tensor:
    """Per-residue distance to the nearest point of ``ref``: [n_res], inf where a residue is absent."""
    d = (poc_pos[:, None] - ref[None]).norm(dim=-1).min(dim=1).values
    out = torch.full((n_res,), float("inf"))
    return out.scatter_reduce_(0, res_id, d, reduce="amin", include_self=True)


def _random_rotation(max_angle_deg: float) -> torch.Tensor:
    """Rotation by theta ~ U[0, max_angle_deg] about a uniformly random axis (Rodrigues)."""
    axis = torch.randn(3)
    a = axis / axis.norm().clamp(min=1e-6)
    theta = torch.rand(()) * math.radians(max_angle_deg)
    k = torch.zeros(3, 3)
    k[0, 1], k[0, 2] = -a[2], a[1]
    k[1, 0], k[1, 2] = a[2], -a[0]
    k[2, 0], k[2, 1] = -a[1], a[0]
    return torch.eye(3) + theta.sin() * k + (1 - theta.cos()) * (k @ k)


def perturb_reference(ref: torch.Tensor, crop: PocketCrop) -> torch.Tensor:
    """A rotated and displaced copy of the selection reference: a stand-in reference ligand.

    Rotation is about the reference's own centroid, so it stays in the pocket and only its
    orientation changes; the displacement then moves it bodily. A single-point reference (an
    explicit pocket centre) is unaffected by the rotation, as it should be.
    """
    if crop.rot_noise > 0 and len(ref) > 1:
        center = ref.mean(0)
        ref = (ref - center) @ _random_rotation(crop.rot_noise).T + center
    if crop.dir_noise > 0:
        u = torch.randn(3)
        ref = ref + u * (float(torch.rand(())) * crop.dir_noise / u.norm().clamp(min=1e-6))
    return ref


def _check_cache_headroom(item: dict, crop: PocketCrop) -> None:
    """Refuse a cache the perturbed reference would reach past (see ``ComplexDataset``)."""
    cached, radius = float(item["poc_cache_cutoff"]), float(item["ref_radius"])
    needed = crop.cache_cutoff_needed(radius)
    if needed > cached + 1e-6:
        raise ValueError(
            f"complex {item.get('name', '?')!r} was cached with pocket_cutoff={cached:g} Å but this "
            f"crop reaches {needed:g} Å for a reference of radius {radius:.1f} Å (cutoff "
            f"{crop.cutoff:g} + jitter {crop.cutoff_jitter:g} + dir_noise {crop.dir_noise:g} + "
            f"{crop.rot_noise:g}° swing). Re-run pocketdit.data with --pocket_cutoff {needed:g} or "
            f"larger, or lower the crop."
        )


def crop_pocket(item: dict, crop: PocketCrop, train: bool = False) -> dict:
    """Narrow one cached complex's pocket to ``crop``, dropping the selection bookkeeping.

    The contact core -- residues within ``core_cutoff`` of the untouched reference -- is always
    kept and always ranked first, because those are the residues that determine the pose. The shell
    around it is re-selected against a rotated and displaced reference and fills whatever budget is
    left, nearest-first.

    ``train=True`` enables the perturbation, cutoff jitter and residue dropout; evaluation and
    inference select against the untouched reference so a pose is reproducible.
    """
    dmin, res_id = item["poc_res_dmin"], item["poc_res_id"]
    n_res = len(dmin)
    core = dmin < crop.core_cutoff

    cutoff, shell = crop.cutoff, dmin
    if train:
        if crop.cutoff_jitter > 0:
            cutoff += float(torch.empty(()).uniform_(-crop.cutoff_jitter, crop.cutoff_jitter))
        if crop.dir_noise > 0 or crop.rot_noise > 0:
            _check_cache_headroom(item, crop)
            ref = perturb_reference(item["ref_pos"], crop)
            shell = _residue_distance(item["poc_pos"], ref, res_id, n_res)

    keep = core | (shell < cutoff)
    if train and crop.shell_dropout > 0:
        keep &= core | (torch.rand(n_res) >= crop.shell_dropout)
    if not keep.any():
        keep[dmin.argmin()] = True  # never hand the model an empty pocket

    # Core first by true distance, then the shell by displaced distance; the budget truncates the
    # far end. Residues that were not kept sort last and are removed by the isfinite mask.
    key = torch.where(core, dmin, shell + 1e3)
    key = torch.where(keep, key, torch.tensor(float("inf")))
    order = key.argsort()
    sizes = torch.bincount(res_id, minlength=n_res)[order]
    take = order[(torch.cumsum(sizes, 0) <= crop.max_atoms) & key[order].isfinite()]
    if len(take) == 0:
        take = order[:1]  # a single residue larger than the whole budget

    atoms = torch.isin(res_id, take)
    out = {k: v for k, v in item.items() if k not in DROPPED_BY_CROP}
    out["poc_feat"], out["poc_pos"] = item["poc_feat"][atoms], item["poc_pos"][atoms]
    return out


# --------------------------------------------------------------------------------------
# Dataset & batching
# --------------------------------------------------------------------------------------
class ComplexDataset(Dataset):
    """Cached complexes, with the pocket narrowed to ``crop`` as each one is loaded."""

    def __init__(self, cache_dir: str | Path, crop: PocketCrop | None = None, train: bool = False) -> None:
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached complexes in {cache_dir}")
        self.crop = crop or PocketCrop()
        self.train = train
        # The crop can only select residues the cache holds, so a perturbed shell reaching past the
        # cache radius would be silently clipped on that side -- which is the one thing the
        # perturbation is supposed to avoid. How far it reaches scales with the reference's own
        # extent, so every complex is checked as it is loaded; this is the early warning for the
        # common case of a cache that is simply too narrow for the configured crop.
        if train:
            _check_cache_headroom(torch.load(self.files[0], weights_only=True), self.crop)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return crop_pocket(torch.load(self.files[idx], weights_only=True), self.crop, self.train)


def collate(items: list[dict]) -> dict:
    """Pad ligands to the largest N and pockets to the largest M in the batch."""
    B = len(items)
    N = max(len(it["lig_pos"]) for it in items)
    M = max(len(it["poc_pos"]) for it in items)
    out = {
        "lig_feat": torch.zeros(B, N, items[0]["lig_feat"].shape[1], dtype=torch.long),
        "lig_bond": torch.zeros(B, N, N, dtype=torch.long),
        "lig_spd": torch.zeros(B, N, N, dtype=torch.long),
        "lig_pos": torch.zeros(B, N, 3),
        "chiral_nbrs": torch.zeros(B, N, 3, dtype=torch.long),
        "chiral_sign": torch.zeros(B, N),
        "lig_mask": torch.zeros(B, N, dtype=torch.bool),
        "poc_feat": torch.zeros(B, M, items[0]["poc_feat"].shape[1], dtype=torch.long),
        "poc_pos": torch.zeros(B, M, 3),
        "poc_mask": torch.zeros(B, M, dtype=torch.bool),
        "pocket_com": torch.stack([it["pocket_com"] for it in items]),
        "name": [it.get("name", "") for it in items],
    }
    for b, it in enumerate(items):
        n, m = len(it["lig_pos"]), len(it["poc_pos"])
        for key in ("lig_feat", "lig_pos", "chiral_nbrs", "chiral_sign"):
            out[key][b, :n] = it[key]
        out["lig_bond"][b, :n, :n] = it["lig_bond"]
        out["lig_spd"][b, :n, :n] = it["lig_spd"]
        out["lig_mask"][b, :n] = True
        out["poc_feat"][b, :m] = it["poc_feat"]
        out["poc_pos"][b, :m] = it["poc_pos"]
        out["poc_mask"][b, :m] = True
    return out


def to_device(batch: dict, device: torch.device | str) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Featurise complexes into a tensor cache.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="folder with one sub-folder per complex")
    src.add_argument("--csv", help="CSV with columns protein,ligand[,ref]")
    p.add_argument("--protein_glob", default="*_protein.pdb")
    p.add_argument("--ligand_glob", default="*_ligand.sdf")
    p.add_argument("--ref_glob", default="", help="reference ligand (cross-docking); default: ligand itself")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--pocket_cutoff",
        type=float,
        default=15.0,
        help="widest shell to cache; must cover PocketCrop.cache_cutoff_needed() for every complex",
    )
    p.add_argument("--com_cutoff", type=float, default=6.0, help="shell whose CoM becomes the prior centre")
    p.add_argument("--max_atoms", type=int, default=128, help="skip ligands with more heavy atoms")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    if args.root:
        rows = index_from_folders(args.root, args.protein_glob, args.ligand_glob, args.ref_glob)
    else:
        rows = index_from_csv(args.csv)
    preprocess(rows, args.out, args.pocket_cutoff, args.com_cutoff, args.max_atoms, args.workers)


if __name__ == "__main__":
    main()
