"""Preprocessing to a tensor cache, the dataset, and padded batching.

Each complex is parsed with RDKit exactly once (``python -m pocketdit.data ...``) and stored as a
``.pt`` file of tensors. Training then only does ``torch.load`` + padding.
"""

from __future__ import annotations

import argparse
import csv
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
def _process(job: tuple[Row, Path, float, int]) -> str | None:
    (protein, ligand, ref), out_dir, cutoff, max_atoms = job
    name = complex_name((protein, ligand, ref))
    try:
        mol = load_ligand(ligand)
        if mol.GetNumAtoms() > max_atoms:
            raise ValueError(f"{mol.GetNumAtoms()} heavy atoms > {max_atoms}")
        ref_pos = load_ligand(ref).GetConformer().GetPositions() if ref else mol.GetConformer().GetPositions()
        item = featurize_complex(protein, mol, ref_pos, pocket_cutoff=cutoff)
        item["name"] = name
        torch.save(item, out_dir / f"{name}.pt")
        return None
    except Exception as e:  # keep going and report every failure at the end
        return f"{name}: {e}"


def preprocess(rows: list[Row], out_dir: str | Path, pocket_cutoff: float, max_atoms: int, workers: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(row, out_dir, pocket_cutoff, max_atoms) for row in rows]
    if workers > 1:
        with Pool(workers) as pool:
            errors = [e for e in pool.imap_unordered(_process, jobs, chunksize=8) if e]
    else:
        errors = [e for e in map(_process, jobs) if e]
    (out_dir / "failures.txt").write_text("\n".join(errors))
    print(f"Cached {len(rows) - len(errors)}/{len(rows)} complexes in {out_dir} ({len(errors)} failures listed).")


# --------------------------------------------------------------------------------------
# Dataset & batching
# --------------------------------------------------------------------------------------
class ComplexDataset(Dataset):
    def __init__(self, cache_dir: str | Path) -> None:
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached complexes in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], weights_only=True)


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
        "ref_center": torch.stack([it["ref_center"] for it in items]),
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
    p.add_argument("--pocket_cutoff", type=float, default=8.0)
    p.add_argument("--max_atoms", type=int, default=128, help="skip ligands with more heavy atoms")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    if args.root:
        rows = index_from_folders(args.root, args.protein_glob, args.ligand_glob, args.ref_glob)
    else:
        rows = index_from_csv(args.csv)
    preprocess(rows, args.out, args.pocket_cutoff, args.max_atoms, args.workers)


if __name__ == "__main__":
    main()
