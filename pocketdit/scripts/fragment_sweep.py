"""Per-molecule fragmentation statistics over a PDBBind-style dataset.

``python -m pocketdit.fragment`` prints one aggregate report. This script records the same
quantities per complex so the distribution can be inspected afterwards: the share of gauge
degenerate fragments, the torsion distribution, and how both vary with ligand flexibility.

Run from the ``pocketdit/`` directory:

    python scripts/fragment_sweep.py --root D:/Data_flie/data/pdbbind/PDBBind_processed --out sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketdit.chem import load_ligand  # noqa: E402
from pocketdit.fragment import (  # noqa: E402
    fragment,
    fragment_positions,
    is_collinear,
    rotatable_bonds,
)

FIELDS = [
    "name",
    "atoms",
    "torsions",
    "cuts",
    "fragments",
    "collinear",
    "min_size",
    "max_size",
    "size_hist",
    "torsions_filtered",
    "cuts_filtered",
]


def analyze(path: Path, selection: str) -> dict:
    mol = load_ligand(path)
    frag = fragment(mol, selection=selection)
    pos = mol.GetConformer().GetPositions()
    sizes = frag.sizes
    collinear = sum(is_collinear(p) for p in fragment_positions(frag.frag_idx, pos))
    # Same molecule seen through the variant that drops weak conjugation (ester C-O etc.).
    filtered = fragment(mol, selection=selection, ignore_conjugated=True)
    hist = np.bincount(sizes)
    return {
        "name": path.parent.name or path.stem,
        "atoms": mol.GetNumAtoms(),
        "torsions": len(frag.rotatable_bonds),
        "cuts": len(frag.cut_bonds),
        "fragments": frag.num_fragments,
        "collinear": collinear,
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "size_hist": ";".join(f"{s}:{c}" for s, c in enumerate(hist) if c),
        "torsions_filtered": len(filtered.rotatable_bonds),
        "cuts_filtered": len(filtered.cut_bonds),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--glob", default="*/*_ligand.sdf")
    p.add_argument("--selection", default="compact", choices=("compact", "random", "none"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--failures", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--prefixes", help="comma separated first letters of the complex names to keep")
    args = p.parse_args(argv)

    paths = sorted(args.root.glob(args.glob))
    if args.prefixes:
        wanted = set(args.prefixes.split(","))
        paths = [p for p in paths if (p.parent.name or p.stem)[:1] in wanted]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        p.error(f"no ligands matched {args.root / args.glob}")

    rows, failures = [], []
    start = time.time()
    for i, path in enumerate(paths, 1):
        try:
            rows.append(analyze(path, args.selection))
        except Exception as exc:  # noqa: BLE001 - one corrupt input must not stop the sweep
            failures.append((path, str(exc)))
        if i % 1000 == 0:
            rate = i / (time.time() - start)
            print(f"  {i}/{len(paths)}  {rate:5.1f}/s  eta {(len(paths) - i) / rate / 60:5.1f} min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out} ({len(failures)} failures, {time.time() - start:.0f} s)")

    if args.failures and failures:
        args.failures.write_text("\n".join(f"{p}\t{exc}" for p, exc in failures))
    for path, exc in failures[:10]:
        print(f"  FAIL {path.parent.name}: {exc}")


if __name__ == "__main__":
    main()
