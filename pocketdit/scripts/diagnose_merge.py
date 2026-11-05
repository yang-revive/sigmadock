"""Why does the degenerate share land at 23.5% instead of 16.7%?

Prints, for one ligand, the number of rotatable bonds, how many fragments it leaves, the
adjacency list of the rotatable-bond graph (degree > 2 is where ``_mergeable`` starts dropping
bonds), and -- per rotatable bond -- whether it was cut and why.

    python scripts/diagnose_merge.py --root D:/Data_flie/data/pdbbind/PDBBind_processed --name 2rqu
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketdit.chem import load_ligand  # noqa: E402
from pocketdit.fragment import (  # noqa: E402
    MIN_FRAGMENT_ATOMS,
    RIGID_EXEMPT_ATOMS,
    _is_rigid,
    _mergeable,
    _neighbours,
    fragment,
    rotatable_bonds,
)


def diagnose(path: Path) -> None:
    mol = load_ligand(path)
    frag = fragment(mol)
    bonds = rotatable_bonds(mol)
    ends = {b: (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx()) for b in bonds}
    deg: dict[int, int] = defaultdict(int)
    for b in bonds:
        u, v = ends[b]
        deg[u] += 1
        deg[v] += 1
    nbrs = _neighbours(mol)
    end_list = [ends[b] for b in bonds]
    sizes = frag.sizes

    print(f"{path.parent.name}: {mol.GetNumAtoms()} heavy atoms, {len(bonds)} torsions, "
          f"{frag.num_fragments} fragments, sizes {sorted(sizes.tolist())}")
    small = int((frag.sizes <= 2).sum())
    print(f"  fragments with <= 2 atoms: {small}")

    hard = [b for b in bonds if mol.GetAtomWithIdx(ends[b][0]).IsInRing() and mol.GetAtomWithIdx(ends[b][1]).IsInRing()]
    merged_deg = [b for b in bonds if not (mol.GetAtomWithIdx(ends[b][0]).IsInRing()
                                           and mol.GetAtomWithIdx(ends[b][1]).IsInRing())
                  and deg[ends[b][0]] == 1 and deg[ends[b][1]] == 1]
    candidates = _mergeable(mol, bonds, ends)
    print(f"  ring-ring torsions (always dropped):      {len(hard)}  -> {len(hard) + len(merged_deg)}")
    print(f"  terminal torsions (deg 1 on both ends):   {len(merged_deg)}")
    print(f"  actually tried for merging:               {len(candidates)} "
          f"({100 * len(candidates) / max(len(bonds), 1):.1f}% of all torsions)")

    # Of the candidates, how many ended up merged, and how many were rejected as non-rigid?
    merged = [b for b in candidates if b not in frag.cut_bonds]
    print(f"    merged back:                            {len(merged)}")
    print(f"    rejected because the fragment stays rigid-free: {len(candidates) - len(merged)}")

    # Which rejections are unavoidable: merging would leave a fragment with an internal torsion.
    unavoidable = 0
    for b in candidates:
        if b in frag.cut_bonds:
            trial_cut = set(frag.cut_bonds) - {b}
            from pocketdit.fragment import _components

            if not _is_rigid(_components(mol, trial_cut), end_list, nbrs):
                unavoidable += 1
    print(f"    of those, merging would break rigidity (unavoidable from this state): {unavoidable}")

    # Rotatable-bond graph degrees.
    degree_hist = defaultdict(int)
    for a, d in deg.items():
        degree_hist[d] += 1
    print("  rotatable-bond graph, heavy atoms by number of incident torsions: "
          + ", ".join(f"deg{d}={c}" for d, c in sorted(degree_hist.items())))

    lowdeg = sum(1 for a, d in deg.items() if d == 1)
    print(f"  atoms with a single incident torsion: {lowdeg} "
          f"(each can end up as a 2-atom fragment's anchor)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--name", nargs="+", required=True)
    args = p.parse_args(argv)
    for name in args.name:
        diagnose(args.root / name / f"{name}_ligand.sdf")


if __name__ == "__main__":
    main()
