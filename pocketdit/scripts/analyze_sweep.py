"""Read the CSVs written by ``fragment_sweep.py`` and report the distributions of interest.

    python scripts/analyze_sweep.py .sweep/shard_*.csv

Two questions, answered separately:

* how much of the fragment set is gauge degenerate (one- or two-atom / collinear), and whether
  that share is a property of the dataset or of a handful of flexible molecules;
* what the torsion (rotatable bond) distribution looks like, including how much of it survives
  as soft joints after merging.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        for k in (
            "atoms",
            "torsions",
            "cuts",
            "fragments",
            "collinear",
            "min_size",
            "max_size",
            "torsions_filtered",
            "cuts_filtered",
        ):
            r[k] = int(r[k])
        r["hist"] = {int(k): int(v) for k, v in (kv.split(":") for kv in r["size_hist"].split(";") if kv)}
    return rows


def pct(a: float, b: float) -> str:
    return "n/a" if b == 0 else f"{100 * a / b:5.1f}%"


def quantiles(values, qs=(10, 25, 50, 75, 90, 95, 99)) -> str:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "n/a"
    parts = "  ".join(f"p{q} {np.percentile(arr, q):5.1f}" for q in qs)
    return f"mean {arr.mean():5.2f}  sd {arr.std():5.2f}  {parts}  max {arr.max():5.1f}"


def bar(count: int, scale: int, width: int = 50) -> str:
    return "#" * max(0, min(width, round(count * width / scale)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--top", type=int, default=15, help="how many worst molecules to list")
    args = p.parse_args(argv)

    rows = load(args.csv)
    if not rows:
        sys.exit("no rows")

    n_mol = len(rows)
    total_frags = sum(r["fragments"] for r in rows)
    total_tors = sum(r["torsions"] for r in rows)
    total_cuts = sum(r["cuts"] for r in rows)
    degenerate = sum(r["collinear"] for r in rows)
    two_atom = sum(r["hist"].get(2, 0) for r in rows)
    one_atom = sum(r["hist"].get(1, 0) for r in rows)
    total_atoms = sum(r["atoms"] for r in rows)

    print("=" * 78)
    print(f"molecules {n_mol}   heavy atoms {total_atoms}   fragments {total_frags}   torsions {total_tors}")
    print("=" * 78)

    # ---------------------------------------------------------------- degenerate share
    print("\n## Gauge degenerate fragments (the 15% question)\n")
    print(f"  degenerate fragments      {degenerate:7d} / {total_frags}  =  {pct(degenerate, total_frags)}")
    print(f"    of which 2-atom         {two_atom:7d}         =  {pct(two_atom, total_frags)}")
    print(f"    of which 1-atom         {one_atom:7d}         =  {pct(one_atom, total_frags)}")
    print(
        f"    collinear with >=3 atoms{degenerate - two_atom - one_atom:7d}         =  "
        f"{pct(degenerate - two_atom - one_atom, total_frags)}"
    )

    mol_frac = np.array([r["collinear"] / r["fragments"] for r in rows])
    print(f"\n  per-molecule share         {quantiles(mol_frac)}")
    print(f"  molecules with none        {int((mol_frac == 0).sum()):6d}  {pct((mol_frac == 0).sum(), n_mol)}")
    print(f"  molecules with some        {int((mol_frac > 0).sum()):6d}  {pct((mol_frac > 0).sum(), n_mol)}")

    # How concentrated is the degenerate mass? Sorted by contribution.
    ranked = sorted(rows, key=lambda r: -r["collinear"])
    cum, marks = 0, {}
    for i, r in enumerate(ranked, 1):
        cum += r["collinear"]
        for hit in (0.25, 0.5, 0.75):
            if hit not in marks and cum >= hit * degenerate:
                marks[hit] = i
    print("\n  concentration of the degenerate fragments:")
    for hit in (0.25, 0.5, 0.75):
        i = marks.get(hit, len(ranked))
        print(f"    {int(hit * 100):2d}% of them sit in the top {i:5d} molecules ({pct(i, n_mol)} of the dataset)")

    # Size histogram, fragment level.
    hist = Counter()
    for r in rows:
        hist.update(r["hist"])
    print("\n  fragment size histogram")
    scale = max(hist.values())
    for size in sorted(hist):
        count = hist[size]
        tag = "  <- degenerate" if size <= 2 else ""
        print(f"    {size:3d} atoms {count:7d}  {pct(count, total_frags)}  {bar(count, scale)}{tag}")

    # ---------------------------------------------------------------- torsions
    print("\n## Rotatable bonds (torsions)\n")
    tors = [r["torsions"] for r in rows]
    print(f"  per molecule               {quantiles(tors)}")
    print(f"  total                      {total_tors}")
    print(f"  cut (kept as soft joints)  {total_cuts}  {pct(total_cuts, total_tors)}")
    print(f"  merged back (rigid)        {total_tors - total_cuts}  {pct(total_tors - total_cuts, total_tors)}")

    print("\n  torsion count histogram (molecules per bin)")
    bins = [
        (0, 0),
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 15),
        (16, 20),
        (21, 25),
        (26, 30),
        (31, 40),
        (41, 10**6),
    ]
    counts = [sum(1 for t in tors if lo <= t <= hi) for lo, hi in bins]
    scale = max(counts)
    for (lo, hi), c in zip(bins, counts, strict=True):
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**6 else f"{lo}-{hi}")
        print(f"    {label:>6} bonds {c:6d}  {pct(c, n_mol)}  {bar(c, scale)}")

    print("\n  torsion count by whether the molecule stays in one fragment")
    single = [r for r in rows if r["fragments"] == 1]
    multi = [r for r in rows if r["fragments"] > 1]
    print(
        f"    single-fragment molecules  {len(single):6d}  {pct(len(single), n_mol)}  "
        f"mean torsions {np.mean([r['torsions'] for r in single]) if single else 0:.2f}"
    )
    print(
        f"    split molecules            {len(multi):6d}  {pct(len(multi), n_mol)}  "
        f"mean torsions {np.mean([r['torsions'] for r in multi]) if multi else 0:.2f}"
        f"  mean fragments {np.mean([r['fragments'] for r in multi]) if multi else 0:.2f}"
    )

    # ---------------------------------------------------------------- fragments per molecule
    print("\n## Fragments per molecule\n")
    print(f"  {quantiles([r['fragments'] for r in rows])}")
    print(
        f"  molecules with exactly one {sum(1 for r in rows if r['fragments'] == 1):6d}  "
        f"{pct(sum(1 for r in rows if r['fragments'] == 1), n_mol)}"
    )
    print(f"  mean atoms per fragment    {total_atoms / total_frags:.2f}")
    print(f"  mean fragments per torsion {total_frags / total_tors:.3f}")

    # ---------------------------------------------------------------- variants
    print("\n## Variant: ignore_conjugated=True (drops weak conjugation, e.g. ester C-O)\n")
    tf = sum(r["torsions_filtered"] for r in rows)
    cf = sum(r["cuts_filtered"] for r in rows)
    print(
        f"  torsions {tf} ({tf - total_tors:+d} vs default)   cuts {cf} ({cf - total_cuts:+d})   "
        f"cut share {pct(cf, tf)}"
    )

    # ---------------------------------------------------------------- worst offenders
    print(f"\n## {args.top} molecules with the most degenerate fragments\n")
    print(f"  {'name':10s} {'atoms':>5s} {'tors':>5s} {'cuts':>5s} {'frags':>5s} {'degen':>5s} {'share':>6s}")
    for r in ranked[: args.top]:
        print(
            f"  {r['name']:10s} {r['atoms']:5d} {r['torsions']:5d} {r['cuts']:5d} {r['fragments']:5d} "
            f"{r['collinear']:5d} {100 * r['collinear'] / r['fragments']:5.1f}%"
        )


if __name__ == "__main__":
    main()
