"""Read the CSVs written by ``fragment_geometry_sweep.py`` and report what they imply.

    python scripts/analyze_geometry_sweep.py geom_*.csv

Each ``<name>.csv`` is read together with the ``<name>.fragments.csv`` next to it. A complex that
appears in several files is taken from the last one, rows and fragments alike, so a re-run of a
subset is merged by passing it after the full sweep:

    python scripts/analyze_geometry_sweep.py geom.csv geom_rerun.csv

Every section is reported over the whole set and split by ligand size (<= 50, 51-100, > 100 heavy
atoms), because the earlier sweep showed the flexible tail behaving very differently from
drug-like ligands.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TIERS = (("<=50", 0, 50), ("51-100", 51, 100), (">100", 101, 10**9))
KINDS = ("aromatic", "aliphatic_ring", "macrocycle", "acyclic")
CHI3_P95 = 2.7955  # 95th percentile of the norm of a standard 3D Gaussian
RING_CLASSES = ("no saturated ring", "saturated ring", "macrocycle")
CUT_CAPS = (10, 15, 20)
FLOAT = ("conf_spread", "oracle_best", "oracle_mean", "rigid_best", "dist_rmse", "joint_err_mean", "joint_err_max",
         "rand_frags_mean", "lig_offset", "atom_offset_rms", "frag_offset_max", "seconds",
         "rmsd_min", "rmsd_mean", "dmin", "offset")  # fmt: skip


def fragments_path(path: Path) -> Path:
    return path.with_name(path.stem + ".fragments.csv")


def _parse(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in ("name", "kind"):
            out[k] = v
        elif v == "":
            out[k] = np.nan
        else:
            out[k] = float(v) if k in FLOAT else int(v)
    return out


def load(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Complex and fragment rows; a complex in several files is taken from the last one."""
    mols: dict[str, dict] = {}
    frags: dict[str, list[dict]] = {}
    for path in paths:
        fpath = fragments_path(path)
        if not fpath.exists():
            sys.exit(f"missing {fpath}")
        with open(path, newline="") as f:
            rows = [_parse(r) for r in csv.DictReader(f)]
        with open(fpath, newline="") as f:
            frag_rows = [_parse(r) for r in csv.DictReader(f)]
        for r in rows:
            mols[r["name"]] = r
            frags[r["name"]] = []
        for r in frag_rows:
            frags[r["name"]].append(r)
    for m in mols.values():
        kinds = {r["kind"] for r in frags[m["name"]]}
        m["ring_class"] = (
            "macrocycle"
            if "macrocycle" in kinds
            else ("saturated ring" if "aliphatic_ring" in kinds else "no saturated ring")
        )
        for r in frags[m["name"]]:
            r["atoms"] = m["atoms"]
    return list(mols.values()), [r for rows in frags.values() for r in rows]


def pct(a: float, b: float) -> str:
    return "  n/a" if b == 0 else f"{100 * a / b:5.1f}%"


def quantiles(values, qs=(50, 75, 90, 95, 99)) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "n/a"
    parts = "  ".join(f"p{q} {np.percentile(arr, q):6.3f}" for q in qs)
    return f"mean {arr.mean():6.3f}  {parts}  max {arr.max():6.3f}"


def tiered(rows: list[dict]):
    yield "all", rows
    for label, lo, hi in TIERS:
        yield label, [r for r in rows if lo <= r["atoms"] <= hi]


def share(rows: list[dict], pred) -> str:
    return pct(sum(1 for r in rows if pred(r)), len(rows))


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - one report, read top to bottom
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--prior_std", type=float, default=4.0, help="per-axis std of the current prior (Å)")
    p.add_argument("--top", type=int, default=15, help="how many worst complexes to list")
    args = p.parse_args(argv)

    mols, frags = load(args.csv)
    if not mols:
        sys.exit("no rows")
    embedded = [m for m in mols if m["n_conf"] > 0]
    print("=" * 86)
    print(
        f"complexes {len(mols)}   fragments {len(frags)}   "
        f"no conformer embedded {len(mols) - len(embedded)} ({pct(len(mols) - len(embedded), len(mols)).strip()})"
    )
    print("=" * 86)

    # ------------------------------------------------------------------ 1. floor
    print("\n## 1. Rigid-fragment floor (ETKDG + MMFF fragments, each Kabsch-placed on the crystal)\n")
    for label, rows in tiered(embedded):
        if not rows:
            continue
        print(f"  [{label}]  n = {len(rows)}")
        print(f"    oracle RMSD, best conformer   {quantiles([r['oracle_best'] for r in rows])}")
        print(f"    oracle RMSD, mean conformer   {quantiles([r['oracle_mean'] for r in rows])}")
        print(
            f"    share with best > 0.5 / 1 / 2 Å   {share(rows, lambda r: r['oracle_best'] > 0.5)}"
            f"  {share(rows, lambda r: r['oracle_best'] > 1.0)}  {share(rows, lambda r: r['oracle_best'] > 2.0)}"
        )
        print(f"    whole conformer rigid (contrast)  {quantiles([r['rigid_best'] for r in rows])}")
        print(f"    short-pair RMSE (< 3 Å)       {quantiles([r['dist_rmse'] for r in rows])}")
        print(f"    worst joint-pair error        {quantiles([r['joint_err_max'] for r in rows])}")
        print(f"    share with a joint pair > 0.3 Å   {share(rows, lambda r: r['joint_err_max'] > 0.3)}")
        print(
            f"    best != mean conformer (> 0.05 Å) {share(rows, lambda r: r['oracle_mean'] - r['oracle_best'] > 0.05)}"
        )
    flips = [m for m in embedded if m["chiral_flip_confs"] > 0]
    print(
        f"\n  conformer sets that invert a stereocentre (should be ~0)  {len(flips)}  {pct(len(flips), len(embedded))}"
    )
    multi = [m for m in embedded if m["n_conf"] > 1]
    if multi and "conf_spread" not in multi[0]:
        print("  WARNING: no conf_spread column -- a run from before the seed fix, where every conformer of a")
        print("           ligand was the same one. Read 'best' as a single-conformer floor, not best-of-N.")
    elif multi:
        flexible = [m for m in multi if m["torsions"] > 0]
        same = [m for m in flexible if not m["conf_spread"] > 0.01]
        print(
            f"  flexible ligands whose conformers coincide (< 0.01 Å, should be small)  {len(same)}  "
            f"{pct(len(same), len(flexible))}"
        )

    print("\n  by ring content: any macrocycle, else any saturated ring, else neither")
    print(f"    {'':24s} {'n':>6s}   {'p50':>6s} {'p95':>6s} {'p99':>6s}   {'>0.5Å':>6s} {'>1Å':>6s}   joint>0.3Å")
    for label, rows in tiered(embedded):
        for cls in RING_CLASSES:
            sub = [r for r in rows if r["ring_class"] == cls]
            if not sub:
                continue
            v = np.array([r["oracle_best"] for r in sub])
            print(
                f"    {label + ', ' + cls:24s} {len(sub):6d}   {np.percentile(v, 50):6.3f} {np.percentile(v, 95):6.3f} "
                f"{np.percentile(v, 99):6.3f}   {pct((v > 0.5).sum(), len(v))} {pct((v > 1).sum(), len(v))}   "
                f"{share(sub, lambda r: r['joint_err_max'] > 0.3)}"
            )

    print("\n  per-fragment RMSD (best conformer) by fragment kind")
    stats = " ".join(c.rjust(6) for c in ("mean", "p50", "p90", "p99"))
    print(f"    {'kind':15s} {'n':>7s} {'share':>6s}   {stats}   {'>0.3Å':>6s} {'>0.5Å':>6s}")
    with_rmsd = [r for r in frags if not np.isnan(r["rmsd_min"])]
    for kind in KINDS:
        rows = [r for r in with_rmsd if r["kind"] == kind]
        if not rows:
            continue
        v = np.array([r["rmsd_min"] for r in rows])
        print(
            f"    {kind:15s} {len(rows):7d} {pct(len(rows), len(with_rmsd))}   {v.mean():6.3f} "
            f"{np.percentile(v, 50):6.3f} {np.percentile(v, 90):6.3f} {np.percentile(v, 99):6.3f}   "
            f"{pct((v > 0.3).sum(), len(v))} {pct((v > 0.5).sum(), len(v))}"
        )
    # Which kind is the worst fragment of the complexes that miss 1 Å?
    worst_kind = defaultdict(int)
    by_name = defaultdict(list)
    for r in with_rmsd:
        by_name[r["name"]].append(r)
    bad = [m for m in embedded if m["oracle_best"] > 1.0]
    for m in bad:
        worst_kind[max(by_name[m["name"]], key=lambda r: r["rmsd_min"])["kind"]] += 1
    if bad:
        print(
            f"\n  worst fragment of the {len(bad)} complexes with floor > 1 Å:  "
            + "  ".join(f"{k} {pct(v, len(bad)).strip()}" for k, v in sorted(worst_kind.items(), key=lambda kv: -kv[1]))
        )

    print(f"\n  {args.top} complexes with the highest floor")
    print(f"    {'name':10s} {'atoms':>5s} {'frags':>5s} {'floor':>6s} {'rigid':>6s} {'joint':>6s}  worst fragment")
    for m in sorted(embedded, key=lambda m: -m["oracle_best"])[: args.top]:
        w = max(by_name[m["name"]], key=lambda r: r["rmsd_min"])
        print(
            f"    {m['name']:10s} {m['atoms']:5d} {m['fragments']:5d} {m['oracle_best']:6.2f} {m['rigid_best']:6.2f} "
            f"{m['joint_err_max']:6.2f}  {w['kind']} ({w['size']} atoms) {w['rmsd_min']:.2f} Å"
        )

    # ------------------------------------------------------------------ 2. stereo
    print("\n## 2. Stereocentres whose heavy neighbours span several fragments\n")
    for label, rows in tiered(mols):
        if not rows:
            continue
        centres = sum(r["stereocentres"] for r in rows)
        split = sum(r["split_centres"] for r in rows)
        with_c = [r for r in rows if r["stereocentres"] > 0]
        print(
            f"  [{label:>6s}]  centres {centres:6d}  split {pct(split, centres)}   "
            f"molecules with a centre {pct(len(with_c), len(rows))}   with a split centre "
            f"{share(rows, lambda r: r['split_centres'] > 0)}"
        )

    # ------------------------------------------------------------------ 3. random
    print("\n## 3. Augmentation from selection='random'\n")
    for label, rows in tiered(mols):
        if not rows:
            continue
        extra = np.array([r["rand_frags_mean"] - r["fragments"] for r in rows])
        print(
            f"  [{label:>6s}]  distinct cut sets   1: {share(rows, lambda r: r['rand_distinct'] == 1)}"
            f"   2: {share(rows, lambda r: r['rand_distinct'] == 2)}"
            f"   3-5: {share(rows, lambda r: 3 <= r['rand_distinct'] <= 5)}"
            f"   6+: {share(rows, lambda r: r['rand_distinct'] >= 6)}"
        )
        print(
            f"            fragments vs compact   mean +{extra.mean():.2f}   more on average in "
            f"{pct((extra > 0).sum(), len(rows))}   a seed with a lone atom in "
            f"{share(rows, lambda r: r['rand_lone_seeds'] > 0)}"
        )

    # ------------------------------------------------------------------ 4. symmetry
    print("\n## 4. Fragments mapped onto themselves by a proper rotation (joints held fixed)\n")
    for label, rows in tiered(frags):
        if not rows:
            continue
        order = np.array([r["sym_order"] for r in rows])
        n = len(rows)
        print(
            f"  [{label:>6s}]  symmetric {pct((order > 1).sum(), n)}   order 2 {pct((order == 2).sum(), n)}"
            f"   order 3 {pct((order == 3).sum(), n)}   order >= 4 {pct((order >= 4).sum(), n)}"
        )
    print(
        "\n  symmetric share by kind:  "
        + "   ".join(
            f"{k} {share([r for r in frags if r['kind'] == k], lambda r: r['sym_order'] > 1).strip()}"
            for k in KINDS
            if any(r["kind"] == k for r in frags)
        )
    )
    print(f"  molecules with at least one symmetric fragment  {share(mols, lambda r: r['sym_fragments'] > 0)}")

    # ------------------------------------------------------------------ 5. offsets
    print(f"\n## 5. Offsets from the pocket centre (current prior: N(pocket_com, {args.prior_std:g}² I))\n")
    tail = args.prior_std * CHI3_P95
    print(f"  the prior puts 95% of its mass within {tail:.1f} Å of the centre\n")
    for (label, rows), (_, frows) in zip(tiered(mols), tiered(frags), strict=True):
        if not rows:
            continue
        atom_rms = np.array([r["atom_offset_rms"] for r in rows])
        off = np.array([r["offset"] for r in frows])
        print(f"  [{label:>6s}]  ligand centroid  {quantiles([r['lig_offset'] for r in rows])}")
        print(f"            fragment centroid {quantiles(off)}")
        print(
            f"            per-axis sigma: atoms {np.sqrt((atom_rms**2).mean() / 3):.2f} Å   "
            f"fragment centroids {np.sqrt((off**2).mean() / 3):.2f} Å   "
            f"fragments beyond {tail:.1f} Å {pct((off > tail).sum(), len(off))}"
        )

    # ------------------------------------------------------------------ 6. exposed
    print("\n## 6. Fragments without a protein heavy atom within 4 Å\n")
    for (label, rows), (_, frows) in zip(tiered(mols), tiered(frags), strict=True):
        if not rows:
            continue
        terminal = [r for r in frows if r["attach"] <= 1]
        internal = [r for r in frows if r["attach"] >= 2]
        print(
            f"  [{label:>6s}]  exposed {share(frows, lambda r: r['contacts'] == 0)}   "
            f"terminal {share(terminal, lambda r: r['contacts'] == 0)}   "
            f"internal {share(internal, lambda r: r['contacts'] == 0)}   "
            f"molecules with one {share(rows, lambda r: r['exposed_fragments'] > 0)}"
        )
    print(
        "\n  exposed share by kind:  "
        + "   ".join(
            f"{k} {share([r for r in frags if r['kind'] == k], lambda r: r['contacts'] == 0).strip()}"
            for k in KINDS
            if any(r["kind"] == k for r in frags)
        )
    )

    # ------------------------------------------------------------------ joints cap
    print("\n## Joint count (cuts after merging) -- the cap a fragment model should filter on\n")
    for label, rows in tiered(mols):
        if not rows:
            continue
        cuts = [r["cuts"] for r in rows]
        print(f"  [{label:>6s}]  cuts {quantiles(cuts)}")
        print(
            f"            cuts > 10 / 15 / 20   {share(rows, lambda r: r['cuts'] > 10)}  "
            f"{share(rows, lambda r: r['cuts'] > 15)}  {share(rows, lambda r: r['cuts'] > 20)}"
            f"     (torsions > 20: {share(rows, lambda r: r['torsions'] > 20).strip()})"
        )

    print("\n  candidate training filters (kept share of the parsed set, and the floor of what is kept)")
    for cap in CUT_CAPS:
        for drop_macro in (False, True):
            kept = [m for m in embedded if m["cuts"] <= cap and not (drop_macro and m["ring_class"] == "macrocycle")]
            if not kept:
                continue
            v = np.array([m["oracle_best"] for m in kept])
            label = f"cuts <= {cap}" + (", no macrocycle" if drop_macro else "")
            print(
                f"    {label:28s} keep {pct(len(kept), len(embedded))}   floor p95 {np.percentile(v, 95):5.3f}  "
                f"p99 {np.percentile(v, 99):5.3f}  > 1 Å {pct((v > 1).sum(), len(v))}"
            )

    secs = np.array([m["seconds"] for m in mols])
    print(f"\nwall time per complex  {quantiles(secs)}")


if __name__ == "__main__":
    main()
