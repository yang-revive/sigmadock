"""The PDB IDs a training run on this repository's data would actually see, and why.

A training set is not "whatever is in the data root". Between the directory and the model there
are three decisions, and each one silently drops complexes unless it is written down:

1. **Can the ligand be parsed at all?** RDKit has to kekulise and sanitise the SDF; a few hundred
   PDBBind ligands cannot be read and never reach any filter downstream.
2. **Is the ligand small enough and not too floppy?** This is the ligand-size rule the fragment
   model is sized for. Atoms are counted as heavy atoms with ``chem.load_ligand``, i.e. after
   ``Chem.RemoveHs`` -- the same count ``fragment.py`` and ``rigid.py`` report -- and torsions are
   ``fragment.fragment(...).rotatable_bonds``, the count ``data.py``'s ``--max_torsions`` uses.
3. **Does the ligand clash with a validation set?** A complex whose PDB ID is in the held-out list
   has to leave the training set or the validation number means nothing.

Nothing here is a statistic of the sort ``fragment_geometry_sweep.py`` computes: it is the index
itself, the three lists, and the counts that justify them. Writes three files next to ``--out``:

``<stem>.train.txt``    IDs that pass every rule, one per line, sorted
``<stem>.val.txt``      IDs held out (only when ``--val``/``--val_ids`` is given)
``<stem>.excluded.txt`` IDs that failed a rule, with the reason, tab separated
``<stem>.summary.json`` the counts each rule removed, and the settings that produced them

Run from the ``pocketdit/`` directory:

    python scripts/dataset_index.py --root D:/Data_flie/data/pdbbind/PDBBind_processed \
        --out data/pdbbind_v2020_50atoms/index --workers 8

The default ``--max_atoms 50`` is ``rigid.MAX_LIGAND_ATOMS``. Passing ``--max_atoms 0`` disables
the heavy-atom cap and leaves only the torsion rule, which is what SigmaDock trains on (its paper
filters at 20 torsional bonds and MW 750); ``--max_weight`` covers the weight half of that rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pocketdit import chem  # noqa: E402
from pocketdit.fragment import fragment  # noqa: E402


def index_ligand(path: Path, max_atoms: int, max_torsions: int, max_weight: float) -> dict:
    """One row: the ligand's id, its counts, and the first rule it fails (if any)."""
    name = path.parent.name or path.stem
    try:
        mol = chem.load_ligand(path)
    except Exception as exc:  # noqa: BLE001 - unreadable ligands are a result, not a crash
        return {"id": name, "status": f"parse: {type(exc).__name__}", "atoms": "", "torsions": "", "mw": ""}

    atoms = mol.GetNumAtoms()  # ``load_ligand`` already removed hydrogens
    torsions = len(fragment(mol).rotatable_bonds)
    mw = float(Chem.rdMolDescriptors.CalcExactMolWt(mol))
    row = {"id": name, "atoms": atoms, "torsions": torsions, "mw": f"{mw:.2f}"}
    if max_atoms and atoms > max_atoms:
        return {**row, "status": f"atoms: {atoms} > {max_atoms}"}
    if max_torsions and torsions > max_torsions:
        return {**row, "status": f"torsions: {torsions} > {max_torsions}"}
    if max_weight and mw > max_weight:
        return {**row, "status": f"mw: {mw:.1f} > {max_weight}"}
    return {**row, "status": "ok"}


def _job(job: tuple[Path, int, int, float]) -> dict:
    path, max_atoms, max_torsions, max_weight = job
    try:
        return index_ligand(path, max_atoms, max_torsions, max_weight)
    except Exception as exc:  # noqa: BLE001 - one bad folder must not stop the index
        return {"id": path.parent.name, "status": f"error: {type(exc).__name__}", "atoms": "", "torsions": "", "mw": ""}


def read_ids(path: Path, column: str = "") -> list[str]:
    """IDs from a text file (one per line) or a CSV (``column``, else the first column)."""
    text = path.read_text().splitlines()
    if path.suffix.lower() == ".csv":
        rows = list(csv.reader(row for row in text if row.strip() and not row.startswith("#")))
        head = [c.strip().lower() for c in rows[0]]
        i = head.index(column.lower()) if column and column.lower() in head else 0
        body = rows[1:]
        return [r[i].strip().lower() for r in body if len(r) > i and r[i].strip()]
    return [t.split()[0].strip().lower() for t in text if t.strip() and not t.startswith("#")]


def related_path(out: Path, suffix: str) -> Path:
    return out.with_name(f"{out.stem}.{suffix}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True, help="directory with one folder per complex")
    p.add_argument("--glob", default="*/*_ligand.sdf", help="ligand glob relative to --root")
    p.add_argument("--ligand_glob", help="per-complex ligand glob, resolved inside each folder; overrides --glob")
    p.add_argument("--out", type=Path, required=True, help="index file stem; three or four files are written around it")
    p.add_argument("--max_atoms", type=int, default=50, help="heavy-atom cap; 0 disables it")
    p.add_argument("--max_torsions", type=int, default=0, help="rotatable-bond cap; 0 disables it")
    p.add_argument("--max_weight", type=float, default=0.0, help="exact molecular weight cap, Da; 0 disables it")
    p.add_argument("--val", type=Path, action="append", default=[], help="hold out the complexes in this data root")
    p.add_argument("--val_glob", default="*/*_ligand.sdf", help="ligand glob used to enumerate a --val root")
    p.add_argument("--val_ids", type=Path, action="append", default=[], help="hold out the ids in this text or CSV file")
    p.add_argument("--val_column", default="", help="CSV column holding the id in --val_ids")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    paths = sorted(q for q in args.root.glob(args.glob))
    if args.ligand_glob:
        paths = sorted(next(iter(sorted(d.glob(args.ligand_glob))), None) for d in args.root.iterdir() if d.is_dir())
        paths = [q for q in paths if q is not None]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        p.error(f"no ligands matched {args.root} / {args.ligand_glob or args.glob}")

    held: dict[str, str] = {}
    for root in args.val:
        for q in sorted(root.glob(args.val_glob)):
            held[q.parent.name.lower()] = f"val: {root.name}"
    for path in args.val_ids:
        for name in read_ids(path, args.val_column):
            held[name] = f"val: {path.name}"

    opts = (args.max_atoms, args.max_torsions, args.max_weight)
    jobs = [(q, *opts) for q in paths]
    rows, start = [], time.time()
    with Pool(args.workers) if args.workers > 1 else nullcontext() as pool:
        results = pool.imap_unordered(_job, jobs) if pool is not None else map(_job, jobs)
        for i, row in enumerate(results, 1):
            name = row["id"].lower()
            if name in held and row["status"] == "ok":
                row["status"] = held[name]
            rows.append(row)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}  {i / (time.time() - start):5.1f}/s", flush=True)

    rows.sort(key=lambda r: r["id"])
    train = [r["id"] for r in rows if r["status"] == "ok"]
    val = sorted(k for k, v in held.items() if v.startswith("val") and any(r["id"].lower() == k for r in rows))
    excluded = [r for r in rows if r["status"] != "ok"]

    reasons: dict[str, int] = {}
    for r in excluded:
        reasons[r["status"].split(":")[0]] = reasons.get(r["status"].split(":")[0], 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    related_path(args.out, "train.txt").write_text("\n".join(train) + "\n")
    out_excluded = related_path(args.out, "excluded.txt")
    out_excluded.write_text("\n".join(f"{r['id']}\t{r['status']}" for r in excluded) + ("\n" if excluded else ""))
    if val:
        related_path(args.out, "val.txt").write_text("\n".join(val) + "\n")

    summary = {
        "root": str(args.root),
        "glob": args.ligand_glob or args.glob,
        "rules": {
            "max_atoms": args.max_atoms or None,
            "max_torsions": args.max_torsions or None,
            "max_weight": args.max_weight or None,
        },
        "found": len(rows),
        "train": len(train),
        "val": len(val),
        "excluded": len(excluded),
        "excluded_by_reason": reasons,
        "atoms": _percentiles([int(r["atoms"]) for r in rows if r["atoms"] != ""]),
        "torsions": _percentiles([int(r["torsions"]) for r in rows if r["torsions"] != ""]),
        "files": {
            "train": str(related_path(args.out, "train.txt")),
            "val": str(related_path(args.out, "val.txt")) if val else None,
            "excluded": str(out_excluded),
        },
        "seconds": round(time.time() - start, 1),
    }
    related_path(args.out, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"found {len(rows)}  train {len(train)}  val {len(val)}  excluded {len(excluded)}"
        f"  ({time.time() - start:.0f} s)"
    )
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:10s} {count}")
    print(f"wrote {related_path(args.out, 'train.txt')} and friends")


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    pick = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
    return {"n": len(values), "min": values[0], "p50": pick(0.50), "p95": pick(0.95), "max": values[-1]}


if __name__ == "__main__":
    sys.exit(main())
