"""What the rigid-fragment model can reach on a PDBBind-style dataset, and what it has to handle.

``fragment_sweep.py`` records how ligands are cut. This script records what the cut costs and
what a fragment model will meet, per complex and per fragment:

1. **Rigid-fragment floor.** At inference a fragment's internal geometry comes from a generated
   conformer (ETKDG + MMFF, as SigmaDock does), not from the crystal. Every fragment of such a
   conformer is Kabsch-placed onto its own crystal atoms independently; the ligand RMSD of the
   result is the best any rigid-fragment model can do with that conformer. Each fragment may be
   relabelled by its graph symmetries first, as the symmetry-corrected RMSD ``dock.py`` reports
   would allow: without that, the two methyls of a gem-dimethyl carbon labelled the mirror way
   round cost 0.45 Å on 1G9V that no model could ever recover or would ever be charged for. Also
   recorded: after that placement, how far the short distances (crystal < 3 Å, the pairs
   ``flow.py``'s distance loss uses) are off -- over all such pairs, and over the pairs that
   straddle a joint.
2. **Stereocentres split across fragments.** When a centre's heavy neighbours sit in two or more
   fragments, its handedness depends on how fragments are placed relative to each other, so no
   fragment template can fix it.
3. **Augmentation from ``selection="random"``.** Distinct cut sets over ``--random_samples``
   seeds and their fragment counts, against ``compact``.
4. **Rotational self-symmetry of each fragment** (phenyl flip, carboxylate, CF3, tert-butyl):
   proper rotations that map the fragment together with its attachment atoms onto itself, atoms
   relabelled. The rotation target of such a fragment is only defined modulo that group.
5. **Fragment centroid offset from the pocket centre** (``chem.pocket_center``), the scale a
   per-fragment translation prior has to cover.
6. **Pocket contacts per fragment.** A fragment with no protein heavy atom within 4 Å is placed
   by its joints alone.

Writes ``--out`` with one row per complex and ``<out stem>.fragments.csv`` next to it with one
row per fragment; ``analyze_geometry_sweep.py`` reads both. Run from the ``pocketdit/`` directory:

    python scripts/fragment_geometry_sweep.py --root D:/Data_flie/data/pdbbind/PDBBind_processed \
        --out geom.csv --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketdit.chem import chirality_reference, load_ligand, pocket_center, read_pdb  # noqa: E402
from pocketdit.fragment import MIN_FRAGMENT_ATOMS, fragment, is_collinear  # noqa: E402

CONTACT_CUTOFF = 4.0  # Å, protein heavy atom to fragment atom
SHORT_PAIR_CUTOFF = 3.0  # Å, same pairs as FlowConfig.dist_loss_cutoff: bonds and angles
SYMMETRY_TOL = 0.3  # Å, RMSD under which a relabelled fragment counts as rotated onto itself
COM_CUTOFF = 6.0  # Å, as in chem.featurize_complex
MACROCYCLE_SIZE = 8
MAX_AUTOMORPHISMS = 10_000

COMPLEX_FIELDS = [
    "name", "atoms", "torsions", "cuts", "fragments",
    # 1. rigid-fragment floor
    "n_conf", "conf_spread", "chiral_flip_confs", "oracle_best", "oracle_mean", "rigid_best",
    "dist_rmse", "joint_err_mean", "joint_err_max",
    # 2. stereocentres
    "stereocentres", "split_centres",
    # 3. random selection
    "rand_distinct", "rand_frags_mean", "rand_frags_max", "rand_lone_seeds",
    # 4-6. per-fragment summaries
    "sym_fragments", "exposed_fragments", "lig_offset", "atom_offset_rms", "frag_offset_max",
    "seconds",
]  # fmt: skip
FRAGMENT_FIELDS = [
    "name", "frag", "size", "kind", "attach", "collinear", "sym_order",
    "rmsd_min", "rmsd_mean", "contacts", "dmin", "offset",
]  # fmt: skip


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def kabsch_place(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """``src`` moved onto ``dst`` by the proper rigid motion that minimises the RMSD.

    One- and two-atom (and collinear) sets are fine: the rotation about their own axis is then
    arbitrary, but the placed coordinates are not.
    """
    src_c, dst_c = src.mean(0), dst.mean(0)
    h = (src - src_c).T @ (dst - dst_c)
    u, _, vt = np.linalg.svd(h)
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (src - src_c) @ rot.T + dst_c


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def place_fragments(
    conf: np.ndarray, crystal: np.ndarray, frags: list[np.ndarray], perms: list[list[np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Each fragment of ``conf`` placed onto its crystal atoms on its own.

    Every fragment takes whichever of its ``local_automorphisms`` fits best, so two methyls on
    the same carbon or the two oxygens of a carboxylate are not charged for a labelling that the
    symmetry-corrected RMSD would not charge either. Returns the placed coordinates and ``conf``
    under the chosen relabelling, which is still a conformer of the same molecule.
    """
    placed, relabelled = np.empty_like(crystal), np.empty_like(conf)
    for idx, frag_perms in zip(frags, perms, strict=True):
        best_err = np.inf
        for perm in frag_perms:
            x = kabsch_place(conf[perm], crystal[idx])
            err = ((x - crystal[idx]) ** 2).sum()
            if err < best_err:
                best_err, placed[idx], relabelled[idx] = err, x, conf[perm]
    return placed, relabelled


def conformers(mol: Chem.Mol, num: int, seed: int, timeout: int) -> list[np.ndarray]:
    """Heavy-atom coordinates of ``num`` ETKDG + force-field conformers, in ``mol``'s atom order.

    Stereo is read from the crystal pose first, so every conformer is the same stereoisomer
    (``chiral_flip_confs`` checks that it actually is). Returns an empty list when embedding
    fails, times out or trips an RDKit invariant, so the rest of the complex is still measured.

    Two RDKit behaviours this guards against, both of which cost the first full-PDBBind run:

    * ``randomSeed = 0`` without sequential seeds makes ``EmbedMultipleConfs`` return ``num``
      copies of one conformer (any other seed does not). Sequential seeds give seed, seed + 1, ...
      and never degenerate; ``conf_spread`` reports the diversity so it cannot happen silently.
    * a timeout returns ``[-1]`` rather than an empty list, with no conformer behind it, so the
      ids are read back from the molecule instead of trusted.
    """
    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work)
    work = Chem.AddHs(work)  # hydrogens go after the heavy atoms; the stereo tags carry over
    # Never let the crystal pose itself pass for a generated conformer, whatever the embedder does.
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enableSequentialRandomSeeds = True
    params.numThreads = 1
    params.timeout = timeout
    ids: list[int] = []
    for random_coords in (False, True):
        params.useRandomCoords = random_coords
        try:
            AllChem.EmbedMultipleConfs(work, num, params)
        except RuntimeError:  # e.g. "bad lower bound" from the bounds-matrix builder
            return []
        ids = [c.GetId() for c in work.GetConformers()]
        if ids:
            break
    if not ids:
        return []
    try:
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMoleculeConfs(work, numThreads=1)
        elif AllChem.UFFHasAllMoleculeParams(work):
            AllChem.UFFOptimizeMoleculeConfs(work, numThreads=1)
    except Exception:  # noqa: BLE001 - an unoptimised ETKDG conformer is still a usable floor
        pass
    n = mol.GetNumAtoms()
    return [work.GetConformer(i).GetPositions()[:n] for i in ids]


def conformer_spread(confs: list[np.ndarray]) -> float | str:
    """Mean RMSD of the other conformers to the first after superposition; blank for fewer than 2.

    Near 0 for a rigid ligand, and for any ligand when the conformers are copies of one another.
    """
    if len(confs) < 2:
        return ""
    return float(np.mean([rmsd(kabsch_place(c, confs[0]), confs[0]) for c in confs[1:]]))


def chiral_flips(mol: Chem.Mol, crystal: np.ndarray, confs: list[np.ndarray]) -> int:
    """How many conformers invert at least one stereocentre of the crystal pose."""
    nbrs, sign = chirality_reference(mol, crystal)
    centres = np.flatnonzero(sign)
    if not len(centres):
        return 0
    flips = 0
    for conf in confs:
        c, a, b, d = (conf[centres], conf[nbrs[centres, 0]], conf[nbrs[centres, 1]], conf[nbrs[centres, 2]])
        vol = np.einsum("ij,ij->i", a - c, np.cross(b - c, d - c))
        flips += bool((np.sign(vol) != sign[centres]).any())
    return flips


# --------------------------------------------------------------------------------------
# Fragment properties
# --------------------------------------------------------------------------------------
def fragment_kind(mol: Chem.Mol, atoms: np.ndarray) -> str:
    """``acyclic``, ``aromatic``, ``aliphatic_ring`` (any non-aromatic ring) or ``macrocycle``.

    Ring bonds are never cut, so every ring lies wholly inside one fragment.
    """
    members = set(atoms.tolist())
    rings = [r for r in mol.GetRingInfo().AtomRings() if members.issuperset(r)]
    if not rings:
        return "acyclic"
    if any(len(r) >= MACROCYCLE_SIZE for r in rings):
        return "macrocycle"
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for r in rings for i in r):
        return "aromatic"
    return "aliphatic_ring"


def attachments(mol: Chem.Mol, frag_idx: np.ndarray, cut_bonds: tuple[int, ...], f: int) -> list[int]:
    """Atoms of other fragments bonded to fragment ``f`` across a cut."""
    out = []
    for b in cut_bonds:
        bond = mol.GetBondWithIdx(b)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frag_idx[i] == f:
            out.append(j)
        elif frag_idx[j] == f:
            out.append(i)
    return out


def local_automorphisms(mol: Chem.Mol, atoms: np.ndarray, attach: list[int]) -> list[np.ndarray]:
    """Relabellings of a fragment that keep its graph and leave its attachment atoms in place.

    Each entry ``perm`` lists, per fragment atom, the atom whose coordinates it takes over; the
    identity is one of them. Elements and connectivity only: bond orders are dropped on purpose,
    because a carboxylate, nitro or amidine group is written with one double and one single bond
    but is symmetric all the same (``rdMolAlign.CalcRMS``, which ``dock.py`` reports, treats those
    as symmetric too). Every such relabelling extends to an automorphism of the whole molecule by
    the identity elsewhere -- each attachment atom has exactly one neighbour inside the fragment,
    since two would close a ring and ring bonds are never cut -- and relabellings of different
    fragments act on disjoint atoms, so they can be chosen fragment by fragment.
    """
    ext = list(atoms.tolist()) + list(attach)
    local = {a: k for k, a in enumerate(ext)}
    sub = Chem.RWMol()
    for a in ext:
        sub.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in local and j in local:
            sub.AddBond(local[i], local[j], Chem.BondType.SINGLE)
    sub.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(sub)

    n, ext_arr = len(atoms), np.asarray(ext)
    perms = [
        ext_arr[list(match[:n])]
        for match in sub.GetSubstructMatches(sub, uniquify=False, useChirality=False, maxMatches=MAX_AUTOMORPHISMS)
        if all(match[k] == k for k in range(n, len(ext)))
    ]
    return perms or [np.asarray(atoms)]


def rotational_symmetry(
    pos: np.ndarray, atoms: np.ndarray, attach: list[int], perms: list[np.ndarray], tol: float = SYMMETRY_TOL
) -> int:
    """Order of the group of proper rotations taking the fragment onto itself, identity included.

    The fragment is taken together with its attachment atoms, which have to stay where they are:
    a rotation that moves a joint is a different pose, not an equivalent one. A relabelling from
    ``local_automorphisms`` counts only if a proper rotation actually maps the heavy-atom geometry
    onto its relabelled self. That geometric test is what rejects relabellings that are
    reflections, such as the two methyls of a gem-dimethyl carbon or the two oxygens of a
    sulfonyl, each bridging two fragments.
    """
    p = pos[np.concatenate([atoms, attach]).astype(np.int64)]
    count = 0
    for perm in perms:
        q = pos[np.concatenate([perm, attach]).astype(np.int64)]
        count += rmsd(kabsch_place(p, q), q) < tol
    return max(count, 1)


def random_selection(mol: Chem.Mol, samples: int) -> tuple[int, float, int, int]:
    """Distinct cut sets, mean and max fragment count, and seeds that leave a lone atom."""
    cut_sets, counts, lone = set(), [], 0
    for seed in range(samples):
        frag = fragment(mol, selection="random", seed=seed)
        cut_sets.add(frag.cut_bonds)
        counts.append(frag.num_fragments)
        lone += bool((frag.sizes < MIN_FRAGMENT_ATOMS).any())
    return len(cut_sets), float(np.mean(counts)), int(max(counts)), lone


# --------------------------------------------------------------------------------------
# One complex
# --------------------------------------------------------------------------------------
def measure(lig_path: Path, prot_path: Path, opts: dict) -> tuple[dict, list[dict]]:
    start = time.time()
    name = lig_path.parent.name or lig_path.stem
    mol = load_ligand(lig_path)
    crystal = mol.GetConformer().GetPositions()
    frag = fragment(mol, selection="compact")
    nf = frag.num_fragments
    frags = [np.flatnonzero(frag.frag_idx == f) for f in range(nf)]
    attach = [attachments(mol, frag.frag_idx, frag.cut_bonds, f) for f in range(nf)]
    perms = [local_automorphisms(mol, idx, att) for idx, att in zip(frags, attach, strict=True)]

    # 1. rigid-fragment floor
    confs = conformers(mol, opts["num_confs"], opts["seed"], opts["embed_timeout"])
    floor = {
        k: "" for k in ("oracle_best", "oracle_mean", "rigid_best", "dist_rmse", "joint_err_mean", "joint_err_max")
    }
    frag_rmsd = np.full((max(len(confs), 1), nf), np.nan)
    if confs:
        d_true = np.linalg.norm(crystal[:, None] - crystal[None], axis=-1)
        iu = np.triu_indices(len(crystal), k=1)
        short = d_true[iu] < SHORT_PAIR_CUTOFF
        joint = short & (frag.frag_idx[iu[0]] != frag.frag_idx[iu[1]])
        placed, relabelled = zip(*(place_fragments(c, crystal, frags, perms) for c in confs), strict=True)
        oracle = [rmsd(p, crystal) for p in placed]
        for k, p in enumerate(placed):
            frag_rmsd[k] = [rmsd(p[idx], crystal[idx]) for idx in frags]
        best = placed[int(np.argmin(oracle))]
        err = np.abs(np.linalg.norm(best[:, None] - best[None], axis=-1)[iu] - d_true[iu])
        floor.update(
            oracle_best=min(oracle),
            oracle_mean=float(np.mean(oracle)),
            rigid_best=min(rmsd(kabsch_place(c, crystal), crystal) for c in confs + list(relabelled)),
            dist_rmse=float(np.sqrt((err[short] ** 2).mean())) if short.any() else 0.0,
            joint_err_mean=float(err[joint].mean()) if joint.any() else 0.0,
            joint_err_max=float(err[joint].max()) if joint.any() else 0.0,
        )

    # 2. stereocentres
    stereo = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(stereo)
    centres = [c for c, _ in Chem.FindMolChiralCenters(stereo, includeUnassigned=False)]
    split = sum(
        any(frag.frag_idx[n.GetIdx()] != frag.frag_idx[c] for n in stereo.GetAtomWithIdx(c).GetNeighbors())
        for c in centres
    )

    # 3. random selection
    rand_distinct, rand_mean, rand_max, rand_lone = random_selection(mol, opts["random_samples"])

    # 5-6. pocket: only protein atoms near the ligand matter for both the centre and the contacts
    protein = read_pdb(prot_path)
    margin = max(COM_CUTOFF, CONTACT_CUTOFF) + 0.5
    box = np.all((protein["pos"] > crystal.min(0) - margin) & (protein["pos"] < crystal.max(0) + margin), axis=1)
    near = protein["pos"][box]
    com = pocket_center({"pos": near}, crystal, COM_CUTOFF)
    dist = np.linalg.norm(crystal[:, None] - near[None], axis=-1) if len(near) else np.full((len(crystal), 1), np.inf)

    frag_rows = []
    for f, idx in enumerate(frags):
        d = dist[idx]
        frag_rows.append(
            {
                "name": name,
                "frag": f,
                "size": len(idx),
                "kind": fragment_kind(mol, idx),
                "attach": len(attach[f]),
                "collinear": int(is_collinear(crystal[idx])),
                "sym_order": rotational_symmetry(crystal, idx, attach[f], perms[f]),
                "rmsd_min": float(np.nanmin(frag_rmsd[:, f])) if confs else "",
                "rmsd_mean": float(np.nanmean(frag_rmsd[:, f])) if confs else "",
                "contacts": int((d < CONTACT_CUTOFF).any(axis=0).sum()),
                "dmin": float(d.min()),
                "offset": float(np.linalg.norm(crystal[idx].mean(0) - com)),
            }
        )

    row = {
        "name": name,
        "atoms": mol.GetNumAtoms(),
        "torsions": len(frag.rotatable_bonds),
        "cuts": len(frag.cut_bonds),
        "fragments": nf,
        "n_conf": len(confs),
        "conf_spread": conformer_spread(confs),
        "chiral_flip_confs": chiral_flips(mol, crystal, confs),
        **floor,
        "stereocentres": len(centres),
        "split_centres": split,
        "rand_distinct": rand_distinct,
        "rand_frags_mean": rand_mean,
        "rand_frags_max": rand_max,
        "rand_lone_seeds": rand_lone,
        "sym_fragments": sum(r["sym_order"] > 1 for r in frag_rows),
        "exposed_fragments": sum(r["contacts"] == 0 for r in frag_rows),
        "lig_offset": float(np.linalg.norm(crystal.mean(0) - com)),
        "atom_offset_rms": float(np.sqrt(((crystal - com) ** 2).sum(-1).mean())),
        "frag_offset_max": max(r["offset"] for r in frag_rows),
        "seconds": round(time.time() - start, 2),
    }
    return row, frag_rows


def _job(job: tuple[Path, Path | None, dict]):
    lig_path, prot_path, opts = job
    try:
        if prot_path is None:
            raise FileNotFoundError(f"no file matching {opts['protein_glob']} next to the ligand")
        return measure(lig_path, prot_path, opts)
    except Exception as exc:  # noqa: BLE001 - one corrupt input must not stop the sweep
        return lig_path, f"{type(exc).__name__}: {exc}"


def _fmt(row: dict) -> dict:
    return {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}


def fragments_path(out: Path) -> Path:
    return out.with_name(out.stem + ".fragments.csv")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--glob", default="*/*_ligand.sdf", help="ligand glob relative to --root")
    p.add_argument("--protein_glob", default="*_protein.pdb", help="protein glob inside each ligand's folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--failures", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--prefixes", help="comma separated first letters of the complex names to keep")
    p.add_argument("--names_file", type=Path, help="only these complexes: one name per line, '#' starts a comment")
    p.add_argument("--num_confs", type=int, default=5, help="ETKDG conformers per ligand")
    p.add_argument("--embed_timeout", type=int, default=60, help="ETKDG timeout per conformer, seconds")
    p.add_argument("--random_samples", type=int, default=20, help="seeds for selection='random'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args(argv)

    paths = sorted(args.root.glob(args.glob))
    if args.names_file:
        lines = (line.split("#", 1)[0].strip() for line in args.names_file.read_text().splitlines())
        names = {line for line in lines if line}
        paths = [q for q in paths if (q.parent.name or q.stem) in names]
    if args.prefixes:
        wanted = set(args.prefixes.split(","))
        paths = [q for q in paths if (q.parent.name or q.stem)[:1] in wanted]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        p.error(f"no ligands matched {args.root / args.glob}")

    opts = {k: getattr(args, k) for k in ("num_confs", "embed_timeout", "random_samples", "seed", "protein_glob")}
    jobs = [(q, next(iter(sorted(q.parent.glob(args.protein_glob))), None), opts) for q in paths]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures, done, start = [], 0, time.time()
    with (
        open(args.out, "w", newline="") as f_mol,
        open(fragments_path(args.out), "w", newline="") as f_frag,
        Pool(args.workers) if args.workers > 1 else nullcontext() as pool,
    ):
        mol_writer = csv.DictWriter(f_mol, fieldnames=COMPLEX_FIELDS)
        frag_writer = csv.DictWriter(f_frag, fieldnames=FRAGMENT_FIELDS)
        mol_writer.writeheader()
        frag_writer.writeheader()
        # One worker runs in-process, which keeps tracebacks and debuggers usable.
        results = pool.imap_unordered(_job, jobs) if pool is not None else map(_job, jobs)
        for i, result in enumerate(results, 1):
            if isinstance(result[1], str):
                failures.append(result)
            else:
                row, frag_rows = result
                mol_writer.writerow(_fmt(row))
                frag_writer.writerows(_fmt(r) for r in frag_rows)
                done += 1
            if i % 200 == 0:
                f_mol.flush()
                f_frag.flush()
                rate = i / (time.time() - start)
                print(f"  {i}/{len(jobs)}  {rate:5.1f}/s  eta {(len(jobs) - i) / rate / 60:5.1f} min", flush=True)

    print(
        f"wrote {done} complexes to {args.out} and {fragments_path(args.out)} "
        f"({len(failures)} failures, {time.time() - start:.0f} s)"
    )
    if args.failures and failures:
        args.failures.write_text("\n".join(f"{q}\t{exc}" for q, exc in failures))
    for q, exc in failures[:10]:
        print(f"  FAIL {q.parent.name}: {exc}")


if __name__ == "__main__":
    main()
