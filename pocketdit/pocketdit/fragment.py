"""Rigid-fragment decomposition of a ligand.

Fragment-based docking keeps each fragment's internal geometry fixed -- taken straight from
the input conformer -- and only places fragments as rigid bodies. This module decides where
to cut.

The procedure follows SigmaDock: cut every rotatable bond, then merge cuts back in while the
result stays valid. Fewer and larger fragments mean fewer soft joints between them, so the
merge step is what makes the parameterisation usable.

A merge is allowed when every fragment stays rigid, i.e. no fragment larger than
``RIGID_EXEMPT_ATOMS`` still contains a rotatable bond. A bond that was rotatable in the whole
molecule often stops being one inside a fragment, because the torsion needs a heavy neighbour
on *both* sides and the cut took one away. That is exactly what makes merging possible at all.

``MIN_FRAGMENT_ATOMS`` is a property of the final state rather than of each step. Cutting all
seven torsions of decane isolates six carbons at once and no single merge repairs more than
one of them, so gating each step on the minimum size would reject every merge and leave the
molecule in pieces. Rigidity alone gates a merge; the order then puts the smallest fragments
first, which is what rescues the lone atoms. That always works: merging a bond incident to a
lone atom preserves rigidity, because that atom keeps no other neighbour inside the fragment.

Merging cannot protect a stereocentre. A tetrahedral centre has at least three heavy
neighbours, so its own rigid block already exceeds ``RIGID_EXEMPT_ATOMS`` and both sides of an
incident torsion keep a neighbour -- the torsion always survives a merge. Every
stereocentre-incident torsion is therefore cut, and handedness across that joint is left to
the model, which sees it through the signed-volume feature in ``chem.chirality_reference``.

Torsions are defined on the heavy-atom graph throughout; ``chem.load_ligand`` already removes
hydrogens. With explicit hydrogens a bond such as methanol's C-O counts as rotatable and
cutting it would isolate single heavy atoms.

Rotation about its own axis leaves the atom positions of a two-atom or collinear fragment
unchanged, so such a fragment has 5 real degrees of freedom against the 6 of SE(3). That
spare degree of freedom is a gauge redundancy, not missing information, and it has to be
pinned down numerically by whoever fits the rigid transform -- it cannot be merged away. Both
``-OCH3`` and ``-CH2CH3`` hang off the only rotatable bond of their molecule, so two-atom
fragments are routine rather than exceptional. ``is_collinear`` reports them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import TorsionFingerprints

MIN_FRAGMENT_ATOMS = 2  # a single atom has no orientation to place
RIGID_EXEMPT_ATOMS = 3  # fragments this small cannot hold a meaningful torsion
COLLINEAR_TOL = 0.1  # Å, second singular value of the centred fragment coordinates

SELECTIONS = ("compact", "random", "none")


@dataclass(frozen=True)
class Fragmentation:
    """Which heavy atom belongs to which rigid fragment."""

    frag_idx: np.ndarray  # [n] int64, fragment index per heavy atom, 0-based and contiguous
    cut_bonds: tuple[int, ...]  # bond indices that were cut
    rotatable_bonds: tuple[int, ...]  # every rotatable bond, cut or merged back

    @property
    def num_fragments(self) -> int:
        return int(self.frag_idx.max()) + 1

    @property
    def sizes(self) -> np.ndarray:
        """[num_fragments] number of heavy atoms per fragment."""
        return np.bincount(self.frag_idx, minlength=self.num_fragments)


# --------------------------------------------------------------------------------------
# Rotatable bonds
# --------------------------------------------------------------------------------------
def rotatable_bonds(mol: Chem.Mol, ignore_conjugated: bool = False) -> list[int]:
    """Non-ring single bonds that carry a torsion on the heavy-atom graph.

    ``ignore_conjugated`` skips conjugated bonds whose two atoms are both outside rings, i.e.
    weak conjugation such as an ester C-O; conjugation stemming from a ring is kept.
    """
    non_ring, _ = TorsionFingerprints.CalculateTorsionLists(mol)
    out: set[int] = set()
    for quads, _ in non_ring:
        for quad in quads:
            bond = mol.GetBondBetweenAtoms(quad[1], quad[2])
            if bond is None or bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if ignore_conjugated and bond.GetIsConjugated():
                begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
                if not begin.IsInRing() and not end.IsInRing():
                    continue
            out.add(bond.GetIdx())
    return sorted(out)


# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------
def _neighbours(mol: Chem.Mol) -> list[set[int]]:
    nbrs: list[set[int]] = [set() for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        nbrs[i].add(j)
        nbrs[j].add(i)
    return nbrs


def _components(mol: Chem.Mol, cut: set[int]) -> np.ndarray:
    """Fragment index per atom once the bonds in ``cut`` are removed."""
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        if bond.GetIdx() in cut:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)
    frag = np.full(n, -1, dtype=np.int64)
    count = 0
    for seed in range(n):
        if frag[seed] >= 0:
            continue
        frag[seed] = count
        stack = [seed]
        while stack:
            atom = stack.pop()
            for nbr in adj[atom]:
                if frag[nbr] < 0:
                    frag[nbr] = count
                    stack.append(nbr)
        count += 1
    return frag


def _is_rigid(frag_idx: np.ndarray, ends: list[tuple[int, int]], nbrs: list[set[int]]) -> bool:
    """Whether no fragment past the size exemption still contains a rotatable bond."""
    sizes = np.bincount(frag_idx)
    for u, v in ends:
        frag = frag_idx[u]
        if frag != frag_idx[v]:
            continue  # the bond is cut, it is a joint rather than an internal torsion
        if sizes[frag] <= RIGID_EXEMPT_ATOMS:
            continue
        # The torsion survives inside the fragment only with a heavy neighbour on both sides.
        if any(frag_idx[w] == frag for w in nbrs[u] if w != v) and any(frag_idx[w] == frag for w in nbrs[v] if w != u):
            return False
    return True


def _mergeable(mol: Chem.Mol, bonds: list[int], ends: dict[int, tuple[int, int]]) -> list[int]:
    """Rotatable bonds that are worth trying to merge back.

    The rest have to stay cut:

    * a bridge between two ring systems -- merging would fuse both rings into one fragment
      with the bond still rotatable inside it;
    * the only rotatable bond at both of its atoms -- each side is then a maximal rigid blob
      of at least two atoms, so merging always leaves the bond rotatable inside the result.
    """
    degree: dict[int, int] = defaultdict(int)
    for bond in bonds:
        u, v = ends[bond]
        degree[u] += 1
        degree[v] += 1

    out = []
    for bond in bonds:
        u, v = ends[bond]
        if mol.GetAtomWithIdx(u).IsInRing() and mol.GetAtomWithIdx(v).IsInRing():
            continue
        if degree[u] == 1 and degree[v] == 1:
            continue
        out.append(bond)
    return out


# --------------------------------------------------------------------------------------
# Fragmentation
# --------------------------------------------------------------------------------------
def fragment(
    mol: Chem.Mol,
    selection: str = "compact",
    seed: int | None = None,
    ignore_conjugated: bool = False,
) -> Fragmentation:
    """Decompose ``mol`` into rigid fragments.

    ``selection`` picks the order in which cuts are tried for merging:

    * ``"compact"`` -- try both merge heuristics and keep the better state by ``_score``.
      Deterministic, and the choice for inference. It minimises lone atoms before the fragment
      count, and it is greedy rather than optimal: decane comes out as five two-atom fragments
      because neither heuristic finds the ``[3, 3, 2, 2]`` that also avoids a lone atom.
    * ``"random"`` -- a random merge order, so the same molecule is split differently on every
      call. The choice for training, where it acts as free augmentation.
    * ``"none"`` -- cut every rotatable bond and merge nothing. For ablations. This is the one
      setting that can leave a fragment below ``MIN_FRAGMENT_ATOMS``.
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")

    bonds = rotatable_bonds(mol, ignore_conjugated)
    if not bonds:
        return Fragmentation(np.zeros(mol.GetNumAtoms(), dtype=np.int64), (), ())

    ends = {b: (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx()) for b in bonds}
    end_list = [ends[b] for b in bonds]
    nbrs = _neighbours(mol)

    cut = set(bonds)
    if selection == "none":
        return Fragmentation(_components(mol, cut), tuple(sorted(cut)), tuple(bonds))

    pending = _mergeable(mol, bonds, ends)
    if selection == "random":
        random.Random(seed).shuffle(pending)
    # Both heuristics walk into a dead end on some molecules and neither dominates, so run
    # each and keep whichever scores better. That is a handful of breadth-first searches over
    # at most a few dozen bonds, which costs nothing. Under ``"random"`` the shuffled order is
    # only the tie-breaker, so running both does not cost any variety.
    states = [_merge(mol, cut, pending, ends, end_list, nbrs, rank) for rank in (_fill_rank, _smallest_rank)]
    frag_idx, cut = min(states, key=lambda s: _score(s[0]))
    return Fragmentation(frag_idx, tuple(sorted(cut)), tuple(bonds))


def _smallest_rank(size_sum: int) -> tuple[int, ...]:
    """Merge the two smallest fragments first."""
    return (size_sum,)


def _fill_rank(size_sum: int) -> tuple[int, ...]:
    """Fill a fragment up to the rigidity exemption first.

    A merge whose result stays within ``RIGID_EXEMPT_ATOMS`` can never be rejected, so taking
    the largest such merge packs atoms efficiently. It can also strand an atom: greedily
    making three-atom fragments out of decane leaves ``[3, 3, 1, 3]``, where ``[3, 3, 2, 2]``
    was available. ``_smallest_rank`` covers that case.
    """
    return (0, -size_sum) if size_sum <= RIGID_EXEMPT_ATOMS else (1, size_sum)


def _score(frag_idx: np.ndarray) -> tuple[int, int]:
    """Lower is better: avoid lone atoms first, then use as few fragments as possible.

    An atom on its own has every one of its bonds left soft, which is worse than carrying one
    more joint, so the fragment count is only the tie-breaker.
    """
    sizes = np.bincount(frag_idx)
    return int((sizes < MIN_FRAGMENT_ATOMS).sum()), len(sizes)


def _merge(
    mol: Chem.Mol,
    cut: set[int],
    pending: list[int],
    ends: dict[int, tuple[int, int]],
    end_list: list[tuple[int, int]],
    nbrs: list[set[int]],
    rank,
) -> tuple[np.ndarray, set[int]]:
    """Remove cuts one at a time, in ``rank`` order, while every fragment stays rigid.

    Merging only grows fragments, so a torsion is only ever more likely to survive afterwards:
    a candidate rejected once stays rejected, and the loop ends once a full scan accepts
    nothing. A fragment below the minimum size is always attacked first, whatever ``rank``
    prefers, because the chance is fleeting -- once its neighbours have grown past the
    rigidity exemption it can never be merged at all. Decane is the case to keep in mind:
    cutting its seven torsions isolates six carbons at once.
    """
    cut, pending = set(cut), list(pending)
    order = {bond: i for i, bond in enumerate(pending)}
    frag_idx = _components(mol, cut)

    def key(bond: int) -> tuple:
        size_u, size_v = sizes[frag_idx[ends[bond][0]]], sizes[frag_idx[ends[bond][1]]]
        rescue = min(size_u, size_v) >= MIN_FRAGMENT_ATOMS  # False sorts first
        return (rescue, rank(int(size_u + size_v)), order[bond])

    while pending:
        sizes = np.bincount(frag_idx)
        pending.sort(key=key)
        accepted, candidate = None, None
        for bond in pending:
            trial = _components(mol, cut - {bond})
            if _is_rigid(trial, end_list, nbrs):
                accepted, candidate = bond, trial
                break
        if accepted is None:
            break
        cut.discard(accepted)
        pending.remove(accepted)
        frag_idx = candidate

    return frag_idx, cut


def is_collinear(pos: np.ndarray, tol: float = COLLINEAR_TOL) -> bool:
    """Whether a fragment's reference geometry leaves a rotation about its own axis free.

    True for one- and two-atom fragments and for collinear ones such as a nitrile. Their atom
    positions are fully determined all the same; only the rigid-transform parameterisation
    carries a spare degree of freedom.
    """
    if len(pos) < 3:
        return True
    singular = np.linalg.svd(pos - pos.mean(axis=0), compute_uv=False)
    return bool(singular[1] < tol)


def fragment_positions(frag_idx: np.ndarray, pos: np.ndarray) -> list[np.ndarray]:
    """Coordinates of each fragment, in fragment-index order."""
    return [pos[frag_idx == f] for f in range(int(frag_idx.max()) + 1)]


# --------------------------------------------------------------------------------------
# Statistics over a dataset
# --------------------------------------------------------------------------------------
def _percentiles(values: list[float] | np.ndarray) -> str:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "n/a"
    q = np.percentile(arr, [50, 90, 99])
    return f"mean {arr.mean():6.2f}  median {q[0]:5.1f}  p90 {q[1]:5.1f}  p99 {q[2]:5.1f}  max {arr.max():5.0f}"


def statistics(paths: list[Path], selection: str = "compact", max_torsions: int | None = None) -> dict:
    """Fragment statistics over a list of ligand files, printed by ``main``."""
    from pocketdit.chem import load_ligand

    rows, failures, over_limit = [], [], 0
    for path in paths:
        try:
            mol = load_ligand(path)
            frag = fragment(mol, selection=selection)
        except Exception as exc:  # noqa: BLE001 - a corrupt input should not stop the sweep
            failures.append((path, str(exc)))
            continue
        if max_torsions is not None and len(frag.rotatable_bonds) > max_torsions:
            over_limit += 1
        pos = mol.GetConformer().GetPositions()
        sizes = frag.sizes
        rows.append(
            {
                "name": path.parent.name or path.stem,
                "atoms": mol.GetNumAtoms(),
                "torsions": len(frag.rotatable_bonds),
                "cuts": len(frag.cut_bonds),
                "fragments": frag.num_fragments,
                "sizes": sizes,
                "collinear": sum(is_collinear(p) for p in fragment_positions(frag.frag_idx, pos)),
            }
        )
    return {"rows": rows, "failures": failures, "over_limit": over_limit, "selection": selection}


def report(stats: dict, max_torsions: int | None = None) -> None:
    rows = stats["rows"]
    if not rows:
        print("no molecules parsed")
        for path, exc in stats["failures"][:10]:
            print(f"  {path}: {exc}")
        return

    sizes = np.concatenate([r["sizes"] for r in rows])
    n_frags = sum(r["fragments"] for r in rows)
    print(f"molecules      {len(rows)}  (failed to parse: {len(stats['failures'])})")
    print(f"selection      {stats['selection']}")
    print(f"heavy atoms    {_percentiles([r['atoms'] for r in rows])}")
    print(f"torsions       {_percentiles([r['torsions'] for r in rows])}")
    print(f"fragments      {_percentiles([r['fragments'] for r in rows])}")
    print(f"fragment size  {_percentiles(sizes)}")
    print(f"cuts merged    {sum(r['torsions'] - r['cuts'] for r in rows)} of {sum(r['torsions'] for r in rows)}")
    print()
    print("fragment size histogram")
    hist = np.bincount(sizes)
    scale = max(hist[1:].max(), 1)
    for size, count in enumerate(hist):
        if size and count:
            bar = "#" * min(60, count * 60 // scale)
            print(f"  {size:3d} atoms  {count:6d}  {100 * count / n_frags:5.1f}%  {bar}")
    two_atom = int((sizes <= 2).sum())
    collinear = sum(r["collinear"] for r in rows)
    print()
    print(f"fragments with <= 2 atoms  {two_atom:6d}  {100 * two_atom / n_frags:5.1f}%")
    print(f"collinear fragments        {collinear:6d}  {100 * collinear / n_frags:5.1f}%  (gauge-degenerate)")
    print(f"single-fragment molecules  {sum(r['fragments'] == 1 for r in rows):6d}")
    if max_torsions is not None:
        over = stats["over_limit"]
        print(f"torsions > {max_torsions:<15d}  {over:6d}  {100 * over / len(rows):5.1f}%")
    if stats["failures"]:
        print()
        print("parse failures")
        for path, exc in stats["failures"][:10]:
            print(f"  {path}: {exc}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fragment statistics over a ligand dataset")
    p.add_argument("--root", type=Path, help="dataset root, one subfolder per complex")
    p.add_argument("--glob", default="*/*_ligand.sdf", help="ligand glob relative to --root")
    p.add_argument("--smiles", nargs="*", help="fragment these SMILES instead of a dataset")
    p.add_argument("--selection", default="compact", choices=SELECTIONS)
    p.add_argument("--max_torsions", type=int, default=20, help="report how many exceed this")
    p.add_argument("--limit", type=int, help="only look at the first N ligands")
    args = p.parse_args(argv)

    if args.smiles:
        from rdkit.Chem import AllChem

        for smiles in args.smiles:
            mol = Chem.MolFromSmiles(smiles)
            mol_h = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_h, randomSeed=0)
            mol = Chem.RemoveHs(mol_h)
            frag = fragment(mol, selection=args.selection)
            pos = mol.GetConformer().GetPositions()
            degenerate = [i for i, p in enumerate(fragment_positions(frag.frag_idx, pos)) if is_collinear(p)]
            print(f"{smiles}")
            print(f"  atoms {mol.GetNumAtoms()}  torsions {len(frag.rotatable_bonds)}  cuts {len(frag.cut_bonds)}")
            print(f"  {frag.num_fragments} fragments, sizes {frag.sizes.tolist()}, collinear {degenerate}")
            print(f"  frag_idx {frag.frag_idx.tolist()}")
        return

    if not args.root:
        p.error("pass either --root or --smiles")
    paths = sorted(args.root.glob(args.glob))[: args.limit]
    if not paths:
        p.error(f"no ligands matched {args.root / args.glob}")
    report(statistics(paths, args.selection, args.max_torsions), args.max_torsions)


if __name__ == "__main__":
    main()
