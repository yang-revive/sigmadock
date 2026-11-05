"""Fragmentation invariants, checked independently of how ``fragment`` reaches them."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from conftest import DUMMY
from pocketdit.chem import load_ligand
from pocketdit.fragment import (
    MIN_FRAGMENT_ATOMS,
    RIGID_EXEMPT_ATOMS,
    SELECTIONS,
    fragment,
    fragment_positions,
    is_collinear,
    rotatable_bonds,
)

# A spread of motifs: terminal two-atom groups, ring bridges, stereocentres, long chains,
# a macrocycle, and drug-like ligands.
CORPUS = [
    "CCc1ccccc1",  # ethylbenzene: the only torsion must be cut, leaving a two-atom ethyl
    "COc1ccccc1",  # anisole: same, two-atom methoxy
    "Cc1ccccc1",  # toluene: no torsion at all
    "CCCCC",  # pentane: cutting both torsions would isolate the middle carbon
    "CCCCCCCCCC",
    "c1ccccc1",
    "CC#N",
    "OCc1ccccc1",
    "CSc1ccccc1",
    "CN(C)c1ccccc1",
    "c1ccc(-c2ccccc2)cc1",  # biphenyl: a ring bridge
    "CC(C)(C)c1ccc(OC)cc1",
    "C[C@H](N)C(=O)O",  # alanine: a stereocentre
    "C[C@@H](O)[C@H](N)C(=O)O",  # two adjacent stereocentres
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "C1CCCCCCCCC1",  # macrocycle: every bond is in the ring
    "NC(=O)CNC(=O)CNC(=O)CN",  # peptide-like backbone
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
]


def embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0, smiles
    return Chem.RemoveHs(mol)


@pytest.fixture(scope="module")
def corpus() -> list[tuple[str, Chem.Mol]]:
    return [(s, embed(s)) for s in CORPUS]


@pytest.fixture(scope="module")
def ligands() -> list[tuple[str, Chem.Mol]]:
    if not DUMMY.exists():
        pytest.skip(f"example data not found at {DUMMY}")
    return [(p.parent.name, load_ligand(p)) for p in sorted(DUMMY.glob("*/*_ligand.sdf"))]


def submol(mol: Chem.Mol, atoms: np.ndarray) -> Chem.Mol:
    """``mol`` restricted to ``atoms``, with the severed valences filled by implicit Hs."""
    keep = set(int(a) for a in atoms)
    editable = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        editable.RemoveAtom(idx)
    out = editable.GetMol()
    Chem.SanitizeMol(out, catchErrors=True)
    return out


# --------------------------------------------------------------------------------------
# Known molecules
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("smiles", "num_torsions", "sizes"),
    [
        ("CCc1ccccc1", 1, [2, 6]),  # a two-atom fragment with nothing to merge it into
        ("COc1ccccc1", 1, [2, 6]),
        ("Cc1ccccc1", 0, [7]),  # a terminal methyl carries no torsion
        ("c1ccccc1", 0, [6]),
        ("CC#N", 0, [3]),
        ("CCCCC", 2, [3, 2]),  # one cut merged back, so no atom is left on its own
        ("c1ccc(-c2ccccc2)cc1", 1, [6, 6]),  # a ring bridge always stays cut
    ],
)
def test_known_fragmentations(smiles: str, num_torsions: int, sizes: list[int]) -> None:
    frag = fragment(embed(smiles))
    assert len(frag.rotatable_bonds) == num_torsions
    assert sorted(frag.sizes.tolist()) == sorted(sizes)


def test_ethylbenzene_cannot_avoid_a_two_atom_fragment() -> None:
    """Its single torsion has to be cut, so no merge policy can reach three atoms a side."""
    mol = embed("CCc1ccccc1")
    for selection in SELECTIONS:
        frag = fragment(mol, selection=selection, seed=0)
        assert frag.sizes.min() == 2
        assert len(frag.cut_bonds) == 1


# --------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------
@pytest.fixture(params=SELECTIONS)
def selection(request: pytest.FixtureRequest) -> str:
    return request.param


def check_invariants(mol: Chem.Mol, frag, selection: str) -> None:
    name = Chem.MolToSmiles(mol)
    frag_idx, sizes = frag.frag_idx, frag.sizes

    # frag_idx is a contiguous 0-based labelling of every atom.
    assert frag_idx.shape == (mol.GetNumAtoms(),)
    assert set(frag_idx.tolist()) == set(range(frag.num_fragments)), name

    # Every fragment is rigid, checked with RDKit's own torsion detection on the extracted
    # fragment rather than with the structural shortcut ``fragment`` uses internally.
    for f in range(frag.num_fragments):
        atoms = np.flatnonzero(frag_idx == f)
        if len(atoms) <= RIGID_EXEMPT_ATOMS:
            continue
        assert not rotatable_bonds(submol(mol, atoms)), f"{name}: fragment {f} is not rigid"

    # A cut bond joins two fragments; a merged-back one stays inside one.
    merged = set(frag.rotatable_bonds) - set(frag.cut_bonds)
    for bond_idx in frag.rotatable_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        same = frag_idx[u] == frag_idx[v]
        assert same == (bond_idx in merged), name

    if selection == "none":
        # Merging nothing is the one setting allowed to leave a lone atom behind.
        assert not merged, name
    else:
        assert sizes.min() >= MIN_FRAGMENT_ATOMS, f"{name}: fragment sizes {sizes.tolist()}"


def test_invariants_on_corpus(corpus: list[tuple[str, Chem.Mol]], selection: str) -> None:
    for _, mol in corpus:
        check_invariants(mol, fragment(mol, selection=selection, seed=0), selection)


def test_invariants_on_real_ligands(ligands: list[tuple[str, Chem.Mol]], selection: str) -> None:
    for _, mol in ligands:
        check_invariants(mol, fragment(mol, selection=selection, seed=0), selection)


def test_invariants_hold_for_every_random_seed(corpus: list[tuple[str, Chem.Mol]]) -> None:
    for _, mol in corpus:
        for seed in range(8):
            check_invariants(mol, fragment(mol, selection="random", seed=seed), "random")


# --------------------------------------------------------------------------------------
# Merge behaviour
# --------------------------------------------------------------------------------------
def test_merging_never_increases_the_fragment_count(corpus: list[tuple[str, Chem.Mol]]) -> None:
    for smiles, mol in corpus:
        unmerged = fragment(mol, selection="none").num_fragments
        for selection in ("compact", "random"):
            assert fragment(mol, selection=selection, seed=0).num_fragments <= unmerged, smiles


def test_compact_trades_a_fragment_for_avoiding_a_lone_atom() -> None:
    """Fewer fragments is not the objective; no atom with every bond soft is.

    Cutting decane's seven torsions isolates six carbons. Filling fragments up to the rigidity
    exemption reaches four fragments but only as ``[3, 3, 1, 3]``, leaving one carbon with both
    of its bonds across a joint, so ``compact`` takes five two-atom fragments instead -- each
    of those at least holds one bond rigid. The merge is greedy, not optimal: ``[3, 3, 2, 2]``
    would satisfy both goals and some random orders do find it.
    """
    mol = embed("CCCCCCCCCC")
    assert fragment(mol, selection="none").num_fragments == 8
    assert sorted(fragment(mol, selection="compact").sizes.tolist()) == [2, 2, 2, 2, 2]
    found = {tuple(sorted(fragment(mol, selection="random", seed=s).sizes.tolist())) for s in range(16)}
    assert (2, 2, 3, 3) in found


def test_compact_is_deterministic(corpus: list[tuple[str, Chem.Mol]]) -> None:
    for _, mol in corpus:
        first = fragment(mol, selection="compact")
        assert np.array_equal(first.frag_idx, fragment(mol, selection="compact").frag_idx)


def test_random_is_reproducible_and_varies(ligands: list[tuple[str, Chem.Mol]]) -> None:
    repeated = [fragment(mol, selection="random", seed=3).cut_bonds for _, mol in ligands]
    assert repeated == [fragment(mol, selection="random", seed=3).cut_bonds for _, mol in ligands]
    # At least one flexible ligand must be split differently by some other seed, otherwise
    # random fragmentation is not buying any augmentation.
    assert any(
        fragment(mol, selection="random", seed=seed).cut_bonds != cuts
        for seed in range(1, 12)
        for (_, mol), cuts in [(pair, cut) for pair, cut in zip(ligands, repeated, strict=True)]
    )


def test_pentane_merges_rather_than_isolating_an_atom() -> None:
    mol = embed("CCCCC")
    assert len(rotatable_bonds(mol)) == 2
    assert len(fragment(mol, selection="none").cut_bonds) == 2
    assert np.bincount(fragment(mol, selection="none").frag_idx).min() == 1  # the invalid state
    assert fragment(mol).sizes.min() == 2  # merging repairs it


def test_macrocycle_is_one_rigid_fragment() -> None:
    assert fragment(embed("C1CCCCCCCCC1")).num_fragments == 1


def test_stereocentre_torsions_are_always_cut(corpus: list[tuple[str, Chem.Mol]]) -> None:
    """Merging can never protect a stereocentre, so handedness is always left to the model.

    A tetrahedral centre has at least three heavy neighbours, so its own rigid block is
    already larger than the size exemption and both sides of an incident torsion keep a
    neighbour: such a torsion survives every merge and therefore stays cut. Pinning this down
    matters because it is what makes the signed-volume chirality feature load-bearing in
    fragment mode too, exactly as in all-atom mode.
    """
    seen = 0
    for smiles, mol in corpus:
        centres = {int(idx) for idx, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True)}
        if not centres:
            continue
        for selection in ("compact", "random"):
            frag = fragment(mol, selection=selection, seed=0)
            for bond_idx in frag.rotatable_bonds:
                bond = mol.GetBondWithIdx(bond_idx)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in centres or v in centres:
                    assert bond_idx in frag.cut_bonds, f"{smiles}: torsion {(u, v)} was merged"
                    seen += 1
    assert seen, "the corpus has no stereocentre-incident torsion to check"


# --------------------------------------------------------------------------------------
# Gauge degeneracy
# --------------------------------------------------------------------------------------
def test_is_collinear() -> None:
    assert is_collinear(np.zeros((1, 3)))
    assert is_collinear(np.array([[0.0, 0, 0], [1.5, 0, 0]]))
    assert is_collinear(np.array([[0.0, 0, 0], [1.2, 0, 0], [2.4, 0, 0]]))  # a nitrile
    assert not is_collinear(np.array([[0.0, 0, 0], [1.5, 0, 0], [0.7, 1.3, 0]]))
    ring = embed("c1ccccc1").GetConformer().GetPositions()
    assert not is_collinear(ring)


def test_two_atom_and_nitrile_fragments_are_reported_degenerate() -> None:
    for smiles in ("CCc1ccccc1", "COc1ccccc1"):
        mol = embed(smiles)
        frag = fragment(mol)
        parts = fragment_positions(frag.frag_idx, mol.GetConformer().GetPositions())
        degenerate = [i for i, p in enumerate(parts) if is_collinear(p)]
        assert len(degenerate) == 1 and len(parts[degenerate[0]]) == 2, smiles


def test_real_ligands_stay_within_one_fragment_per_atom(ligands: list[tuple[str, Chem.Mol]]) -> None:
    """Fragment coordinates partition the molecule, so a rigid fit can reconstruct it exactly."""
    for name, mol in ligands:
        frag = fragment(mol)
        pos = mol.GetConformer().GetPositions()
        assert sum(len(p) for p in fragment_positions(frag.frag_idx, pos)) == len(pos), name
