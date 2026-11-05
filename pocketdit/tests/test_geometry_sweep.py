"""The measurements of ``scripts/fragment_geometry_sweep.py`` on cases whose answer is known."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from conftest import DUMMY
from pocketdit.chem import load_ligand
from pocketdit.fragment import fragment

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load("fragment_geometry_sweep")
analyze = _load("analyze_geometry_sweep")


def embedded(smiles: str, seed: int = 0) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def parts(mol: Chem.Mol):
    """Fragmentation, per-fragment atoms, attachment atoms and local automorphisms."""
    frag = fragment(mol)
    atoms = [np.flatnonzero(frag.frag_idx == f) for f in range(frag.num_fragments)]
    attach = [sweep.attachments(mol, frag.frag_idx, frag.cut_bonds, f) for f in range(frag.num_fragments)]
    perms = [sweep.local_automorphisms(mol, a, t) for a, t in zip(atoms, attach, strict=True)]
    return frag, atoms, attach, perms


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


# --------------------------------------------------------------------------------------
# Rigid placement
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 12])
def test_kabsch_recovers_a_rigid_motion(n):
    rng = np.random.default_rng(n)
    src = rng.normal(size=(n, 3)) * 2
    dst = src @ random_rotation(rng).T + rng.normal(size=3) * 5
    assert np.allclose(sweep.kabsch_place(src, dst), dst, atol=1e-6)


def test_kabsch_never_reflects():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(6, 3))
    mirror = src * np.array([1.0, 1.0, -1.0])
    assert sweep.rmsd(sweep.kabsch_place(src, mirror), mirror) > 0.1


def test_floor_ignores_independent_fragment_motions():
    """Fragments are placed one by one, so moving each rigidly on its own costs nothing."""
    mol = load_ligand(DUMMY / "1R1H_BIR" / "1R1H_BIR_ligand.sdf")
    crystal = mol.GetConformer().GetPositions()
    _, atoms, _, perms = parts(mol)
    rng = np.random.default_rng(0)
    conf = crystal.copy()
    for idx in atoms:
        conf[idx] = crystal[idx] @ random_rotation(rng).T + rng.normal(size=3) * 10
    placed, _ = sweep.place_fragments(conf, crystal, atoms, perms)
    assert sweep.rmsd(placed, crystal) < 1e-6


def test_floor_forgives_a_mirror_labelled_gem_dimethyl():
    """Swapping the two methyl labels is a reflection, which Kabsch alone cannot undo.

    The symmetry-corrected RMSD would not charge for it, so the floor must not either.
    """
    mol = embedded("CC(C)(Oc1ccccc1)C(=O)O")
    crystal = mol.GetConformer().GetPositions()
    _, atoms, _, perms = parts(mol)
    methyls = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1 and a.GetSymbol() == "C"]
    assert len(methyls) == 2
    conf = crystal.copy()
    conf[methyls] = crystal[methyls[::-1]]

    identity = [[idx] for idx in atoms]
    naive, _ = sweep.place_fragments(conf, crystal, atoms, identity)
    placed, relabelled = sweep.place_fragments(conf, crystal, atoms, perms)
    gem = next(idx for idx in atoms if set(methyls) <= set(idx.tolist()))
    assert sweep.rmsd(naive[gem], crystal[gem]) > 0.3
    assert sweep.rmsd(placed, crystal) < 1e-6
    assert np.allclose(np.sort(relabelled, axis=0), np.sort(conf, axis=0))  # same atoms, relabelled


# --------------------------------------------------------------------------------------
# Rotational symmetry
# --------------------------------------------------------------------------------------
SYMMETRY = [
    ("CCc1ccccc1", {"c1ccccc1": 2, "CC": 1}),  # terminal phenyl flips about its attachment axis
    ("CC(C)(C)c1ccccc1", {"CC(C)C": 3, "c1ccccc1": 2}),
    ("FC(F)(F)c1ccccc1", {"FC(F)F": 3}),
    ("OC(=O)c1ccccc1", {"O=CO": 2}),  # written with one C=O, symmetric all the same
    ("[O-][N+](=O)c1ccccc1", {"O=[N+][O-]": 2}),
    ("NC(=N)c1ccccc1", {"N=CN": 2}),
    ("CCc1ccc(CC)cc1", {"c1ccccc1": 2}),  # para: the flip axis passes through both joints
    ("CCc1cccc(CC)c1", {"c1ccccc1": 1}),  # meta: the only relabelling is a reflection
    ("CC(C)(Oc1ccccc1)C(=O)O", {"CC(C)O": 1}),  # gem-dimethyl between two joints: reflection
    ("c1ccc(S(=O)(=O)Nc2ccccc2)cc1", {"NS(=O)=O": 1}),  # sulfonyl between two joints: reflection
    ("C1CCCCC1CC", {"C1CCCCC1": 1}),  # a chair's mirror relabelling is not a rotation
]


@pytest.mark.parametrize("smiles,expected", SYMMETRY)
@pytest.mark.parametrize("seed", [0, 1])
def test_rotational_symmetry(smiles, expected, seed):
    mol = embedded(smiles, seed)
    pos = mol.GetConformer().GetPositions()
    _, atoms, attach, perms = parts(mol)
    found = {
        Chem.MolFragmentToSmiles(mol, atomsToUse=idx.tolist()): sweep.rotational_symmetry(pos, idx, att, p)
        for idx, att, p in zip(atoms, attach, perms, strict=True)
    }
    for frag_smiles, order in expected.items():
        assert found[frag_smiles] == order, (frag_smiles, found)


def test_fragment_kind():
    # atoms 0-5 cyclohexane, 6 CH2, 7-12 benzene, 13 O, 14 CH2, 15-23 cyclononane, 24-33 tetralin
    mol = Chem.MolFromSmiles("C1CCCCC1Cc1ccccc1OCC1CCCCCCCC1.c1ccc2c(c1)CCCC2")
    kind = lambda atoms: sweep.fragment_kind(mol, np.array(atoms))  # noqa: E731
    assert kind(range(6)) == "aliphatic_ring"
    assert kind(range(6, 7)) == kind([13, 14]) == "acyclic"
    assert kind(range(7, 13)) == "aromatic"
    assert kind(range(15, 24)) == "macrocycle"
    assert kind(range(24, 34)) == "aliphatic_ring"  # one saturated ring is enough
    assert kind(range(7, 13 + 1)) == "aromatic"  # an exocyclic atom does not change the kind


# --------------------------------------------------------------------------------------
# A whole complex and the command line
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def measured():
    if not DUMMY.exists():
        pytest.skip(f"example data not found at {DUMMY}")
    opts = {"num_confs": 3, "embed_timeout": 30, "random_samples": 5, "seed": 0, "protein_glob": "*_protein.pdb"}
    folder = DUMMY / "1R1H_BIR"
    return sweep.measure(folder / "1R1H_BIR_ligand.sdf", folder / "1R1H_BIR_protein.pdb", opts)


def test_measure_one_complex(measured):
    row, frag_rows = measured
    assert set(row) == set(sweep.COMPLEX_FIELDS)
    assert all(set(r) == set(sweep.FRAGMENT_FIELDS) for r in frag_rows)
    assert row["n_conf"] == 3 and row["chiral_flip_confs"] == 0
    assert row["oracle_best"] <= row["oracle_mean"] < 0.5 < row["rigid_best"]
    # 1R1H has four stereocentres; every one sits at a cut, as the fragment module predicts
    assert row["stereocentres"] == row["split_centres"] == 4
    assert len(frag_rows) == row["fragments"] and sum(r["size"] for r in frag_rows) == row["atoms"]
    assert row["sym_fragments"] == sum(r["sym_order"] > 1 for r in frag_rows) >= 2  # carboxylate, phenyl
    assert row["exposed_fragments"] == sum(r["contacts"] == 0 for r in frag_rows)


def test_cli_end_to_end(tmp_path, capsys):
    if not DUMMY.exists():
        pytest.skip(f"example data not found at {DUMMY}")
    out = tmp_path / "geom.csv"
    sweep.main(["--root", str(DUMMY), "--out", str(out), "--limit", "3", "--workers", "1",
                "--num_confs", "2", "--random_samples", "3"])  # fmt: skip
    mols, frags = analyze.load([out])
    assert len(mols) == 3 and len(frags) == sum(m["fragments"] for m in mols)
    analyze.main([str(out), "--top", "2"])
    report = capsys.readouterr().out
    for section in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## Joint count"):
        assert section in report
