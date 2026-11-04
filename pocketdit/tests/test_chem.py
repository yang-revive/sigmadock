import numpy as np
import torch

from conftest import DUMMY, load_uncropped
from pocketdit.chem import pocket_center, read_pdb
from pocketdit.model import chirality_feature


def test_read_pdb_keeps_heavy_protein_atoms_only():
    pdb = DUMMY / "1V4S_MRK" / "1V4S_MRK_protein.pdb"
    atoms = read_pdb(pdb)
    n_atom_records = sum(line.startswith("ATOM  ") for line in pdb.read_text().splitlines())
    assert len(atoms["pos"]) == n_atom_records  # the file has no hydrogens; the Na HETATM is dropped
    assert set(atoms["element"]) <= {"C", "N", "O", "S"}


def test_ligand_graph(items):
    for it in items:
        n = len(it["lig_pos"])
        bond, spd = it["lig_bond"], it["lig_spd"]
        assert bond.shape == spd.shape == (n, n)
        assert torch.equal(bond, bond.T) and torch.equal(spd, spd.T)
        assert (spd.diagonal() == 0).all()
        assert ((bond > 0) == (spd == 1)).all()  # bonded <=> one hop apart


def test_pocket_surrounds_ligand(items):
    for it in items:
        d = torch.cdist(it["lig_pos"], it["poc_pos"])
        assert d.min() > 2.0  # no clashing atoms (e.g. the ligand itself) were kept
        assert (d.min(dim=1).values < 8.0).all()  # every ligand atom has pocket context


def test_pocket_residues_are_ordered_by_distance(items_uncropped):
    """crop_pocket relies on this: nearest residue first, atoms grouped by residue."""
    for it in items_uncropped:
        res_id, dmin = it["poc_res_id"], it["poc_res_dmin"]
        assert (res_id.diff() >= 0).all()  # atoms grouped by residue
        assert res_id.unique().tolist() == list(range(len(dmin)))  # every residue has atoms
        assert (dmin.diff() >= 0).all()  # residues sorted by distance
        # Explicit subtraction, not cdist: its matmul expansion loses ~1e-3 Å in float32.
        d = (it["lig_pos"][:, None] - it["poc_pos"][None]).norm(dim=-1).min(dim=0).values
        for r in range(len(dmin)):
            assert torch.isclose(d[res_id == r].min(), dmin[r], atol=1e-4)


def test_pocket_center_is_not_the_ligand_centroid(items_uncropped):
    """The prior centre must come from the protein, not from the answer. See chem.pocket_center."""
    for it in items_uncropped:
        offset = (it["pocket_com"] - it["lig_pos"].mean(0)).norm()
        assert 0.5 < offset < 6.0, offset  # a real translation to solve, but still the same pocket


def test_pocket_center_matches_the_shell_it_is_defined_from():
    protein = read_pdb(DUMMY / "1V4S_MRK" / "1V4S_MRK_protein.pdb")
    it = load_uncropped("1V4S_MRK")
    lig = it["lig_pos"].numpy()
    d = np.linalg.norm(protein["pos"][:, None] - lig[None], axis=-1).min(axis=1)
    expected = protein["pos"][d < 6.0].mean(axis=0)
    assert np.allclose(pocket_center(protein, lig, com_cutoff=6.0), expected, atol=1e-4)
    assert np.allclose(it["pocket_com"].numpy(), expected, atol=1e-4)


def test_explicit_centre_is_taken_at_face_value():
    protein = read_pdb(DUMMY / "1V4S_MRK" / "1V4S_MRK_protein.pdb")
    point = np.array([[12.3, 4.5, -7.8]], dtype=np.float32)
    assert np.allclose(pocket_center(protein, point), point[0])


def test_chirality_feature_sign(items):
    it = items[0]
    chiral = it["chiral_sign"] != 0
    assert chiral.sum() == 4
    args = (it["chiral_nbrs"][None], it["chiral_sign"][None])
    true = chirality_feature(it["lig_pos"][None], *args)[0]
    mirror = chirality_feature(-it["lig_pos"][None], *args)[0]
    assert (true[chiral] > 0.1).all() and (mirror[chiral] < -0.1).all()
    assert (true[~chiral] == 0).all()
