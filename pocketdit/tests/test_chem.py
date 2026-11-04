import torch

from conftest import DUMMY
from pocketdit.chem import read_pdb
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
        assert torch.allclose(it["ref_center"], it["lig_pos"].mean(0))


def test_chirality_feature_sign(items):
    it = items[0]
    chiral = it["chiral_sign"] != 0
    assert chiral.sum() == 4
    args = (it["chiral_nbrs"][None], it["chiral_sign"][None])
    true = chirality_feature(it["lig_pos"][None], *args)[0]
    mirror = chirality_feature(-it["lig_pos"][None], *args)[0]
    assert (true[chiral] > 0.1).all() and (mirror[chiral] < -0.1).all()
    assert (true[~chiral] == 0).all()
