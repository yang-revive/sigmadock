import json

import torch
from rdkit import Chem

from conftest import DUMMY
from pocketdit import data, dock, train

TINY = ["--model.dim", "32", "--model.depth", "1", "--model.heads", "2", "--model.pocket_depth", "1",
        "--model.pair_dim", "8", "--model.head_hidden", "8", "--model.num_rbf", "8"]  # fmt: skip


def test_preprocess_train_sample(tmp_path):
    cache = tmp_path / "cache"
    data.main(["--root", str(DUMMY), "--out", str(cache), "--workers", "1"])
    assert len(list(cache.glob("*.pt"))) == 10

    run = tmp_path / "run"
    train.main(["--train_dir", str(cache), "--val_dir", str(cache), "--out_dir", str(run), "--max_steps", "4",
                "--batch_size", "4", "--num_workers", "0", "--log_every", "2", "--val_every", "4",
                "--val_num_sampled", "4", "--flow.num_steps", "3", "--crop.max_atoms", "128", *TINY])  # fmt: skip
    assert (run / "last.pt").exists()
    logged = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert sum("val/rmsd_median" in entry for entry in logged) == 1
    assert all("centroid_err" in entry for entry in logged if "coord_mse" in entry)
    assert any("val/centroid_err_median" in entry for entry in logged)

    # The budget travels in the checkpoint, so docking reproduces the pocket that was trained on.
    ckpt = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert ckpt["config"]["crop"]["max_atoms"] == 128

    folder = DUMMY / "1G9V_RQ3"
    dock.main(["--ckpt", str(run / "last.pt"), "--protein", str(folder / "1G9V_RQ3_protein.pdb"),
                 "--ligand", str(folder / "query_1V4S_MRK.sdf"), "--ref", str(folder / "1G9V_RQ3_ligand.sdf"),
                 "--num_samples", "3", "--out_dir", str(tmp_path / "docked")])  # fmt: skip
    poses = [m for m in Chem.SDMolSupplier(str(tmp_path / "docked" / "1G9V_RQ3__query_1V4S_MRK.sdf")) if m]
    assert len(poses) == 3


def test_dock_from_an_explicit_centre(tmp_path):
    """--center gives a point, so the shell has to reach out by a ligand radius to find a pocket."""
    cache, run = tmp_path / "cache", tmp_path / "run"
    data.main(["--root", str(DUMMY), "--out", str(cache), "--workers", "1"])
    train.main(["--train_dir", str(cache), "--out_dir", str(run), "--max_steps", "2", "--batch_size", "2",
                "--num_workers", "0", "--log_every", "2", "--flow.num_steps", "2", *TINY])  # fmt: skip

    folder = DUMMY / "1V4S_MRK"
    ligand = next(folder.glob("*_ligand.sdf"))
    centre = Chem.SDMolSupplier(str(ligand))[0].GetConformer().GetPositions().mean(axis=0)
    model, flow, crop = dock.load_model(run / "last.pt", torch.device("cpu"))
    poses = dock.dock(
        model, flow, crop,
        folder / "1V4S_MRK_protein.pdb", ligand, None,
        center=[float(c) for c in centre], num_samples=2, num_steps=2,
        pocket_cutoff=8.0, center_margin=6.0, device=torch.device("cpu"),
    )  # fmt: skip
    assert len(poses) == 2
    # A pose generated from a bare centre still lands inside the pocket, not out in solvent.
    pos = poses[0].GetConformer().GetPositions()
    assert ((pos - centre) ** 2).sum(axis=1).max() ** 0.5 < 25.0
