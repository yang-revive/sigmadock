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
                "--val_num_sampled", "4", "--flow.num_steps", "3", *TINY])  # fmt: skip
    assert (run / "last.pt").exists() and (run / "log.jsonl").read_text().count("val/rmsd_median") == 1

    folder = DUMMY / "1G9V_RQ3"
    dock.main(["--ckpt", str(run / "last.pt"), "--protein", str(folder / "1G9V_RQ3_protein.pdb"),
                 "--ligand", str(folder / "query_1V4S_MRK.sdf"), "--ref", str(folder / "1G9V_RQ3_ligand.sdf"),
                 "--num_samples", "3", "--out_dir", str(tmp_path / "docked")])  # fmt: skip
    poses = [m for m in Chem.SDMolSupplier(str(tmp_path / "docked" / "1G9V_RQ3__query_1V4S_MRK.sdf")) if m]
    assert len(poses) == 3
