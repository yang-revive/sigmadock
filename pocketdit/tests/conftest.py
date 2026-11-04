from pathlib import Path

import pytest
import torch

from pocketdit.chem import featurize_complex, load_ligand
from pocketdit.model import ModelConfig, PocketDiT

DUMMY = Path(__file__).resolve().parents[2] / "notebooks" / "dummy_data"


def load_item(name: str, pocket_cutoff: float = 8.0) -> dict:
    folder = DUMMY / name
    mol = load_ligand(next(folder.glob("*_ligand.sdf")))
    item = featurize_complex(next(folder.glob("*_protein.pdb")), mol, mol.GetConformer().GetPositions(), pocket_cutoff)
    item["name"] = name
    return item


@pytest.fixture(scope="session")
def items() -> list[dict]:
    if not DUMMY.exists():
        pytest.skip(f"example data not found at {DUMMY}")
    # 1R1H_BIR has 4 stereocentres; 1U1C_BAU has a smaller ligand and pocket.
    return [load_item("1R1H_BIR"), load_item("1U1C_BAU")]


def tiny_config() -> ModelConfig:
    return ModelConfig(dim=64, depth=2, heads=4, pocket_depth=1, pair_dim=16, head_hidden=16, num_rbf=16)


@pytest.fixture
def random_model() -> PocketDiT:
    """Tiny model with all weights randomised, so zero-initialised layers do not hide bugs."""
    torch.manual_seed(0)
    model = PocketDiT(tiny_config()).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model
