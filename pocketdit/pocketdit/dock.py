"""Dock ligands into a pocket with a trained checkpoint.

Single complex (re-docking: the pocket is defined by the ligand's own coordinates):
    python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand l.sdf --out_dir out/

Cross-docking / real use: define the pocket with a reference ligand or an explicit centre:
    ... --ref ref_ligand.sdf          or          ... --center 12.3 4.5 -7.8

Many complexes: ``--csv`` with columns protein,ligand[,ref] (paths relative to the CSV).
Each complex writes ``<out_dir>/<name>.sdf`` with ``--num_samples`` poses (unranked).

The pocket budget and the prior come from the checkpoint, so docking reproduces the pocket the
model was trained on without having to repeat any flags. A reference ligand is the better input of
the two: ``--center`` gives a point, and the pocket then has to be a ball around it (see
``--center_margin``) rather than a shell wrapped around a plausible ligand.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

from pocketdit.chem import featurize_complex, load_ligand
from pocketdit.data import PocketCrop, collate, complex_name, crop_pocket, index_from_csv, to_device
from pocketdit.flow import FlowConfig, sample
from pocketdit.model import ModelConfig, PocketDiT


def load_model(
    ckpt: str | Path, device: torch.device, use_ema: bool = True
) -> tuple[PocketDiT, FlowConfig, PocketCrop]:
    """Model plus the flow and pocket-crop settings it was trained with, so docking matches."""
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = PocketDiT(ModelConfig(**state["config"]["model"])).to(device).eval()
    model.load_state_dict(state["ema" if use_ema else "model"])
    crop = PocketCrop(**state["config"].get("crop", {}))
    return model, FlowConfig(**state["config"]["flow"]), crop


def with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, (x, y, z) in enumerate(coords.astype(float)):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return out


def dock(
    model: PocketDiT,
    flow: FlowConfig,
    crop: PocketCrop,
    protein: Path,
    ligand: Path,
    ref: Path | None,
    center: list[float] | None,
    num_samples: int,
    num_steps: int | None,
    pocket_cutoff: float,
    center_margin: float,
    device: torch.device,
) -> list[Chem.Mol]:
    mol = load_ligand(ligand)
    explicit_center = ref is None and center is not None
    if ref is not None:
        ref_pos = load_ligand(ref).GetConformer().GetPositions()
    elif center is not None:
        ref_pos = np.asarray(center, dtype=np.float32)[None]
    else:
        ref_pos = mol.GetConformer().GetPositions()
    # With an explicit centre, residue distances are measured from one point instead of from a
    # ligand's atoms, so the whole distance scale shifts out by about a ligand radius. Widen the
    # shell by the same margin, or the pocket comes back as a bare ball around the centre.
    margin = center_margin if explicit_center else 0.0
    item = featurize_complex(protein, mol, ref_pos, pocket_cutoff=pocket_cutoff + margin)
    item = crop_pocket(item, replace(crop, cutoff=crop.cutoff + margin, core_cutoff=crop.core_cutoff + margin))
    batch = to_device(collate([item] * num_samples), device)
    poses = sample(model, batch, flow, num_steps).cpu().numpy()
    out = []
    for k, pos in enumerate(poses):
        pose = with_coords(mol, pos)
        pose.SetProp("_Name", f"{complex_name((protein, ligand, ref))}_sample{k}")
        # Only meaningful for re-docking, where the input ligand is the crystal pose.
        pose.SetDoubleProp("rmsd_to_input", rdMolAlign.CalcRMS(pose, mol))
        out.append(pose)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--protein")
    p.add_argument("--ligand")
    p.add_argument("--ref", help="reference ligand defining the pocket")
    p.add_argument("--center", type=float, nargs=3, help="explicit pocket centre (Å), used as the prior centre")
    p.add_argument("--csv", help="CSV with columns protein,ligand[,ref]")
    p.add_argument("--out_dir", default="docked")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=None, help="default: value stored in the checkpoint")
    p.add_argument("--pocket_cutoff", type=float, default=15.0, help="widest shell considered before cropping")
    p.add_argument("--max_pocket_atoms", type=int, default=None, help="default: the trained value")
    p.add_argument(
        "--center_margin",
        type=float,
        default=6.0,
        help="with --center, how far past the shell to reach (roughly a ligand radius)",
    )
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, flow, crop = load_model(args.ckpt, device, use_ema=not args.no_ema)
    if args.max_pocket_atoms is not None:
        crop = replace(crop, max_atoms=args.max_pocket_atoms)
    if args.csv:
        rows = index_from_csv(args.csv)
    elif args.protein and args.ligand:
        rows = [(Path(args.protein), Path(args.ligand), Path(args.ref) if args.ref else None)]
    else:
        p.error("give --csv, or --protein and --ligand")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        poses = dock(
            model,
            flow,
            crop,
            *row,
            center=args.center,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            pocket_cutoff=args.pocket_cutoff,
            center_margin=args.center_margin,
            device=device,
        )
        with Chem.SDWriter(str(out_dir / f"{complex_name(row)}.sdf")) as w:
            for pose in poses:
                w.write(pose)
        rmsds = [pose.GetDoubleProp("rmsd_to_input") for pose in poses]
        name = complex_name(row)
        print(f"{name}: {len(poses)} poses, rmsd_to_input min {min(rmsds):.2f} median {np.median(rmsds):.2f}")


if __name__ == "__main__":
    main()
