"""Training loop (plain PyTorch).

    python -m pocketdit.train --train_dir cache/train --val_dir cache/val --out_dir runs/exp1 \
        --model.dim 256 --model.depth 6 --flow.prior_std 4.0

Every field of ``TrainConfig`` (including the nested ``model.*`` / ``flow.*``) is a CLI flag.
Logs go to ``<out_dir>/log.jsonl``; checkpoints to ``<out_dir>/last.pt`` and ``step_*.pt``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pocketdit.data import ComplexDataset, PocketCrop, collate, to_device
from pocketdit.flow import FlowConfig, centroid_error, flow_matching_loss, pose_rmsd, rmsd, sample, select_targets
from pocketdit.model import ModelConfig, PocketDiT, assemble


@dataclass
class TrainConfig:
    train_dir: str = "cache/train"
    val_dir: str = ""
    out_dir: str = "runs/pocketdit"
    resume: str = ""  # path to a checkpoint
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_steps: int = 200_000
    min_lr_ratio: float = 0.1  # cosine decay to lr * min_lr_ratio
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    bf16: bool = True  # autocast to bfloat16 on CUDA
    log_every: int = 50
    val_every: int = 2000
    val_num_sampled: int = 64  # complexes sampled for RMSD at each validation
    save_every: int = 5000
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    crop: PocketCrop = field(default_factory=PocketCrop)


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def add(obj: object, prefix: str) -> None:
        for f in fields(obj):
            value = getattr(obj, f.name)
            if is_dataclass(value):
                add(value, f"{prefix}{f.name}.")
            elif isinstance(value, bool):
                parser.add_argument(f"--{prefix}{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
            else:
                parser.add_argument(f"--{prefix}{f.name}", type=type(value))

    add(cfg, "")
    for key, value in vars(parser.parse_args(argv)).items():
        if value is None:
            continue
        *path, name = key.split(".")
        target = cfg
        for p in path:
            target = getattr(target, p)
        setattr(target, name, value)
    return cfg


def lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters(), strict=True):
        e.lerp_(p, 1 - decay)


@torch.no_grad()
def validate(model: PocketDiT, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    losses, rmsds, centroids, floors = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(cfg, device):
            _, m = flow_matching_loss(model, batch, cfg.flow)
            losses.append(m["loss"])
            if len(rmsds) < cfg.val_num_sampled:
                x = sample(model, batch, cfg.flow)
                if cfg.flow.fragments:
                    # Against the true pose, scored under each fragment's best tied labelling, next
                    # to the floor: what the same metric gives the best pose the templates can take.
                    rmsds.extend(pose_rmsd(x, batch).tolist())
                    eye = torch.eye(3, device=device).expand_as(batch["frag_rot"][:, :, 0])
                    target = assemble(batch, select_targets(batch, eye)[0], batch["frag_trans"])
                    floors.extend(pose_rmsd(target, batch).tolist())
                else:
                    rmsds.extend(rmsd(x, batch["lig_pos"], batch["lig_mask"]).tolist())
                centroids.extend(centroid_error(x, batch["lig_pos"], batch["lig_mask"]).tolist())
    model.train()
    out = {"val/loss": sum(losses) / len(losses)}
    if rmsds:
        r, c = torch.tensor(rmsds), torch.tensor(centroids)
        out.update({"val/rmsd_median": r.median().item(), "val/rmsd<2A": (r < 2.0).float().mean().item()})
        # Compare against the same number on a cross-docked set: see flow.centroid_error.
        out["val/centroid_err_median"] = c.median().item()
    if floors:
        out["val/floor_median"] = torch.tensor(floors).median().item()
    return out


def autocast(cfg: TrainConfig, device: torch.device) -> torch.autocast:
    enabled = cfg.bf16 and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def save(path: Path, model, ema, opt, sched, step: int, cfg: TrainConfig) -> None:
    state = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "config": asdict(cfg),
    }
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_config(argv)
    if cfg.flow.rot_loss not in ("chordal", "geodesic"):
        raise ValueError(f"--flow.rot_loss must be chordal or geodesic, got {cfg.flow.rot_loss!r}")
    torch.manual_seed(cfg.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    loader_kw = {"collate_fn": collate, "num_workers": cfg.num_workers, "persistent_workers": cfg.num_workers > 0}
    train_ds = ComplexDataset(cfg.train_dir, cfg.crop, train=True, fragments=cfg.flow.fragments)
    batch_size = min(cfg.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = None
    if cfg.val_dir:
        # train=False: the deterministic shell, so validation numbers are comparable across steps.
        val_ds = ComplexDataset(cfg.val_dir, cfg.crop, train=False, fragments=cfg.flow.fragments)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, **loader_kw)

    model = PocketDiT(cfg.model).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    step = 0
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, {len(train_ds)} train complexes")

    log = open(out_dir / "log.jsonl", "a")  # noqa: SIM115
    tic = time.time()
    while step < cfg.max_steps:
        for batch in train_loader:
            batch = to_device(batch, device)
            with autocast(cfg, device):
                loss, metrics = flow_matching_loss(model, batch, cfg.flow)
            if not torch.isfinite(loss):
                print(f"[step {step}] non-finite loss on {batch['name']}, skipping batch")
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ema_update(ema, model, cfg.ema_decay)
            step += 1

            if step % cfg.log_every == 0:
                metrics.update(step=step, lr=sched.get_last_lr()[0], grad_norm=grad_norm.item())
                metrics["sec_per_step"] = (time.time() - tic) / cfg.log_every
                tic = time.time()
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
            if val_loader is not None and step % cfg.val_every == 0:
                val = validate(ema, val_loader, cfg, device) | {"step": step}
                log.write(json.dumps(val) + "\n")
                log.flush()
                print(val)
            if step % cfg.save_every == 0:
                save(out_dir / f"step_{step}.pt", model, ema, opt, sched, step, cfg)
            if step >= cfg.max_steps:
                break
    save(out_dir / "last.pt", model, ema, opt, sched, step, cfg)
    log.close()


if __name__ == "__main__":
    main()
