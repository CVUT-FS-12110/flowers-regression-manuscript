import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Optional

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import CustomDataset, set_random_seed


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (1, 12, 23, 34, 45, 56, 67, 78, 89, 90)
GROUND_TRUTHS = {
    "manual": True,
    "ground_truth": True,
    "true": True,
    "visual": False,
    "visual_estimation": False,
    "false": False,
}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    run_name: str
    method: str
    ground_truth: str
    seed: int
    folds: int
    epochs: int
    batch_size: int
    test_batch_size: int
    lr: Optional[float]
    lr_decay: str
    backbone: str
    fpn: bool
    fusion: str
    pretrained: bool
    device: str
    data_glob: str
    results_dir: str
    checkpoints_dir: str
    num_workers: int
    log_every: int


def add_common_args(parser: argparse.ArgumentParser, include_lr_decay: bool = True) -> None:
    parser.add_argument("--ground-truth", "--gt", nargs="+", default=["manual", "visual"],
                        help="Target(s): manual, visual, true, false, or both.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate. Defaults match the original model setup.")
    if include_lr_decay:
        parser.add_argument("--lr-decay", choices=["annealing", "exponential"], default="exponential")
    parser.add_argument("--backbone", default="resnet50",
                        help="Kept for compatibility; original proposed/countnet models ignore this.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-glob", default=str(ROOT / "final_data" / "*.jpg"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print progress every N epochs. Use 0 to disable epoch logs.")
    parser.add_argument("--no-pretrained", action="store_true")


def expand_ground_truths(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        key = value.lower()
        if key == "both":
            expanded.extend(["manual", "visual"])
            continue
        if key not in GROUND_TRUTHS:
            raise ValueError(f"Unknown ground truth '{value}'")
        expanded.append("manual" if GROUND_TRUTHS[key] else "visual")
    return list(dict.fromkeys(expanded))


def slug(*parts: object) -> str:
    text = "_".join(str(part) for part in parts if part is not None and str(part) != "")
    return text.replace("/", "-").replace(" ", "_")


def discover_observations(data_glob: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    filenames = glob(data_glob)
    observations = sorted({
        tuple(Path(name).stem.split("_")[:3])
        for name in filenames
    })
    trees = sorted({observation[1:3] for observation in observations})
    if not observations:
        raise FileNotFoundError(f"No images found for data glob: {data_glob}")
    return observations, trees


def kfold_indices(n_items: int, n_splits: int):
    if n_splits < 2:
        raise ValueError("--folds must be at least 2")
    if n_splits > n_items:
        raise ValueError(f"--folds ({n_splits}) cannot exceed number of trees ({n_items})")

    indices = np.arange(n_items)
    fold_sizes = np.full(n_splits, n_items // n_splits, dtype=int)
    fold_sizes[: n_items % n_splits] += 1
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_ids = indices[start:stop]
        train_ids = np.concatenate([indices[:start], indices[stop:]])
        yield train_ids, test_ids
        start = stop


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.method == "countnet":
        from models.reference_nn import ReferenceMultiViewNetwork

        return ReferenceMultiViewNetwork()

    if config.method == "proposed":
        if config.fusion != "sum":
            from models.original_fusion import OriginalFeatureFusionNetwork

            return OriginalFeatureFusionNetwork(use_fpn=config.fpn, fusion=config.fusion)

        if config.fpn:
            from models.nn_fpn import MultiViewNetwork
        else:
            from models.nn import MultiViewNetwork

        return MultiViewNetwork()

    raise ValueError(f"Unsupported method '{config.method}'")


def build_optimizer(model: nn.Module, config: ExperimentConfig):
    if config.method == "countnet":
        lr = config.lr if config.lr is not None else 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        lr = config.lr if config.lr is not None else 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    if config.lr_decay == "annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30,
            eta_min=1e-6,
        )
    else:
        gamma = 0.972351 if config.method == "countnet" else 0.95
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return optimizer, scheduler


def train_one_epoch(model, criterion, device, optimizer, train_loader) -> float:
    model.train()
    losses = []
    for images_a, images_b, targets, _ in train_loader:
        images_a = images_a.to(device)
        images_b = images_b.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images_a, images_b)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model, device, test_loader) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    rows = []
    absolute_errors = []

    with torch.no_grad():
        for images_a, images_b, targets, names in test_loader:
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            targets = targets.to(device)
            outputs = model(images_a, images_b).reshape(-1)
            truths = targets.reshape(-1)

            for name, truth, prediction in zip(names, truths.cpu().numpy(), outputs.cpu().numpy()):
                error = float(truth - prediction)
                absolute_errors.append(abs(error))
                rows.append({
                    "name": str(name),
                    "truth": float(truth),
                    "prediction": float(prediction),
                    "error": error,
                    "absolute_error": abs(error),
                })

    return float(np.mean(absolute_errors)) if absolute_errors else float("inf"), rows


def run_seed(config: ExperimentConfig) -> Path:
    set_random_seed(config.seed)

    use_cuda = config.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": config.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": config.test_batch_size, "shuffle": False}
    if use_cuda:
        loader_kwargs = {"num_workers": config.num_workers, "pin_memory": True}
        train_kwargs.update(loader_kwargs)
        test_kwargs.update(loader_kwargs)

    observations, trees = discover_observations(config.data_glob)
    trees = list(trees)
    np.random.shuffle(trees)
    ground_truth_flag = GROUND_TRUTHS[config.ground_truth]

    run_dir = Path(config.results_dir) / config.experiment / config.run_name
    checkpoint_dir = Path(config.checkpoints_dir) / config.experiment / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    timing_path = run_dir / f"timing_seed_{config.seed}_{config.ground_truth}.tsv"
    timing_path.write_text("")

    print(
        "Starting "
        f"experiment={config.experiment} run={config.run_name} method={config.method} "
        f"truth={config.ground_truth} seed={config.seed} device={device}",
        flush=True,
    )

    all_rows = []
    convergence = np.zeros((config.epochs, config.folds), dtype=float)
    timing_rows = []

    for fold_idx, (train_ids, test_ids) in enumerate(kfold_indices(len(trees), config.folds), start=1):
        print(f"Fold {fold_idx}/{config.folds} | train={len(train_ids)} trees test={len(test_ids)} trees", flush=True)
        train_trees = [trees[i] for i in train_ids]
        test_trees = [trees[i] for i in test_ids]
        train_samples = [obs for obs in observations if obs[1:3] in train_trees]
        test_samples = [obs for obs in observations if obs[1:3] in test_trees]

        train_dataset = CustomDataset(train_samples, config.data_glob, ground_truth=ground_truth_flag)
        test_dataset = CustomDataset(test_samples, config.data_glob, augment=False, ground_truth=ground_truth_flag)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        model = build_model(config).to(device)
        criterion = nn.MSELoss() if config.method == "countnet" else nn.SmoothL1Loss()
        optimizer, scheduler = build_optimizer(model, config)

        best_error = float("inf")
        best_rows: list[dict[str, object]] = []
        best_epoch = 0

        for epoch in range(1, config.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, criterion, device, optimizer, train_loader)
            scheduler.step()
            elapsed = time.time() - t0

            fold_error, fold_rows = evaluate(model, device, test_loader)
            convergence[epoch - 1, fold_idx - 1] = fold_error
            timing_row = {
                "seed": config.seed,
                "fold": fold_idx,
                "epoch": epoch,
                "seconds": elapsed,
                "train_loss": train_loss,
                "validation_mae": fold_error,
                "lr": scheduler.get_last_lr()[0],
            }
            timing_rows.append(timing_row)
            append_tsv_row(timing_path, timing_row)

            if fold_error < best_error:
                best_error = fold_error
                best_epoch = epoch
                best_rows = fold_rows
                torch.save(model.state_dict(), checkpoint_dir / f"seed_{config.seed}_fold_{fold_idx}.pt")
                best_marker = " *"
            else:
                best_marker = ""

            if config.log_every and (epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs):
                print(
                    f"{config.run_name} | seed {config.seed} | fold {fold_idx}/{config.folds} "
                    f"| epoch {epoch}/{config.epochs} | train_loss {train_loss:.4f} "
                    f"| val_mae {fold_error:.4f} | best {best_error:.4f}@{best_epoch} "
                    f"| lr {scheduler.get_last_lr()[0]:.3g} | {elapsed:.1f}s{best_marker}",
                    flush=True,
                )

        for row in best_rows:
            row.update({
                "seed": config.seed,
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "method": config.method,
                "ground_truth": config.ground_truth,
                "run_name": config.run_name,
            })
        all_rows.extend(best_rows)

    predictions_path = run_dir / f"predictions_seed_{config.seed}_{config.ground_truth}.tsv"
    convergence_path = run_dir / f"convergence_seed_{config.seed}_{config.ground_truth}.tsv"
    write_tsv(predictions_path, all_rows)
    np.savetxt(
        convergence_path,
        convergence,
        delimiter="\t",
        header="\t".join(f"fold_{idx}" for idx in range(1, config.folds + 1)),
        comments="",
    )
    write_tsv(timing_path, timing_rows)
    return predictions_path


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def append_tsv_row(path: Path, row: dict[str, object]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
