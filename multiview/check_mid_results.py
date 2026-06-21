import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_SEEDS = (1, 12, 23, 34, 45, 56, 67, 78, 89, 90)
DEFAULT_FOLDS = 10
DEFAULT_EPOCHS = 300


@dataclass
class RunProgress:
    experiment: str
    run_name: str
    ground_truth: str
    expected_seeds: int
    expected_rows: int
    timing_rows: int
    completed_predictions: int
    latest_seed: str
    latest_fold: int
    latest_epoch: int
    latest_val_mae: Optional[float]
    latest_lr: Optional[float]
    train_hours: Optional[float]
    eta_hours: Optional[float]
    total_hours: Optional[float]
    wall_hours: Optional[float]
    percent: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live progress for multiview method comparison and ablation experiments."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment", choices=["all", "method_comparison", "ablations"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()


def read_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_last_timing(path: Path) -> tuple[int, int, int, Optional[float], Optional[float], int, float]:
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0, 0, None, None, 0, 0.0

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0, 0, 0, None, None, 0, 0.0

    seconds_sum = 0.0
    for line in lines[1:]:
        parts = line.split("\t")
        try:
            seconds_sum += float(parts[3])
        except (IndexError, ValueError):
            pass

    last = lines[-1].split("\t")
    try:
        seed = int(last[0])
        fold = int(last[1])
        epoch = int(last[2])
        val_mae = float(last[5])
        lr = float(last[6])
    except (IndexError, ValueError):
        return 0, 0, 0, None, None, len(lines) - 1, seconds_sum
    return seed, fold, epoch, val_mae, lr, len(lines) - 1, seconds_sum


def summarize_run(experiment: str, run_dir: Path, seeds: list[int], folds: int, epochs: int) -> RunProgress:
    config = read_config(run_dir)
    ground_truth = config.get("ground_truth", infer_ground_truth(run_dir.name))
    expected_seeds = len(seeds)
    expected_rows = expected_seeds * folds * epochs

    timing_files = sorted(run_dir.glob("timing_seed_*.tsv"))
    prediction_files = sorted(run_dir.glob("predictions_seed_*.tsv"))

    timing_rows = 0
    seconds_sum = 0.0
    latest = (0, 0, 0, None, None)
    latest_seed = "-"
    first_mtime = None
    latest_mtime = None
    for path in timing_files:
        seed, fold, epoch, val_mae, lr, rows, file_seconds = read_last_timing(path)
        timing_rows += rows
        seconds_sum += file_seconds
        mtime = path.stat().st_mtime
        first_mtime = mtime if first_mtime is None else min(first_mtime, mtime)
        latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
        if rows and (seed, fold, epoch) >= (latest[0], latest[1], latest[2]):
            latest = (seed, fold, epoch, val_mae, lr)
            latest_seed = str(seed)

    percent = min(100.0, 100.0 * timing_rows / expected_rows) if expected_rows else 0.0
    mean_seconds = seconds_sum / timing_rows if timing_rows else None
    remaining_rows = max(0, expected_rows - timing_rows)
    train_hours = seconds_sum / 3600 if timing_rows else None
    eta_hours = (remaining_rows * mean_seconds / 3600) if mean_seconds is not None else None
    total_hours = (expected_rows * mean_seconds / 3600) if mean_seconds is not None else None
    wall_hours = ((time.time() - first_mtime) / 3600) if first_mtime is not None else None
    completed_predictions = len(prediction_files)
    if completed_predictions >= expected_seeds:
        status = "done"
    elif timing_rows > 0:
        status = "running/partial"
    else:
        status = "waiting"

    return RunProgress(
        experiment=experiment,
        run_name=run_dir.name,
        ground_truth=ground_truth,
        expected_seeds=expected_seeds,
        expected_rows=expected_rows,
        timing_rows=timing_rows,
        completed_predictions=completed_predictions,
        latest_seed=latest_seed,
        latest_fold=latest[1],
        latest_epoch=latest[2],
        latest_val_mae=latest[3],
        latest_lr=latest[4],
        train_hours=train_hours,
        eta_hours=eta_hours,
        total_hours=total_hours,
        wall_hours=wall_hours,
        percent=percent,
        status=status,
    )


def infer_ground_truth(run_name: str) -> str:
    if run_name.endswith("_manual"):
        return "manual"
    if run_name.endswith("_visual"):
        return "visual"
    return "?"


def expected_runs(experiment: str) -> set[str]:
    if experiment == "method_comparison":
        return {
            "proposed_fpn_sum_exponential_manual",
            "proposed_fpn_sum_exponential_visual",
            "countnet_exponential_manual",
            "countnet_exponential_visual",
        }

    if experiment == "ablations":
        runs = set()
        for truth in ("manual", "visual"):
            for fpn in ("fpn", "no_fpn"):
                for decay in ("annealing", "exponential"):
                    for fusion in ("sum", "concat", "attention"):
                        runs.add(f"proposed_{fpn}_{decay}_{fusion}_{truth}")
        return runs

    return set()


def collect_progress(results_dir: Path, experiment: str, seeds: list[int], folds: int, epochs: int) -> list[RunProgress]:
    experiments = ["method_comparison", "ablations"] if experiment == "all" else [experiment]
    rows: list[RunProgress] = []

    for exp in experiments:
        exp_dir = results_dir / exp
        found = {path.name: path for path in exp_dir.iterdir() if path.is_dir()} if exp_dir.exists() else {}
        for run_name in sorted(expected_runs(exp) | set(found)):
            if run_name in found:
                rows.append(summarize_run(exp, found[run_name], seeds, folds, epochs))
            else:
                rows.append(RunProgress(
                    experiment=exp,
                    run_name=run_name,
                    ground_truth=infer_ground_truth(run_name),
                    expected_seeds=len(seeds),
                    expected_rows=len(seeds) * folds * epochs,
                    timing_rows=0,
                    completed_predictions=0,
                    latest_seed="-",
                    latest_fold=0,
                    latest_epoch=0,
                    latest_val_mae=None,
                    latest_lr=None,
                    train_hours=None,
                    eta_hours=None,
                    total_hours=None,
                    wall_hours=None,
                    percent=0.0,
                    status="missing",
                ))
    return rows


def print_table(rows: list[RunProgress]) -> None:
    headers = [
        "experiment", "run", "truth", "status", "progress", "pred",
        "latest", "val_mae", "train_h", "eta_h",
    ]
    table = []
    for row in rows:
        table.append([
            row.experiment,
            row.run_name,
            row.ground_truth,
            row.status,
            f"{row.percent:5.1f}%",
            f"{row.completed_predictions}/{row.expected_seeds}",
            f"s{row.latest_seed} f{row.latest_fold} e{row.latest_epoch}",
            "-" if row.latest_val_mae is None else f"{row.latest_val_mae:.3f}",
            format_hours(row.train_hours),
            format_hours(row.eta_hours),
        ])

    widths = [
        max(len(str(item)) for item in [header] + [line[idx] for line in table])
        for idx, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for line in table:
        print("  ".join(str(item).ljust(widths[idx]) for idx, item in enumerate(line)))


def format_hours(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value < 0.01:
        return f"{value:.4f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def print_summary(rows: list[RunProgress]) -> None:
    print()
    for experiment in ("method_comparison", "ablations"):
        subset = [row for row in rows if row.experiment == experiment]
        if not subset:
            continue
        done = sum(row.status == "done" for row in subset)
        active = sum(row.status == "running/partial" for row in subset)
        total_percent = sum(row.timing_rows for row in subset) / sum(row.expected_rows for row in subset) * 100
        train_hours = sum(row.train_hours or 0.0 for row in subset)
        eta_hours = sum(row.eta_hours or 0.0 for row in subset if row.status != "done")
        total_hours = train_hours + eta_hours
        print(
            f"{experiment}: done {done}/{len(subset)} | active {active} | progress {total_percent:.1f}% "
            f"| elapsed {train_hours:.2f} h | remaining {eta_hours:.2f} h | total {total_hours:.2f} h"
        )
    print_overall_finish_estimate(rows)


def print_overall_finish_estimate(rows: list[RunProgress]) -> None:
    comparison_eta = estimate_method_comparison_hours(rows)
    ablation_eta = estimate_ablation_parallel_hours(rows)
    estimates = [value for value in (comparison_eta, ablation_eta) if value is not None]
    if not estimates:
        return
    final_eta = max(estimates)
    parts = []
    if comparison_eta is not None:
        parts.append(f"comparison {comparison_eta:.2f} h")
    if ablation_eta is not None:
        parts.append(f"ablations {ablation_eta:.2f} h")
    print(f"overall estimated finish: {final_eta:.2f} h from now ({' | '.join(parts)})")


def estimate_method_comparison_hours(rows: list[RunProgress]) -> Optional[float]:
    subset = [row for row in rows if row.experiment == "method_comparison"]
    active_or_done = [row for row in subset if row.train_hours is not None]
    if not active_or_done:
        return None
    mean_total = sum((row.train_hours or 0.0) + (row.eta_hours or 0.0) for row in active_or_done) / len(active_or_done)
    remaining = 0.0
    for row in subset:
        if row.status == "done":
            continue
        if row.eta_hours is not None:
            remaining += row.eta_hours
        else:
            remaining += mean_total
    return remaining


def estimate_ablation_parallel_hours(rows: list[RunProgress]) -> Optional[float]:
    subset = [row for row in rows if row.experiment == "ablations"]
    active_or_done = [row for row in subset if row.train_hours is not None]
    if not active_or_done:
        return None
    mean_total = sum((row.train_hours or 0.0) + (row.eta_hours or 0.0) for row in active_or_done) / len(active_or_done)
    branches: dict[tuple[str, str], list[RunProgress]] = {}
    for row in subset:
        branch = ablation_branch(row.run_name)
        branches.setdefault(branch, []).append(row)

    branch_remaining = []
    for branch_rows in branches.values():
        remaining = 0.0
        for row in branch_rows:
            if row.status == "done":
                continue
            if row.eta_hours is not None:
                remaining += row.eta_hours
            else:
                remaining += mean_total
        branch_remaining.append(remaining)
    return max(branch_remaining) if branch_remaining else None


def ablation_branch(run_name: str) -> tuple[str, str]:
    truth = infer_ground_truth(run_name)
    if "_no_fpn_" in run_name:
        fpn = "no_fpn"
    elif "_fpn_" in run_name:
        fpn = "fpn"
    else:
        fpn = "?"
    return truth, fpn


def main() -> None:
    args = parse_args()
    rows = collect_progress(Path(args.results_dir), args.experiment, args.seeds, args.folds, args.epochs)
    print_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
