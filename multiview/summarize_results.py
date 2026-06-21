import argparse
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd


TRUTH_COLUMNS = {
    "manual": "Ground Truth",
    "visual": "Visual Estimation",
    "true": "Ground Truth",
    "false": "Visual Estimation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multiview prediction TSVs into MAE, RMSE, RE, and correlation tables."
    )
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--truth-table", default=str(Path(__file__).resolve().parents[1] / "paper" / "df_dump.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pattern", default="**/*predictions*.tsv")
    parser.add_argument("--legacy-pattern", default="**/results_*.tsv")
    return parser.parse_args()


def read_truth_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "orig_name" not in df.columns:
        raise ValueError(f"Missing orig_name column in {path}")
    return df.set_index("orig_name")


def read_prediction_file(path: Path, truth_df: pd.DataFrame) -> pd.DataFrame:
    first_line = path.read_text().splitlines()[0] if path.stat().st_size else ""
    if first_line.startswith("name\t"):
        df = pd.read_csv(path, sep="\t")
        df["run_name"] = df.get("run_name", path.parent.name)
        df["source_file"] = str(path)
        if "ground_truth" not in df.columns:
            df["ground_truth"] = infer_ground_truth(path)
        return df

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["name", "prediction", "error", "best_epoch"],
    )
    ground_truth = infer_ground_truth(path)
    truth_col = TRUTH_COLUMNS[ground_truth]
    df = df.join(truth_df[[truth_col]], on="name")
    df = df.rename(columns={truth_col: "truth"})
    df["absolute_error"] = df["error"].abs()
    df["seed"] = infer_seed(path)
    df["fold"] = None
    df["method"] = path.parent.name
    df["ground_truth"] = ground_truth
    df["run_name"] = path.parent.name
    df["source_file"] = str(path)
    return df


def infer_ground_truth(path: Path) -> str:
    text = path.stem.lower()
    if re.search(r"(_true|_manual)$", text):
        return "manual"
    if re.search(r"(_false|_visual)$", text):
        return "visual"
    for part in path.parts:
        part = part.lower()
        if part.endswith("_true") or "manual" in part:
            return "manual"
        if part.endswith("_false") or "visual" in part:
            return "visual"
    return "manual"


def infer_seed(path: Path) -> Optional[int]:
    match = re.search(r"(?:seed_|results_)(\d+)", path.stem)
    return int(match.group(1)) if match else None


def collect_predictions(results_dir: Path, truth_df: pd.DataFrame, pattern: str, legacy_pattern: str) -> pd.DataFrame:
    paths = sorted(results_dir.glob(pattern))
    paths.extend(path for path in sorted(results_dir.glob(legacy_pattern)) if path not in paths)
    frames = [read_prediction_file(path, truth_df) for path in paths if path.stat().st_size]
    if not frames:
        raise FileNotFoundError(f"No prediction TSV files found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def metric_row(group_keys, group: pd.DataFrame) -> dict[str, object]:
    errors = pd.to_numeric(group["error"], errors="coerce")
    predictions = pd.to_numeric(group["prediction"], errors="coerce")
    truths = pd.to_numeric(group["truth"], errors="coerce")
    mae = errors.abs().mean()
    rmse = math.sqrt((errors ** 2).mean())
    re = (mae / truths.mean()) * 100 if truths.mean() else float("nan")
    corr = predictions.corr(truths)

    if not isinstance(group_keys, tuple):
        group_keys = (group_keys,)

    return {
        "run_name": group_keys[0],
        "ground_truth": group_keys[1],
        "seed": group_keys[2] if len(group_keys) > 2 else "all",
        "n": int(group["name"].nunique()),
        "mae": mae,
        "rmse": rmse,
        "re_percent": re,
        "correlation": corr,
    }


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = pd.DataFrame([
        metric_row(keys, group)
        for keys, group in predictions.groupby(["run_name", "ground_truth", "seed"], dropna=False)
    ])

    rows = []
    for keys, group in per_seed.groupby(["run_name", "ground_truth"], dropna=False):
        run_name, ground_truth = keys
        rows.append({
            "run_name": run_name,
            "ground_truth": ground_truth,
            "seeds": int(group["seed"].nunique()),
            "mae": group["mae"].mean(),
            "mae_std": group["mae"].std(ddof=1),
            "rmse": group["rmse"].mean(),
            "rmse_std": group["rmse"].std(ddof=1),
            "re_percent": group["re_percent"].mean(),
            "re_percent_std": group["re_percent"].std(ddof=1),
            "correlation": group["correlation"].mean(),
            "correlation_std": group["correlation"].std(ddof=1),
        })
    summary = pd.DataFrame(rows).sort_values(["ground_truth", "mae", "run_name"])
    return per_seed, summary


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.4g}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator] + rows)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_df = read_truth_table(args.truth_table)
    predictions = collect_predictions(results_dir, truth_df, args.pattern, args.legacy_pattern)
    per_seed, summary = summarize(predictions)

    predictions.to_csv(output_dir / "all_predictions.tsv", sep="\t", index=False)
    per_seed.to_csv(output_dir / "metrics_per_seed.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "metrics_summary.tsv", sep="\t", index=False)
    markdown = to_markdown(summary)
    (output_dir / "metrics_summary.md").write_text(markdown + "\n")

    print(markdown)
    print(f"\nWrote tables to {output_dir}")


if __name__ == "__main__":
    main()
