import argparse

from experiment_lib import (
    ExperimentConfig,
    add_common_args,
    expand_ground_truths,
    run_seed,
    slug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run method comparison experiments: ProposedMethod vs CountNet."
    )
    add_common_args(parser)
    parser.add_argument("--methods", nargs="+", default=["proposed", "countnet"],
                        choices=["proposed", "countnet"])
    parser.add_argument("--fpn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion", default="sum", choices=["sum", "concat", "attention"])
    parser.add_argument("--experiment-name", default="method_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truths = expand_ground_truths(args.ground_truth)

    for ground_truth in ground_truths:
        for method in args.methods:
            for seed in args.seeds:
                run_name = slug(
                    method,
                    "fpn" if args.fpn and method == "proposed" else "no_fpn" if method == "proposed" else None,
                    args.fusion if method == "proposed" else None,
                    args.lr_decay,
                    ground_truth,
                )
                config = ExperimentConfig(
                    experiment=args.experiment_name,
                    run_name=run_name,
                    method=method,
                    ground_truth=ground_truth,
                    seed=seed,
                    folds=args.folds,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size,
                    lr=args.lr,
                    lr_decay=args.lr_decay,
                    backbone=args.backbone,
                    fpn=args.fpn,
                    fusion=args.fusion,
                    pretrained=not args.no_pretrained,
                    device=args.device,
                    data_glob=args.data_glob,
                    results_dir=args.results_dir,
                    checkpoints_dir=args.checkpoints_dir,
                    num_workers=args.num_workers,
                    log_every=args.log_every,
                )
                output = run_seed(config)
                print(f"Wrote {output}")


if __name__ == "__main__":
    main()
