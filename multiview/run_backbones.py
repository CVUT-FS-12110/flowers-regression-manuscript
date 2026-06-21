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
        description="Run ProposedMethod ablations over FPN, LR decay, and A/B fusion."
    )
    add_common_args(parser, include_lr_decay=False)
    parser.add_argument("--fpn", nargs="+", default=["yes", "no"],
                        choices=["yes", "no"])
    parser.add_argument("--fusion", nargs="+", default=["sum", "concat", "attention"],
                        choices=["sum", "concat", "attention"])
    parser.add_argument("--lr-decay", nargs="+", default=["annealing", "exponential"],
                        choices=["annealing", "exponential"])
    parser.add_argument("--experiment-name", default="ablations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truths = expand_ground_truths(args.ground_truth)
    fpn_values = [value == "yes" for value in args.fpn]

    for ground_truth in ground_truths:
        for use_fpn in fpn_values:
            for lr_decay in args.lr_decay:
                for fusion in args.fusion:
                    for seed in args.seeds:
                        run_name = slug(
                            "proposed",
                            "fpn" if use_fpn else "no_fpn",
                            lr_decay,
                            fusion,
                            ground_truth,
                        )
                        config = ExperimentConfig(
                            experiment=args.experiment_name,
                            run_name=run_name,
                            method="proposed",
                            ground_truth=ground_truth,
                            seed=seed,
                            folds=args.folds,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            test_batch_size=args.test_batch_size,
                            lr=args.lr,
                            lr_decay=lr_decay,
                            backbone=args.backbone,
                            fpn=use_fpn,
                            fusion=fusion,
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
