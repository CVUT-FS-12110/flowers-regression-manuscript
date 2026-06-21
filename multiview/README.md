# Multiview Experiments

The code is organized around two experiment entrypoints and one result processor.

## 1. ProposedMethod vs CountNet

```bash
python multiview/run_experiments.py \
  --methods proposed countnet \
  --ground-truth manual visual \
  --seeds 101 202 303 404 505 606 707 808 909 1010
```

Useful options:

- `--fpn` / `--no-fpn` for the proposed method.
- `--fusion sum|concat|attention` for A/B view fusion.
- `--lr-decay annealing|exponential`.
- `--ground-truth manual|visual|both`.
- `--log-every N` prints progress every N epochs. The default is every epoch.

The default `proposed --fpn --fusion sum` model is the original repository's
`nn_fpn.py` model, including its unusual FPN head. `countnet` is the original
`reference_nn.py` model. `--fusion concat` and `--fusion attention` are ablation
variants built on top of the original FPN/standard feature extractors.

## 2. Ablations

```bash
python multiview/run_backbones.py \
  --ground-truth both \
  --fpn yes no \
  --lr-decay annealing exponential \
  --fusion sum concat attention
```

This runs the ProposedMethod grid over FPN on/off, LR schedule, and A/B fusion strategy.

## Tables

```bash
python multiview/summarize_results.py --results-dir multiview/results
```

Outputs are written to `multiview/results/tables/`:

- `all_predictions.tsv`
- `metrics_per_seed.tsv`
- `metrics_summary.tsv`
- `metrics_summary.md`

The summary table reports MAE, RMSE, RE (%), and correlation. The script accepts both
the new `predictions_seed_*.tsv` files and the older `results_*.tsv` layout.
