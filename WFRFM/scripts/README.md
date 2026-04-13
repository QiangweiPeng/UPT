# Scripts

Reusable shell entrypoints for common experiment workflows.

Current layout:

- `scripts/norman/train_norman_xpca50.sh`
  Train the Norman 50D unscaled-PCA experiment.
- `scripts/norman/infer_norman_test.sh`
  Run Norman test-set inference from an existing run directory.
- `scripts/norman/benchmark_non_w.sh`
  Run the non-W benchmark for a Norman prediction file.
- `scripts/norman/run_infer_and_benchmark.sh`
  Convenience wrapper that runs inference first, then the non-W benchmark.

These scripts resolve the repository root relative to their own location, so
they can be run from anywhere inside the repo. If you want to copy them to the
current directory for ad-hoc use, they will still work as long as the repo
layout stays the same around them.
