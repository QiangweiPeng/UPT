import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (  # noqa: E402
    build_mass_accuracy_tables,
    plot_mass_accuracy_condition_grid,
    plot_mass_accuracy_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot zebrafish mass-accuracy summaries from saved evaluation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing mass_summary.csv and real_pair_counts_full.csv.",
    )
    parser.add_argument(
        "--mass-summary-path",
        type=Path,
        default=None,
        help="Optional explicit path to mass_summary.csv.",
    )
    parser.add_argument(
        "--real-counts-path",
        type=Path,
        default=None,
        help="Optional explicit path to real pair counts. Defaults to <eval-dir>/real_pair_counts_full.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <eval-dir>/mass_accuracy.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Zebrafish Mass Accuracy",
        help="Figure title prefix.",
    )
    return parser.parse_args()


def resolve_input_path(explicit_path: Path | None, default_path: Path, label: str):
    path = explicit_path.resolve() if explicit_path is not None else default_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def main():
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    mass_summary_path = resolve_input_path(
        args.mass_summary_path,
        eval_dir / "mass_summary.csv",
        "mass summary",
    )
    real_counts_path = resolve_input_path(
        args.real_counts_path,
        eval_dir / "real_pair_counts_full.csv",
        "real pair counts",
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (eval_dir / "mass_accuracy")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_df, transition_df, metrics = build_mass_accuracy_tables(
        mass_summary=mass_summary_path,
        real_counts=real_counts_path,
    )

    pair_df.to_csv(output_dir / "mass_accuracy_pairs.csv", index=False)
    transition_df.to_csv(output_dir / "mass_accuracy_transitions.csv", index=False)
    with open(output_dir / "mass_accuracy_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    plot_mass_accuracy_summary(
        pair_df=pair_df,
        transition_df=transition_df,
        metrics=metrics,
        output_path=output_dir / "mass_accuracy_summary.png",
        title=args.title,
    )
    plot_mass_accuracy_condition_grid(
        pair_df=pair_df,
        metrics=metrics,
        output_path=output_dir / "mass_accuracy_condition_grid.png",
    )

    index_df = pd.DataFrame(
        [
            {
                "eval_dir": str(eval_dir),
                "mass_summary_path": str(mass_summary_path),
                "real_counts_path": str(real_counts_path),
                "pairs_csv": str(output_dir / "mass_accuracy_pairs.csv"),
                "transitions_csv": str(output_dir / "mass_accuracy_transitions.csv"),
                "metrics_json": str(output_dir / "mass_accuracy_metrics.json"),
                "summary_png": str(output_dir / "mass_accuracy_summary.png"),
                "condition_grid_png": str(output_dir / "mass_accuracy_condition_grid.png"),
            }
        ]
    )
    index_df.to_csv(output_dir / "mass_accuracy_index.csv", index=False)
    print(f"Saved mass-accuracy outputs to {output_dir}")


if __name__ == "__main__":
    main()
