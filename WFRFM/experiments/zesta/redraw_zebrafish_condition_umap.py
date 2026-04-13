import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.evals_plot import plot_saved_condition_umap_with_real_categories  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redraw saved zebrafish condition UMAPs with reduced overplotting and real-data category coloring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition-umap-dir",
        type=Path,
        required=True,
        help="Directory containing condition_umap_index.csv and per-condition *_umap.h5ad files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save the redrawn figures. Defaults to <condition-umap-dir>/redraw_<real-color-by>/.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition list. Supports repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--real-color-by",
        type=str,
        default="major_group",
        help="Primary real-data obs column used for coloring.",
    )
    parser.add_argument(
        "--secondary-real-color-by",
        type=str,
        default="tissue",
        help="Secondary real-data obs column used for coloring.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=12,
        help="Maximum number of displayed real categories. Remaining categories are collapsed into Other.",
    )
    parser.add_argument(
        "--pred-alpha",
        type=float,
        default=0.08,
        help="Point alpha for sampled predictions.",
    )
    parser.add_argument(
        "--real-alpha",
        type=float,
        default=0.85,
        help="Point alpha for real cells.",
    )
    parser.add_argument(
        "--pred-size",
        type=float,
        default=2.0,
        help="Point size for sampled predictions.",
    )
    parser.add_argument(
        "--real-size",
        type=float,
        default=5.0,
        help="Point size for real cells.",
    )
    return parser.parse_args()


def parse_condition_args(raw_conditions):
    if not raw_conditions:
        return None
    conditions = []
    for item in raw_conditions:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        conditions.extend(parts)
    return conditions or None


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def main():
    args = parse_args()
    condition_umap_dir = args.condition_umap_dir.resolve()
    index_path = condition_umap_dir / "condition_umap_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing condition UMAP index: {index_path}")

    index_df = pd.read_csv(index_path)
    selected_conditions = parse_condition_args(args.conditions)
    if selected_conditions is not None:
        index_df = index_df[index_df["condition"].astype(str).isin(selected_conditions)].copy()
    if index_df.empty:
        raise RuntimeError("No matching conditions found in condition_umap_index.csv")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            condition_umap_dir
            / (
                f"redraw_{sanitize_name(args.real_color_by)}_"
                f"{sanitize_name(args.secondary_real_color_by)}"
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for row in index_df.itertuples(index=False):
        condition = str(row.condition)
        umap_h5ad = Path(row.umap_h5ad)
        condition_name = sanitize_name(condition)
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        adata_vis = ad.read_h5ad(umap_h5ad)
        category_summary = plot_saved_condition_umap_with_real_categories(
            adata_vis=adata_vis,
            output_path=condition_dir / f"{condition_name}_redraw.png",
            real_color_key=args.real_color_by,
            secondary_real_color_key=args.secondary_real_color_by,
            max_categories=args.max_categories,
            pred_alpha=args.pred_alpha,
            real_alpha=args.real_alpha,
            pred_size=args.pred_size,
            real_size=args.real_size,
        )
        category_summary.to_csv(
            condition_dir
            / (
                f"{condition_name}_{sanitize_name(args.real_color_by)}_"
                f"{sanitize_name(args.secondary_real_color_by)}_summary.csv"
            ),
            index=False,
        )
        index_rows.append(
            {
                "condition": condition,
                "umap_h5ad": str(umap_h5ad),
                "redraw_png": str(condition_dir / f"{condition_name}_redraw.png"),
                "category_summary_csv": str(
                    condition_dir
                    / (
                        f"{condition_name}_{sanitize_name(args.real_color_by)}_"
                        f"{sanitize_name(args.secondary_real_color_by)}_summary.csv"
                    )
                ),
                "real_color_by": args.real_color_by,
                "secondary_real_color_by": args.secondary_real_color_by,
            }
        )
        print(f"Saved redrawn UMAP for {condition} to {condition_dir}")

    pd.DataFrame(index_rows).to_csv(output_dir / "redraw_index.csv", index=False)
    print(f"Saved redraw index to {output_dir / 'redraw_index.csv'}")


if __name__ == "__main__":
    main()
