import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODE_ORDER = ["real", "full", "transport-only", "growth-only"]
PRED_MODES = ["full", "transport-only", "growth-only"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot direct observable maps from cached zebrafish mode-UMAP files: "
            "cell occupancy enrichment/depletion and mass enrichment/depletion against control."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode-umap-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=85)
    parser.add_argument("--eps", type=float, default=1.0)
    return parser.parse_args()


def _robust_limits(arrays, q=0.98):
    values = np.concatenate([np.ravel(arr) for arr in arrays], axis=0)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    bound = float(np.quantile(np.abs(values), q))
    bound = max(bound, 0.25)
    return -bound, bound


def _compute_hist(df, x_edges, y_edges, weight_col=None):
    weights = None if weight_col is None else df[weight_col].astype(float).to_numpy()
    hist, _, _ = np.histogram2d(
        df["UMAP1"].astype(float).to_numpy(),
        df["UMAP2"].astype(float).to_numpy(),
        bins=[x_edges, y_edges],
        weights=weights,
    )
    return hist.T


def _observable_maps(adata, bins, eps):
    obs = adata.obs.copy()
    coords = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    obs["UMAP1"] = coords[:, 0]
    obs["UMAP2"] = coords[:, 1]

    x_min, x_max = float(obs["UMAP1"].min()), float(obs["UMAP1"].max())
    y_min, y_max = float(obs["UMAP2"].min()), float(obs["UMAP2"].max())
    x_pad = 0.03 * (x_max - x_min + 1e-6)
    y_pad = 0.03 * (y_max - y_min + 1e-6)
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, bins + 1)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, bins + 1)

    control_df = obs[obs["datatype"].astype(str) == "control"].copy()
    if control_df.empty:
        raise RuntimeError("No control rows found in mode_compare_umap.h5ad")
    control_count_hist = _compute_hist(control_df, x_edges, y_edges, weight_col=None)

    count_maps = {}
    mass_maps = {}
    summary_rows = []
    for mode in MODE_ORDER:
        mode_df = obs[obs["datatype"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        count_hist = _compute_hist(mode_df, x_edges, y_edges, weight_col=None)
        count_map = np.log2((count_hist + eps) / (control_count_hist + eps))
        count_maps[mode] = count_map

        if mode == "real":
            mass_hist = count_hist
        else:
            weight_col = "display_mass" if "display_mass" in mode_df.columns else "mass"
            mass_hist = _compute_hist(mode_df, x_edges, y_edges, weight_col=weight_col)
        mass_map = np.log2((mass_hist + eps) / (control_count_hist + eps))
        mass_maps[mode] = mass_map

        summary_rows.append(
            {
                "mode": mode,
                "count_sum": float(count_hist.sum()),
                "mass_sum": float(mass_hist.sum()),
                "count_enrichment_mean": float(count_map.mean()),
                "mass_enrichment_mean": float(mass_map.mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if "real" in count_maps:
        real_count = count_maps["real"]
        real_mass = mass_maps["real"]
        for mode in PRED_MODES:
            if mode not in count_maps:
                continue
            count_corr = float(np.corrcoef(real_count.ravel(), count_maps[mode].ravel())[0, 1])
            mass_corr = float(np.corrcoef(real_mass.ravel(), mass_maps[mode].ravel())[0, 1])
            summary_df.loc[summary_df["mode"] == mode, "real_count_map_corr"] = count_corr
            summary_df.loc[summary_df["mode"] == mode, "real_mass_map_corr"] = mass_corr

    return count_maps, mass_maps, summary_df


def _plot_maps(adata, count_maps, mass_maps, output_path):
    metadata = adata.uns.get("mode_umap_metadata", {})
    condition = metadata.get("condition", "unknown")
    timepoint = float(metadata.get("timepoint", np.nan))

    count_vmin, count_vmax = _robust_limits(list(count_maps.values()))
    mass_vmin, mass_vmax = _robust_limits(list(mass_maps.values()))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, mode in enumerate(MODE_ORDER):
        count_ax = axes[0, col]
        mass_ax = axes[1, col]
        count_map = count_maps[mode]
        mass_map = mass_maps[mode]

        im_count = count_ax.imshow(
            count_map,
            origin="lower",
            cmap="coolwarm",
            vmin=count_vmin,
            vmax=count_vmax,
            aspect="auto",
        )
        count_ax.set_title(f"{mode}: count vs control")
        count_ax.set_xticks([])
        count_ax.set_yticks([])

        im_mass = mass_ax.imshow(
            mass_map,
            origin="lower",
            cmap="coolwarm",
            vmin=mass_vmin,
            vmax=mass_vmax,
            aspect="auto",
        )
        mass_ax.set_title(f"{mode}: mass vs control")
        mass_ax.set_xticks([])
        mass_ax.set_yticks([])

    fig.colorbar(im_count, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02, label="log2 enrichment/depletion")
    fig.colorbar(im_mass, ax=axes[1, :].tolist(), fraction=0.025, pad=0.02, label="log2 enrichment/depletion")
    fig.suptitle(
        f"{condition} @ {timepoint:g} hpf: direct observable maps\n"
        "top = local cell occupancy vs control, bottom = local calibrated mass vs control"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    for path in args.mode_umap_paths:
        adata = ad.read_h5ad(path)
        count_maps, mass_maps, summary_df = _observable_maps(
            adata=adata,
            bins=args.bins,
            eps=args.eps,
        )
        output_dir = path.parent
        _plot_maps(
            adata=adata,
            count_maps=count_maps,
            mass_maps=mass_maps,
            output_path=output_dir / "mode_compare_observable_maps.png",
        )
        summary_df.to_csv(output_dir / "mode_compare_observable_maps_summary.csv", index=False)


if __name__ == "__main__":
    main()
