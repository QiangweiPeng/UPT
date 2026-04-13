from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROLLUP_COLORS = {
    "real": "#2f2f2f",
    "full": "#1f77b4",
    "transport-only": "#ff7f0e",
    "growth-only": "#2ca02c",
}


def _composition_long_df(global_validation_df):
    comp_cols = [column for column in global_validation_df.columns if column.startswith("comp__")]
    if not comp_cols:
        raise RuntimeError("No composition columns found in global_validation_df")
    long_df = global_validation_df.melt(
        id_vars=[
            "target_condition",
            "target_timepoint",
            "rollout_mode",
        ],
        value_vars=comp_cols,
        var_name="component",
        value_name="fraction",
    )
    long_df["terminal_label"] = long_df["component"].str.replace("comp__", "", regex=False)
    return long_df


def plot_condition_mechanism_summary(
    global_validation_df: pd.DataFrame,
    cohort_scores_df: pd.DataFrame,
    output_path,
    condition: str,
):
    condition = str(condition)
    global_cond = global_validation_df[
        global_validation_df["target_condition"].astype(str) == condition
    ].copy()
    cohort_cond = cohort_scores_df[
        cohort_scores_df["target_condition"].astype(str) == condition
    ].copy()
    if global_cond.empty or cohort_cond.empty:
        raise RuntimeError(f"No mechanism analysis rows found for condition {condition}")

    final_timepoint = float(global_cond["target_timepoint"].astype(float).max())
    global_final = global_cond[
        global_cond["target_timepoint"].astype(float) == final_timepoint
    ].copy()
    cohort_final = cohort_cond[
        cohort_cond["target_timepoint"].astype(float) == final_timepoint
    ].copy()

    comp_long = _composition_long_df(global_final)
    label_order = sorted(comp_long["terminal_label"].astype(str).unique().tolist())
    rollout_order = ["real", "full", "transport-only", "growth-only"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    for rollout_mode in rollout_order:
        subset = global_cond[global_cond["rollout_mode"].astype(str) == rollout_mode]
        if subset.empty:
            continue
        y_col = "real_count_ratio" if rollout_mode == "real" else "pred_mass_ratio"
        ax.plot(
            subset["target_timepoint"].astype(float),
            subset[y_col].astype(float),
            marker="o",
            linewidth=2.0,
            color=ROLLUP_COLORS.get(rollout_mode, "#777777"),
            label=rollout_mode,
        )
    ax.set_title("Aggregate Mass Ratio vs Real")
    ax.set_xlabel("target_timepoint")
    ax.set_ylabel("ratio to source")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    width = 0.18
    x = np.arange(len(label_order), dtype=np.float64)
    for idx, rollout_mode in enumerate(rollout_order):
        subset = comp_long[comp_long["rollout_mode"].astype(str) == rollout_mode]
        if subset.empty:
            continue
        heights = []
        for label in label_order:
            value = subset.loc[
                subset["terminal_label"].astype(str) == label,
                "fraction",
            ]
            heights.append(float(value.iloc[0]) if not value.empty else 0.0)
        offset = (idx - (len(rollout_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            label=rollout_mode,
            color=ROLLUP_COLORS.get(rollout_mode, "#777777"),
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Terminal major_group Composition @ t={final_timepoint:g}")
    ax.set_ylabel("fraction")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    sns.scatterplot(
        data=cohort_final,
        x="drift_support",
        y="growth_support",
        hue="source_cohort",
        size="n_source",
        sizes=(60, 320),
        palette="tab10",
        ax=ax,
    )
    ax.axvline(0.6, linestyle="--", linewidth=1.0, color="#888888")
    ax.axhline(0.6, linestyle="--", linewidth=1.0, color="#888888")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Cohort Mechanism Scores @ t={final_timepoint:g}")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")

    ax = axes[1, 1]
    retention_cols = [
        "retention__full",
        "retention__transport-only",
        "retention__growth-only",
    ]
    heatmap_df = cohort_final[
        ["source_cohort", *retention_cols]
    ].copy()
    heatmap_df = heatmap_df.set_index("source_cohort")
    heatmap_df.columns = ["full", "transport-only", "growth-only"]
    sns.heatmap(
        heatmap_df.astype(float),
        cmap="mako",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "retention"},
        ax=ax,
    )
    ax.set_title(f"Cohort Retention @ t={final_timepoint:g}")
    ax.set_xlabel("rollout_mode")
    ax.set_ylabel("source_cohort")

    fig.suptitle(condition)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_overview(
    global_validation_df: pd.DataFrame,
    cohort_scores_df: pd.DataFrame,
    output_path,
):
    if global_validation_df.empty or cohort_scores_df.empty:
        raise RuntimeError("global_validation_df and cohort_scores_df must both be non-empty")

    agg_pred = global_validation_df[
        global_validation_df["rollout_mode"].astype(str).isin(["full", "transport-only", "growth-only"])
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    sns.scatterplot(
        data=cohort_scores_df,
        x="drift_support",
        y="growth_support",
        hue="target_condition",
        style="source_cohort",
        alpha=0.85,
        s=90,
        ax=ax,
    )
    ax.axvline(0.6, linestyle="--", linewidth=1.0, color="#888888")
    ax.axhline(0.6, linestyle="--", linewidth=1.0, color="#888888")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Cohort Drift vs Growth Support")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

    ax = axes[1]
    sns.boxplot(
        data=agg_pred,
        x="rollout_mode",
        y="mass_abs_error",
        order=["full", "transport-only", "growth-only"],
        palette=ROLLUP_COLORS,
        ax=ax,
    )
    ax.set_title("Aggregate Mass Error vs Real")
    ax.set_xlabel("rollout_mode")
    ax.set_ylabel("abs(pred_ratio - real_ratio)")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[2]
    sns.boxplot(
        data=agg_pred,
        x="rollout_mode",
        y="composition_tvd",
        order=["full", "transport-only", "growth-only"],
        palette=ROLLUP_COLORS,
        ax=ax,
    )
    ax.set_title("Aggregate Composition TVD vs Real")
    ax.set_xlabel("rollout_mode")
    ax.set_ylabel("TVD")
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_heatmap_grid(
    cohort_scores_df: pd.DataFrame,
    metric_cols: list[str],
    output_path,
    value_format: str = ".3f",
):
    if cohort_scores_df.empty:
        raise RuntimeError("cohort_scores_df must be non-empty")
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")

    n_panels = len(metric_cols)
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, metric_col in enumerate(metric_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        pivot = cohort_scores_df.pivot_table(
            index="source_cohort",
            columns="target_condition",
            values=metric_col,
            aggfunc="mean",
        )
        sns.heatmap(
            pivot,
            cmap="rocket_r",
            annot=True,
            fmt=value_format,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(metric_col)
        ax.set_xlabel("target_condition")
        ax.set_ylabel("source_cohort")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)

    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
