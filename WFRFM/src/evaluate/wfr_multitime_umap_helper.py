
import math
from typing import Dict, Iterable, Optional, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from typing import Optional, Sequence, Iterable
import numpy as np
import torch
import anndata as ad


@torch.no_grad()
def wfr_euler_solve_with_traj(
    model,
    z: torch.Tensor,
    cond: torch.Tensor,
    cov_dict: dict,
    n_steps: int,
    dt: float,
    clamp_g: float = 100.0,
    save_steps: Optional[Sequence[int]] = None,
    mode: str = "all",   # "all" | "v_only" | "g_only"
):
    """
    Euler solver with trajectory recording.

    Parameters
    ----------
    mode : str
        - "all":    update both z and m
        - "v_only": update z only; keep m = 1
        - "g_only": update m only; keep z unchanged

    Returns
    -------
    z_final : torch.Tensor [B, D]
    m_final : torch.Tensor [B, 1]
    traj : dict
        Keys:
            - steps: np.ndarray [T]
            - times: np.ndarray [T]
            - z: list[np.ndarray], each [B, D]
            - m: list[np.ndarray], each [B, 1]
            - g: list[np.ndarray], each [B, 1]
    """
    if mode not in {"all", "v_only", "g_only"}:
        raise ValueError(f"Unknown mode={mode}. Expected one of: all, v_only, g_only")

    B = z.shape[0]
    device = z.device
    m = torch.ones((B, 1), device=device, dtype=z.dtype)

    if save_steps is None:
        save_steps = list(range(n_steps + 1))
    save_steps = sorted(set(int(s) for s in save_steps if 0 <= int(s) <= n_steps))

    traj = {"steps": [], "times": [], "z": [], "m": [], "g": []}

    for k in range(n_steps + 1):
        t_val = k * dt
        t = torch.full((B, 1), t_val, device=device, dtype=z.dtype)

        v, g = model(t, z, cond, cov_dict)
        if g.dim() == 1:
            g = g.unsqueeze(1)
        g = g.clamp(-clamp_g, clamp_g)

        if k in save_steps:
            traj["steps"].append(k)
            traj["times"].append(float(t_val))
            traj["z"].append(z.detach().cpu().numpy().copy())
            traj["m"].append(m.detach().cpu().numpy().copy())
            traj["g"].append(g.detach().cpu().numpy().copy())

        if k == n_steps:
            break

        if mode in {"all", "v_only"}:
            z = z + v * dt

        if mode in {"all", "g_only"}:
            m = m * torch.exp(g * dt)

    traj["steps"] = np.asarray(traj["steps"], dtype=int)
    traj["times"] = np.asarray(traj["times"], dtype=float)
    return z, m, traj


def _times_to_steps(save_times: Iterable[float], t_destiny: float, n_steps: int):
    save_times = np.asarray(list(save_times), dtype=float)
    save_times = np.clip(save_times, 0.0, float(t_destiny))
    save_steps = np.rint(save_times / float(t_destiny) * n_steps).astype(int)
    save_steps = np.clip(save_steps, 0, n_steps)
    return sorted(set(save_steps.tolist()))


@torch.no_grad()
def run_batch_inference_with_traj(
    model,
    adata_source: ad.AnnData,
    rulebook: dict,
    wishlist: list,
    condition_embeddings: dict,
    condition_key_name: str = "perturbation",
    sample_rep: str = "X_pca_scaled",
    n_steps: int = 50,
    device: str = "cuda",
    source_celltype_key: str | None = None,
    t_destiny: float = 1.0,
    save_times: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    modes: Sequence[str] = ("all", "v_only", "g_only"),
):
    """
    Run batch inference and record trajectories for multiple modes.

    Parameters
    ----------
    modes : Sequence[str]
        Any subset of ("all", "v_only", "g_only")

    Returns
    -------
    results : dict
        results[cond_name] = {
            "wish": wish,
            "source_celltype": ...,
            "source_obs_names": ...,
            "all": {
                "z_pred": ...,
                "m_pred": ...,
                "traj": ...
            },
            "v_only": {
                "z_pred": ...,
                "m_pred": ...,
                "traj": ...
            },
            "g_only": {
                "z_pred": ...,
                "m_pred": ...,
                "traj": ...
            }
        }
    """
    valid_modes = {"all", "v_only", "g_only"}
    modes = list(dict.fromkeys(modes))  # deduplicate while preserving order
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode={mode}. Expected subset of {valid_modes}")

    model.eval()
    model.to(device)
    dt = t_destiny / n_steps
    results = {}

    print(f"Running inference for {len(wishlist)} conditions ...")
    print(f"Modes: {modes}")
    save_steps = _times_to_steps(save_times, t_destiny=t_destiny, n_steps=n_steps)

    control_groups_keys = rulebook.get("stratification", {}).get("control_groups", [])

    for wish in wishlist:
        schema = rulebook["condition_tuple_schema"]
        current_tuple = [wish[var] for var in schema]
        cond_name = str(tuple(current_tuple))

        if len(control_groups_keys) == 0:
            adata_source_cur = adata_source
        else:
            mask = np.ones(len(adata_source), dtype=bool)
            for k in control_groups_keys:
                if k in wish:
                    mask = mask & (adata_source.obs[k].astype(str) == str(wish[k]))
            adata_source_cur = adata_source[mask]

        if len(adata_source_cur) == 0:
            print(f"Warning: no matched control cells found for {cond_name}, skipped.")
            continue

        source_celltype = None
        if source_celltype_key is not None and source_celltype_key in adata_source_cur.obs.columns:
            source_celltype = adata_source_cur.obs[source_celltype_key].astype(str).to_numpy()

        source_obs_names = adata_source_cur.obs_names.astype(str).to_numpy()

        z0_np = adata_source_cur.obsm[sample_rep]
        z0 = torch.tensor(z0_np, dtype=torch.float32, device=device)
        B = z0.shape[0]

        base_cond_val = wish[condition_key_name]
        cond_vec = condition_embeddings[base_cond_val]
        if not isinstance(cond_vec, torch.Tensor):
            cond_vec = torch.tensor(cond_vec, dtype=torch.float32, device=device)
        else:
            cond_vec = cond_vec.to(device=device, dtype=torch.float32)
        cond_batch = cond_vec.unsqueeze(0).expand(B, -1)

        cov_dict_batch = {}

        for cov_name, info in rulebook["model_inputs"]["categorical"].items():
            m_source = info["model_source"]
            if m_source in ["control", "base_cell"]:
                obs_col = info["obs_col"]
                val_np = adata_source_cur.obs[obs_col].to_numpy(dtype=np.int64)
                cov_dict_batch[cov_name] = torch.tensor(val_np, device=device)
            else:
                raw_val = str(wish[cov_name])
                idx = rulebook["categorical_mappings"][cov_name].get(raw_val, -1)
                if idx == -1:
                    raise ValueError(
                        f"Categorical mapping missing for {cov_name}={raw_val} in rulebook."
                    )
                cov_dict_batch[cov_name] = torch.full((B,), idx, dtype=torch.long, device=device)

        for cov_name, info in rulebook["model_inputs"]["continuous"].items():
            m_source = info["model_source"]
            if m_source in ["control", "base_cell"]:
                obs_col = info["obs_col"]
                val_np = adata_source_cur.obs[obs_col].to_numpy(dtype=np.float32)
                cov_dict_batch[cov_name] = torch.tensor(val_np, device=device)
            else:
                raw_val = float(wish[cov_name])
                stats = rulebook["continuous_stats"][cov_name]
                mu, std, transform = stats["mean"], stats["std"], stats["transform"]

                if transform == "log1p_zscore":
                    val_tf = np.log1p(raw_val)
                else:
                    val_tf = raw_val

                scaled_val = (val_tf - mu) / std
                cov_dict_batch[cov_name] = torch.full(
                    (B,), scaled_val, dtype=torch.float32, device=device
                )

        cond_result = {
            "wish": wish,
            "source_celltype": source_celltype,
            "source_obs_names": source_obs_names,
        }

        for mode in modes:
            z_pred, m_pred, traj = wfr_euler_solve_with_traj(
                model=model,
                z=z0.clone(),
                cond=cond_batch,
                cov_dict=cov_dict_batch,
                n_steps=n_steps,
                dt=dt,
                save_steps=save_steps,
                mode=mode,
            )

            cond_result[mode] = {
                "z_pred": z_pred.cpu().numpy(),
                "m_pred": m_pred.cpu().numpy(),
                "traj": traj,
            }

        results[cond_name] = cond_result

    return results


def fit_umap_reducer(
    adata_source: ad.AnnData,
    sample_rep: str = "X_pca_scaled",
    n_neighbors: int = 30,
    min_dist: float = 0.35,
    random_state: int = 0,
):
    """
    Fit a UMAP reducer on the latent representation so predicted states can be projected
    consistently across time points.

    Requires `umap-learn`.
    """
    try:
        import umap.umap_ as umap
    except ImportError as e:
        raise ImportError(
            "Please install umap-learn first: pip install umap-learn"
        ) from e

    X = np.asarray(adata_source.obsm[sample_rep], dtype=np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
    )
    reducer.fit(X)
    return reducer


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from typing import Dict, Optional, Sequence


def _robust_minmax(x, low_q=1.0, high_q=99.0):
    x = np.asarray(x).reshape(-1)
    lo = np.nanpercentile(x, low_q)
    hi = np.nanpercentile(x, high_q)
    if not np.isfinite(lo):
        lo = np.nanmin(x)
    if not np.isfinite(hi):
        hi = np.nanmax(x)
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _get_global_color_limits(arrays, symmetric=False, low_q=1.0, high_q=99.0):
    cat = np.concatenate([np.asarray(a).reshape(-1) for a in arrays], axis=0)
    lo, hi = _robust_minmax(cat, low_q=low_q, high_q=high_q)
    if symmetric:
        vmax = max(abs(lo), abs(hi), 1e-6)
        return -vmax, vmax
    return lo, hi


def plot_multitime_umap_for_condition(
    results: Dict,
    cond_name: str,
    adata_source,
    sample_rep: str = "X_pca_scaled",
    save_times: Optional[Sequence[float]] = None,
    reducer=None,
    background: str = "all_source",

    # 当前固定画法：
    # [all mass] [v_only mass] [g_only mass] [g_mode 的 g(t)]
    mass_modes: Sequence[str] = ("all", "v_only", "g_only"),
    g_mode: str = "all",

    mass_cmap: str = "Greys",
    g_cmap: str = "coolwarm",

    point_size: float = 10.0,
    point_alpha: float = 0.90,
    bg_size: float = 5.0,
    bg_alpha: float = 0.10,

    # robust color range
    mass_high_q: float = 99.0,
    g_high_q: float = 99.0,

    # layout / style
    figsize_per_panel: tuple = (3.8, 3.4),
    title_prefix: Optional[str] = None,
    show_ticks: bool = False,
    rasterized: bool = True,
):
    """
    Publication-style multi-time UMAP plot.

    Layout
    ------
    One row per time point, four columns:
        1) all      | Mass m(t)
        2) v_only   | Mass m(t)
        3) g_only   | Mass m(t)
        4) g_mode   | g(t)

    Notes
    -----
    - Requires results[cond_name][mode]["traj"] structure.
    - All mass panels share one global color scale.
    - All g panels share one global symmetric color scale centered at 0.
    """

    if cond_name not in results:
        raise KeyError(f"{cond_name} not found in results.")

    if len(mass_modes) != 3:
        raise ValueError("mass_modes must contain exactly three modes, e.g. ('all', 'v_only', 'g_only').")

    item = results[cond_name]

    required_modes = list(dict.fromkeys(list(mass_modes) + [g_mode]))
    for mode in required_modes:
        if mode not in item:
            raise KeyError(
                f"{cond_name} does not contain mode={mode}. "
                f"Available keys: {list(item.keys())}"
            )
        if "traj" not in item[mode]:
            raise KeyError(f"{cond_name}[{mode}] has no trajectory data.")

    if reducer is None:
        reducer = fit_umap_reducer(adata_source=adata_source, sample_rep=sample_rep)

    # ---- check time alignment across modes ----
    ref_times = np.asarray(item[required_modes[0]]["traj"]["times"], dtype=float)
    for mode in required_modes[1:]:
        cur_times = np.asarray(item[mode]["traj"]["times"], dtype=float)
        if len(cur_times) != len(ref_times) or not np.allclose(cur_times, ref_times):
            raise ValueError(f"Trajectory times are not aligned across modes: {required_modes}")

    # ---- choose time points ----
    if save_times is None:
        use_idx = np.arange(len(ref_times), dtype=int)
    else:
        save_times = np.asarray(save_times, dtype=float)
        use_idx = []
        for t in save_times:
            idx = int(np.argmin(np.abs(ref_times - t)))
            use_idx.append(idx)
        use_idx = np.asarray(sorted(set(use_idx)), dtype=int)

    n_rows = len(use_idx)
    n_cols = 4

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows + 1,
        ncols=n_cols,
        height_ratios=[1.0] * n_rows + [0.08],
        hspace=0.08,
        wspace=0.05,
    )

    axes_grid = np.empty((n_rows, n_cols), dtype=object)

    ref_ax = None
    for r in range(n_rows):
        for c in range(n_cols):
            if ref_ax is None:
                ax = fig.add_subplot(gs[r, c])
                ref_ax = ax
            else:
                ax = fig.add_subplot(gs[r, c], sharex=ref_ax, sharey=ref_ax)
            axes_grid[r, c] = ax

    cax_mass = fig.add_subplot(gs[n_rows, 0:3])
    cax_g = fig.add_subplot(gs[n_rows, 3])

    # ---- background ----
    bg_xy = None
    if background == "all_source":
        bg_xy = reducer.transform(np.asarray(adata_source.obsm[sample_rep], dtype=np.float32))
    elif background != "none":
        raise ValueError("background must be 'all_source' or 'none'.")

    # ---- global mass color scale ----
    all_mass_arrays = []
    for mode in mass_modes:
        traj_mode = item[mode]["traj"]
        for idx in use_idx:
            all_mass_arrays.append(np.asarray(traj_mode["m"][idx]).reshape(-1))

    if len(all_mass_arrays) == 0:
        all_mass_arrays = [np.array([0.0], dtype=np.float32)]

    _, mass_hi = _get_global_color_limits(
        all_mass_arrays,
        symmetric=False,
        low_q=0.0,
        high_q=mass_high_q,
    )
    mass_norm = Normalize(vmin=0.0, vmax=max(float(mass_hi), 1e-8))

    # ---- global g color scale ----
    g_arrays = [np.asarray(item[g_mode]["traj"]["g"][idx]).reshape(-1) for idx in use_idx]
    if len(g_arrays) == 0:
        g_arrays = [np.array([0.0], dtype=np.float32)]

    g_lo, g_hi = _get_global_color_limits(
        g_arrays,
        symmetric=True,
        low_q=100.0 - g_high_q,
        high_q=g_high_q,
    )
    g_abs = max(abs(float(g_lo)), abs(float(g_hi)), 1e-6)
    g_norm = TwoSlopeNorm(vmin=-g_abs, vcenter=0.0, vmax=g_abs)

    # ---- titles ----
    col_titles = [
        "all | Mass",
        "v_only | Mass",
        "g_only | Mass",
        f"{g_mode} | g(t)",
    ]

    # ---- plotting ----
    mass_mappable = None
    g_mappable = None

    all_x = []
    all_y = []

    if bg_xy is not None:
        all_x.append(bg_xy[:, 0])
        all_y.append(bg_xy[:, 1])

    for row_i, idx in enumerate(use_idx):
        t_show = float(ref_times[idx])

        # 3 mass panels
        for col_i, mode in enumerate(mass_modes):
            ax = axes_grid[row_i, col_i]
            traj_mode = item[mode]["traj"]

            z_t = np.asarray(traj_mode["z"][idx], dtype=np.float32)
            m_t = np.asarray(traj_mode["m"][idx]).reshape(-1)
            xy_t = reducer.transform(z_t)

            all_x.append(xy_t[:, 0])
            all_y.append(xy_t[:, 1])

            if bg_xy is not None:
                ax.scatter(
                    bg_xy[:, 0],
                    bg_xy[:, 1],
                    s=bg_size,
                    c="lightgray",
                    alpha=bg_alpha,
                    linewidths=0,
                    rasterized=rasterized,
                )

            sc = ax.scatter(
                xy_t[:, 0],
                xy_t[:, 1],
                s=point_size,
                c=m_t,
                cmap=mass_cmap,
                norm=mass_norm,
                alpha=point_alpha,
                linewidths=0,
                rasterized=rasterized,
            )
            if mass_mappable is None:
                mass_mappable = sc

        # g panel
        axg = axes_grid[row_i, 3]
        traj_g = item[g_mode]["traj"]

        z_g = np.asarray(traj_g["z"][idx], dtype=np.float32)
        g_t = np.asarray(traj_g["g"][idx]).reshape(-1)
        xy_g = reducer.transform(z_g)

        all_x.append(xy_g[:, 0])
        all_y.append(xy_g[:, 1])

        if bg_xy is not None:
            axg.scatter(
                bg_xy[:, 0],
                bg_xy[:, 1],
                s=bg_size,
                c="lightgray",
                alpha=bg_alpha,
                linewidths=0,
                rasterized=rasterized,
            )

        scg = axg.scatter(
            xy_g[:, 0],
            xy_g[:, 1],
            s=point_size,
            c=g_t,
            cmap=g_cmap,
            norm=g_norm,
            alpha=point_alpha,
            linewidths=0,
            rasterized=rasterized,
        )
        if g_mappable is None:
            g_mappable = scg

        # row time label: put on the far left of the first panel
        axes_grid[row_i, 0].text(
            -0.30, 0.5,
            f"t = {t_show:.2f}",
            transform=axes_grid[row_i, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # ---- unified x/y limits ----
    x_cat = np.concatenate(all_x, axis=0)
    y_cat = np.concatenate(all_y, axis=0)

    x_min, x_max = np.nanmin(x_cat), np.nanmax(x_cat)
    y_min, y_max = np.nanmin(y_cat), np.nanmax(y_cat)

    x_pad = 0.03 * max(x_max - x_min, 1e-6)
    y_pad = 0.03 * max(y_max - y_min, 1e-6)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes_grid[r, c]
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect("equal", adjustable="box")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # titles only for top row
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight="bold", pad=8)

            # axis labels
            if r == n_rows - 1:
                ax.set_xlabel("UMAP1", fontsize=10)
            else:
                ax.set_xlabel("")

            if c == 0:
                ax.set_ylabel("UMAP2", fontsize=10)
            else:
                ax.set_ylabel("")

            # ticks
            if show_ticks:
                if r < n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                if c > 0:
                    ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    # ---- colorbars ----
    if mass_mappable is not None:
        cbar_m = fig.colorbar(mass_mappable, cax=cax_mass, orientation="horizontal")
        cbar_m.set_label("Mass m(t)", fontsize=10)
        cbar_m.ax.tick_params(labelsize=9)

    if g_mappable is not None:
        cbar_g = fig.colorbar(g_mappable, cax=cax_g, orientation="horizontal")
        cbar_g.set_label("g_net output g(t)", fontsize=10)
        cbar_g.ax.tick_params(labelsize=9)

    # ---- title ----
    if title_prefix is None:
        fig.suptitle(f"{cond_name}", y=0.98, fontsize=14, fontweight="bold")
    else:
        fig.suptitle(f"{title_prefix} | {cond_name}", y=0.98, fontsize=14, fontweight="bold")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.10)
    return fig, axes_grid


import ast
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from typing import Optional, Sequence, Union
from matplotlib.lines import Line2D


def _match_condition_mask(
    adata: ad.AnnData,
    condition_key: Union[str, Sequence[str]],
    cond_name,
):
    if isinstance(condition_key, str):
        return adata.obs[condition_key].astype(str).to_numpy() == str(cond_name)

    if isinstance(cond_name, str):
        try:
            cond_tuple = ast.literal_eval(cond_name)
        except Exception:
            raise ValueError(
                "When condition_key is a sequence of columns, cond_name should be "
                "a tuple or a tuple-like string."
            )
    else:
        cond_tuple = tuple(cond_name)

    if len(condition_key) != len(cond_tuple):
        raise ValueError(
            f"condition_key has {len(condition_key)} columns, "
            f"but cond_name has {len(cond_tuple)} elements."
        )

    mask = np.ones(adata.n_obs, dtype=bool)
    for k, v in zip(condition_key, cond_tuple):
        col = adata.obs[k].to_numpy()
        if isinstance(v, (int, float, np.integer, np.floating)):
            mask &= np.isclose(np.asarray(col, dtype=float), float(v))
        else:
            mask &= (adata.obs[k].astype(str).to_numpy() == str(v))
    return mask


def _match_time_mask(
    adata: ad.AnnData,
    time_key: Optional[str] = None,
    time_value: Optional[float] = 1.0,
):
    if time_key is None:
        return np.ones(adata.n_obs, dtype=bool)

    col = adata.obs[time_key].to_numpy()
    try:
        col_float = np.asarray(col, dtype=float)
        return np.isclose(col_float, float(time_value))
    except Exception:
        return adata.obs[time_key].astype(str).to_numpy() == str(time_value)


def _build_categorical_palette(labels, palette: Optional[dict] = None):
    labels = np.asarray(labels).astype(str)
    uniq = sorted(np.unique(labels).tolist())

    if palette is not None:
        missing = [u for u in uniq if u not in palette]
        if missing:
            raise ValueError(f"Palette is missing labels: {missing}")
        return palette, uniq

    base_colors = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )

    if len(uniq) <= len(base_colors):
        colors = base_colors[:len(uniq)]
    else:
        cmap = plt.get_cmap("gist_ncar")
        colors = [cmap(i / max(len(uniq) - 1, 1)) for i in range(len(uniq))]

    color_map = {u: colors[i] for i, u in enumerate(uniq)}
    return color_map, uniq


def plot_real_umap_for_condition(
    adata_test: ad.AnnData,
    cond_name,
    condition_key: Union[str, Sequence[str]],
    reducer,
    sample_rep: str = "X_pca_scaled",

    # left panel background
    adata_background: Optional[ad.AnnData] = None,
    background: str = "all_source",

    # true treated filter
    time_key: Optional[str] = None,
    time_value: Optional[float] = 1.0,

    # true treated style
    point_color: str = "crimson",
    point_alpha: float = 0.90,
    point_size: float = 10.0,

    # background style
    bg_alpha: float = 0.10,
    bg_size: float = 5.0,

    # new: optional control panel
    adata_control: Optional[ad.AnnData] = None,
    control_celltype_key: Optional[str] = None,
    control_palette: Optional[dict] = None,
    control_point_alpha: float = 0.90,
    control_point_size: Optional[float] = None,
    show_control_legend: bool = True,
    legend_fontsize: float = 8.5,

    # layout
    figsize_single=(4.2, 4.0),
    figsize_double=(9.0, 4.0),
    title_prefix: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    show_ticks: bool = False,
    rasterized: bool = True,
):
    """
    Publication-style real treated UMAP.

    Behavior
    --------
    - If adata_control is None or control_celltype_key is None:
        single panel: True treated
    - Otherwise:
        two panels:
            left  = True treated
            right = Control colored by control_celltype_key
    """

    if sample_rep not in adata_test.obsm:
        raise KeyError(f"{sample_rep} not found in adata_test.obsm")

    cond_mask = _match_condition_mask(adata_test, condition_key, cond_name)
    time_mask = _match_time_mask(adata_test, time_key=time_key, time_value=time_value)
    mask = cond_mask & time_mask

    if mask.sum() == 0:
        raise ValueError(
            f"No cells found for cond_name={cond_name} "
            f"with time_key={time_key}, time_value={time_value}"
        )

    adata_real = adata_test[mask].copy()
    real_z = np.asarray(adata_real.obsm[sample_rep], dtype=np.float32)
    real_xy = reducer.transform(real_z)

    # background for treated panel
    bg_xy = None
    if background == "all_source":
        if adata_background is None:
            raise ValueError("adata_background must be provided when background='all_source'")
        if sample_rep not in adata_background.obsm:
            raise KeyError(f"{sample_rep} not found in adata_background.obsm")
        bg_xy = reducer.transform(
            np.asarray(adata_background.obsm[sample_rep], dtype=np.float32)
        )
    elif background != "none":
        raise ValueError("background must be 'all_source' or 'none'")

    # optional control panel
    use_control_panel = (
        adata_control is not None and control_celltype_key is not None
    )

    ctrl_xy = None
    ctrl_labels = None
    ctrl_color_map = None
    ctrl_unique = None

    if use_control_panel:
        if sample_rep not in adata_control.obsm:
            raise KeyError(f"{sample_rep} not found in adata_control.obsm")
        if control_celltype_key not in adata_control.obs.columns:
            raise KeyError(f"{control_celltype_key} not found in adata_control.obs")

        ctrl_z = np.asarray(adata_control.obsm[sample_rep], dtype=np.float32)
        ctrl_xy = reducer.transform(ctrl_z)
        ctrl_labels = adata_control.obs[control_celltype_key].astype(str).fillna("NA").to_numpy()
        ctrl_color_map, ctrl_unique = _build_categorical_palette(
            ctrl_labels, palette=control_palette
        )

    # create figure
    if use_control_panel:
        fig, axes = plt.subplots(1, 2, figsize=figsize_double)
        axes = np.atleast_1d(axes)
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize_single)
        axes = np.atleast_1d(axes)

    ax_real = axes[0]

    # left: true treated
    if bg_xy is not None:
        ax_real.scatter(
            bg_xy[:, 0],
            bg_xy[:, 1],
            s=bg_size,
            c="lightgray",
            alpha=bg_alpha,
            linewidths=0,
            rasterized=rasterized,
        )

    ax_real.scatter(
        real_xy[:, 0],
        real_xy[:, 1],
        s=point_size,
        c=point_color,
        alpha=point_alpha,
        linewidths=0,
        rasterized=rasterized,
    )

    if title_prefix is None:
        if time_key is None:
            ax_real.set_title("True treated", fontsize=12, fontweight="bold", pad=8)
        else:
            ax_real.set_title(
                f"True treated | t = {time_value}",
                fontsize=12,
                fontweight="bold",
                pad=8,
            )
    else:
        if time_key is None:
            ax_real.set_title(
                f"{title_prefix} | True treated",
                fontsize=12,
                fontweight="bold",
                pad=8,
            )
        else:
            ax_real.set_title(
                f"{title_prefix} | True treated | t = {time_value}",
                fontsize=12,
                fontweight="bold",
                pad=8,
            )

    # right: control colored by cell type
    if use_control_panel:
        ax_ctrl = axes[1]
        ctrl_point_size = point_size if control_point_size is None else control_point_size

        ctrl_colors = [ctrl_color_map[x] for x in ctrl_labels]
        ax_ctrl.scatter(
            ctrl_xy[:, 0],
            ctrl_xy[:, 1],
            s=ctrl_point_size,
            c=ctrl_colors,
            alpha=control_point_alpha,
            linewidths=0,
            rasterized=rasterized,
        )
        ax_ctrl.set_title(
            f"Control | colored by {control_celltype_key}",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )

        if show_control_legend:
            handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=ctrl_color_map[k],
                    markeredgewidth=0,
                    markersize=5.5,
                    label=k,
                )
                for k in ctrl_unique
            ]
            ax_ctrl.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=legend_fontsize,
                title=control_celltype_key,
                title_fontsize=legend_fontsize,
            )

    # shared limits
    if xlim is None or ylim is None:
        xs = [real_xy[:, 0]]
        ys = [real_xy[:, 1]]

        if bg_xy is not None:
            xs.append(bg_xy[:, 0])
            ys.append(bg_xy[:, 1])

        if use_control_panel:
            xs.append(ctrl_xy[:, 0])
            ys.append(ctrl_xy[:, 1])

        x_cat = np.concatenate(xs, axis=0)
        y_cat = np.concatenate(ys, axis=0)

        x_min, x_max = np.nanmin(x_cat), np.nanmax(x_cat)
        y_min, y_max = np.nanmin(y_cat), np.nanmax(y_cat)

        x_pad = 0.03 * max(x_max - x_min, 1e-6)
        y_pad = 0.03 * max(y_max - y_min, 1e-6)

        if xlim is None:
            xlim = (x_min - x_pad, x_max + x_pad)
        if ylim is None:
            ylim = (y_min - y_pad, y_max + y_pad)

    for i, ax in enumerate(axes):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("UMAP1", fontsize=10)
        if i == 0:
            ax.set_ylabel("UMAP2", fontsize=10)
        else:
            ax.set_ylabel("")

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

    if use_control_panel and show_control_legend:
        fig.subplots_adjust(left=0.08, right=0.82, top=0.88, bottom=0.12, wspace=0.10)
    else:
        fig.tight_layout()

    return fig, axes




# for figure: colormap-based mass display

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from typing import Dict, Optional, Sequence
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# obs / result helpers
# ---------------------------------------------------------------------------
def _value_equal(a, b):
    try:
        return np.isclose(float(a), float(b))
    except Exception:
        return str(a) == str(b)


def _build_obs_mask(adata: ad.AnnData, filters: dict):
    mask = np.ones(adata.n_obs, dtype=bool)

    for k, v in filters.items():
        if k not in adata.obs.columns:
            continue

        col = adata.obs[k].to_numpy()
        cur = np.array([_value_equal(x, v) for x in col], dtype=bool)
        mask &= cur

    return mask


def _find_result_key_by_wish(results: Dict, wish: dict):
    for cond_name, item in results.items():
        item_wish = item.get("wish", None)
        if item_wish is None:
            continue

        ok = True
        for k, v in wish.items():
            if k not in item_wish:
                ok = False
                break
            if not _value_equal(item_wish[k], v):
                ok = False
                break

        if ok:
            return cond_name

    raise KeyError(f"No matching condition found in results for wish={wish}")


def _nearest_traj_index(traj: dict, inference_time: float):
    if "times" not in traj:
        raise KeyError("traj does not contain 'times'.")

    times = np.asarray(traj["times"], dtype=float)

    if len(times) == 0:
        raise ValueError("traj['times'] is empty.")

    return int(np.argmin(np.abs(times - float(inference_time))))


def _get_z_from_mode_at_inference_time(item: dict, mode: str, inference_time: float):
    if mode not in item:
        raise KeyError(f"mode={mode} not found. Available keys: {list(item.keys())}")

    mode_item = item[mode]

    if "traj" in mode_item and "z" in mode_item["traj"]:
        traj = mode_item["traj"]
        idx = _nearest_traj_index(traj, inference_time)
        return np.asarray(traj["z"][idx], dtype=np.float32)

    if np.isclose(float(inference_time), 1.0) and "z_pred" in mode_item:
        return np.asarray(mode_item["z_pred"], dtype=np.float32)

    raise KeyError(
        f"Cannot get z for mode={mode}, inference_time={inference_time}. "
        f"Expected traj['z'] or z_pred."
    )


def _get_m_from_mode_at_inference_time(item: dict, mode: str, inference_time: float):
    """
    Extract mass m at a given inference time.

    Priority:
    1. mode_item["traj"]["m"][idx], where idx is nearest to inference_time
    2. mode_item["m_pred"] when inference_time == 1.0

    Notes:
    - m is assumed to be non-negative.
    - No abs() is applied.
    - No dataset-level normalization is applied here.
    """
    if mode not in item:
        raise KeyError(f"mode={mode} not found. Available keys: {list(item.keys())}")

    mode_item = item[mode]

    if "traj" in mode_item and "m" in mode_item["traj"]:
        traj = mode_item["traj"]
        idx = _nearest_traj_index(traj, inference_time)
        return np.asarray(traj["m"][idx], dtype=np.float32).reshape(-1)

    if np.isclose(float(inference_time), 1.0) and "m_pred" in mode_item:
        return np.asarray(mode_item["m_pred"], dtype=np.float32).reshape(-1)

    raise KeyError(
        f"Cannot get mass m for mode={mode}, inference_time={inference_time}. "
        f"Expected traj['m'] or m_pred."
    )


def _extract_real_target_z(adata_target, row_spec, sample_rep="X_pca_scaled"):
    if sample_rep not in adata_target.obsm:
        raise KeyError(f"{sample_rep} not found in adata_target.obsm")

    if "real_filters" in row_spec and row_spec["real_filters"] is not None:
        filters = dict(row_spec["real_filters"])
    elif "target_filters" in row_spec and row_spec["target_filters"] is not None:
        filters = dict(row_spec["target_filters"])
    else:
        filters = dict(row_spec["wish"])

    filters = {k: v for k, v in filters.items() if k in adata_target.obs.columns}

    if len(filters) == 0:
        raise ValueError("No valid filters for real target cells.")

    mask = _build_obs_mask(adata_target, filters)

    if mask.sum() == 0:
        raise ValueError(f"No real target cells with filters={filters}.")

    return (
        np.asarray(adata_target[mask].obsm[sample_rep], dtype=np.float32),
        int(mask.sum()),
        filters,
    )


# ---------------------------------------------------------------------------
# mass display helpers
# ---------------------------------------------------------------------------
def _transform_mass_for_display(m_values, mass_transform: str = "raw", eps: float = 1e-8):
    """
    Transform mass values for display.

    mass_transform:
    - "raw": display raw m
    - "log2": display log2(m), useful for growth / depletion visualization
    """
    m = np.asarray(m_values, dtype=float).reshape(-1)

    if mass_transform == "raw":
        return m

    if mass_transform == "log2":
        return np.log2(np.clip(m, eps, None))

    raise ValueError("mass_transform must be 'raw' or 'log2'.")


def _build_mass_norm_and_mappable(
    mass_arrays,
    mass_transform: str = "raw",
    mass_cmap: str = "Greys",
    mass_vmin: Optional[float] = 0.0,
    mass_vmax: Optional[float] = None,
    mass_high_q: float = 99.0,
):
    """
    Build global normalization for mass display.

    For raw mass:
        default vmin = 0
        default vmax = percentile(mass, mass_high_q)

    For log2 mass:
        default uses symmetric TwoSlopeNorm centered at 0
    """
    mass_display_arrays = [
        _transform_mass_for_display(x, mass_transform=mass_transform)
        for x in mass_arrays
    ]

    mass_cat = np.concatenate(
        [np.asarray(x).reshape(-1) for x in mass_display_arrays],
        axis=0,
    )
    mass_cat = mass_cat[np.isfinite(mass_cat)]

    if len(mass_cat) == 0:
        mass_cat = np.array([0.0], dtype=float)

    if mass_transform == "raw":
        if mass_vmin is None:
            vmin = 0.0
        else:
            vmin = float(mass_vmin)

        if mass_vmax is None:
            vmax = float(np.nanpercentile(mass_cat, mass_high_q))
        else:
            vmax = float(mass_vmax)

        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6

        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        label = r"Mass $m(t)$"

    elif mass_transform == "log2":
        if mass_vmin is None or mass_vmax is None:
            abs_v = float(np.nanpercentile(np.abs(mass_cat), mass_high_q))
            abs_v = max(abs_v, 1e-6)
            vmin = -abs_v
            vmax = abs_v
        else:
            vmin = float(mass_vmin)
            vmax = float(mass_vmax)

        if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
            vmin, vmax = -1.0, 1.0

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        label = r"$\log_2$ Mass"

    else:
        raise ValueError("mass_transform must be 'raw' or 'log2'.")

    mappable = ScalarMappable(norm=norm, cmap=mass_cmap)
    mappable.set_array([])

    return norm, mappable, label


# ---------------------------------------------------------------------------
# per-panel axis style
# ---------------------------------------------------------------------------
def _apply_nature_umap_axis_style(
    ax,
    xlim,
    ylim,
    show_ticks: bool = False,
    show_panel_border: bool = True,
    panel_border_color: str = "#D0D0D0",
    panel_border_lw: float = 0.35,
):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    if show_panel_border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(panel_border_color)
            spine.set_linewidth(panel_border_lw)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show_ticks:
        ax.tick_params(axis="both", labelsize=7, length=2, width=0.5, pad=1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")


# ---------------------------------------------------------------------------
# main plotting function
# ---------------------------------------------------------------------------
def plot_flow_inference_cellline_umap_grid_nm_style(
    results: Dict,
    row_specs: Sequence[dict],
    adata_source: ad.AnnData,
    adata_target: ad.AnnData,
    reducer,
    sample_rep: str = "X_pca_scaled",

    # flow-model inference time
    source_inference_time: float = 0.0,
    target_inference_time: float = 1.0,

    source_color: str = "#202020",
    source_alpha: float = 0.95,
    source_point_size: Optional[float] = None,

    # background
    adata_background: Optional[ad.AnnData] = None,
    background: str = "all_source",  # "all_source" | "none"

    # mass colormap display
    mass_cmap: str = "Greys",
    mass_transform: str = "raw",     # "raw" | "log2"
    mass_vmin: Optional[float] = 0.0,
    mass_vmax: Optional[float] = None,
    mass_high_q: float = 99.0,
    prediction_alpha: float = 0.90,
    sort_points_by_mass: bool = True,

    # observed target style
    observed_color: str = "#A8322E",
    observed_alpha: float = 0.90,

    # point style
    point_size: float = 5.5,
    observed_point_size: Optional[float] = None,
    bg_size: float = 1.5,
    bg_alpha: float = 0.035,
    background_color: str = "#CFCFCF",

    # panel divider
    show_panel_border: bool = True,
    panel_border_color: str = "#D0D0D0",
    panel_border_lw: float = 0.35,

    # layout
    figsize_per_panel: tuple = (1.35, 1.25),
    title_prefix: Optional[str] = None,
    show_ticks: bool = False,
    show_panel_labels: bool = True,
    panel_label: str = "a",
    rasterized: bool = True,

    # typography
    font_family: str = "Arial",
):
    """
    Nature Methods-style UMAP grid for flow / WFR / flow-matching inference.

    Columns:
    - Source
    - Full model
    - Target data
    - Velocity only
    - Growth only

    Main visual encoding:
    - Model prediction panels use colormap to display mass m(t).
    - Alpha is fixed, so visual intensity is not dominated by density stacking.
    - All model panels share one global mass color scale.
    - Observed target cells use a contrasting warm hue.

    Parameters
    ----------
    mass_transform : str
        - "raw": display raw mass m(t), closest to your old plotting function.
        - "log2": display log2(m), useful for growth/depletion visualization.

    mass_vmin, mass_vmax : Optional[float]
        Color scale limits after transformation.
        For raw mass, default mass_vmin=0 and mass_vmax=global percentile.
        For log2 mass, if either is None, a symmetric scale around 0 is used.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    if reducer is None:
        raise ValueError("Please provide a fitted UMAP reducer.")

    if len(row_specs) == 0:
        raise ValueError("row_specs is empty.")

    if observed_point_size is None:
        observed_point_size = point_size

    if source_point_size is None:
        source_point_size = point_size

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }

    with plt.rc_context(rc):

        # ------------------------------------------------------------------
        # background projection
        # ------------------------------------------------------------------
        if background == "all_source":
            if adata_background is None:
                adata_background = adata_source

            if sample_rep not in adata_background.obsm:
                raise KeyError(f"{sample_rep} not found in adata_background.obsm")

            bg_xy = reducer.transform(
                np.asarray(adata_background.obsm[sample_rep], dtype=np.float32)
            )

        elif background == "none":
            bg_xy = None

        else:
            raise ValueError("background must be 'all_source' or 'none'.")

        n_rows = len(row_specs)
        n_cols = 5

        col_titles = [
            "Source",
            "Full",
            "Target",
            "Velocity",
            "Growth",
        ]


        # ------------------------------------------------------------------
        # gather per-row data
        # ------------------------------------------------------------------
        row_cache = []
        all_xy_for_limits = []
        all_mass_for_norm = []

        for row_spec in row_specs:
            if "wish" not in row_spec:
                raise KeyError("Each row_spec must contain a 'wish' field.")

            wish = row_spec["wish"]
            row_label = row_spec.get("row_label", str(wish.get("cell_line", "NA")))
            row_sublabel = row_spec.get("row_sublabel", None)

            cond_name = row_spec.get("cond_name", None)
            if cond_name is None:
                cond_name = _find_result_key_by_wish(results, wish)

            item = results[cond_name]

            # Latent states
            z_initial = _get_z_from_mode_at_inference_time(
                item, "all", source_inference_time
            )
            z_all = _get_z_from_mode_at_inference_time(
                item, "all", target_inference_time
            )
            z_v_only = _get_z_from_mode_at_inference_time(
                item, "v_only", target_inference_time
            )
            z_g_only = _get_z_from_mode_at_inference_time(
                item, "g_only", target_inference_time
            )

            # Mass values
            m_initial = _get_m_from_mode_at_inference_time(
                item, "all", source_inference_time
            )
            m_all = _get_m_from_mode_at_inference_time(
                item, "all", target_inference_time
            )
            m_v_only = _get_m_from_mode_at_inference_time(
                item, "v_only", target_inference_time
            )
            m_g_only = _get_m_from_mode_at_inference_time(
                item, "g_only", target_inference_time
            )

            # Observed target cells
            z_real, n_real, real_filters = _extract_real_target_z(
                adata_target=adata_target,
                row_spec=row_spec,
                sample_rep=sample_rep,
            )

            # UMAP projection
            xy_initial = reducer.transform(z_initial)
            xy_all = reducer.transform(z_all)
            xy_real = reducer.transform(z_real)
            xy_v_only = reducer.transform(z_v_only)
            xy_g_only = reducer.transform(z_g_only)

            row_cache.append({
                "row_label": row_label,
                "row_sublabel": row_sublabel,
                "n_real": n_real,
                "real_filters": real_filters,

                "xy_initial": xy_initial,
                "xy_all": xy_all,
                "xy_real": xy_real,
                "xy_v_only": xy_v_only,
                "xy_g_only": xy_g_only,

                "m_initial": m_initial,
                "m_all": m_all,
                "m_v_only": m_v_only,
                "m_g_only": m_g_only,
            })

            all_xy_for_limits.extend([
                xy_initial,
                xy_all,
                xy_real,
                xy_v_only,
                xy_g_only,
            ])

            all_mass_for_norm.extend([
                m_initial,
                m_all,
                m_v_only,
                m_g_only,
            ])

        if bg_xy is not None:
            all_xy_for_limits.append(bg_xy)

        # ------------------------------------------------------------------
        # shared UMAP limits
        # ------------------------------------------------------------------
        x_cat = np.concatenate([xy[:, 0] for xy in all_xy_for_limits], axis=0)
        y_cat = np.concatenate([xy[:, 1] for xy in all_xy_for_limits], axis=0)

        x_min, x_max = np.nanmin(x_cat), np.nanmax(x_cat)
        y_min, y_max = np.nanmin(y_cat), np.nanmax(y_cat)

        x_pad = 0.035 * max(x_max - x_min, 1e-6)
        y_pad = 0.035 * max(y_max - y_min, 1e-6)

        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        # ------------------------------------------------------------------
        # global mass color scale
        # ------------------------------------------------------------------
        mass_norm, mass_mappable, mass_cbar_label = _build_mass_norm_and_mappable(
            mass_arrays=all_mass_for_norm,
            mass_transform=mass_transform,
            mass_cmap=mass_cmap,
            mass_vmin=mass_vmin,
            mass_vmax=mass_vmax,
            mass_high_q=mass_high_q,
        )

        # ------------------------------------------------------------------
        # figure
        # ------------------------------------------------------------------
        fig_w = figsize_per_panel[0] * n_cols + 0.95
        fig_h = figsize_per_panel[1] * n_rows + 0.70

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )

        for r, row_data in enumerate(row_cache):
            panels = [
                ("initial",  row_data["xy_initial"], row_data["m_initial"]),
                ("all",      row_data["xy_all"],     row_data["m_all"]),
                ("observed", row_data["xy_real"],    None),
                ("v_only",   row_data["xy_v_only"],  row_data["m_v_only"]),
                ("g_only",   row_data["xy_g_only"],  row_data["m_g_only"]),
            ]

            for c, (panel_name, xy, m_values) in enumerate(panels):
                ax = axes[r, c]

                # Background layer
                if bg_xy is not None:
                    ax.scatter(
                        bg_xy[:, 0],
                        bg_xy[:, 1],
                        s=bg_size,
                        c=background_color,
                        alpha=bg_alpha,
                        linewidths=0,
                        rasterized=rasterized,
                        zorder=1,
                    )

                # Foreground layer
                if panel_name == "initial":
                    ax.scatter(
                        xy[:, 0],
                        xy[:, 1],
                        s=source_point_size,
                        c=source_color,
                        alpha=source_alpha,
                        linewidths=0,
                        rasterized=rasterized,
                        zorder=3,
                    )
                
                elif panel_name == "observed":
                    ax.scatter(
                        xy[:, 0],
                        xy[:, 1],
                        s=observed_point_size,
                        c=observed_color,
                        alpha=observed_alpha,
                        linewidths=0,
                        rasterized=rasterized,
                        zorder=3,
                    )
                
                else:
                    m_values = np.asarray(m_values, dtype=float).reshape(-1)
                    m_display = _transform_mass_for_display(
                        m_values,
                        mass_transform=mass_transform,
                    )
                
                    if len(m_display) != xy.shape[0]:
                        raise ValueError(
                            f"Mass length mismatch in panel={panel_name}: "
                            f"len(m)={len(m_display)}, n_points={xy.shape[0]}"
                        )
                
                    if sort_points_by_mass:
                        order = np.argsort(m_display)
                        xy_plot = xy[order]
                        m_plot = m_display[order]
                    else:
                        xy_plot = xy
                        m_plot = m_display
                
                    ax.scatter(
                        xy_plot[:, 0],
                        xy_plot[:, 1],
                        s=point_size,
                        c=m_plot,
                        cmap=mass_cmap,
                        norm=mass_norm,
                        alpha=prediction_alpha,
                        linewidths=0,
                        rasterized=rasterized,
                        zorder=3,
                    )


                _apply_nature_umap_axis_style(
                    ax,
                    xlim,
                    ylim,
                    show_ticks=show_ticks,
                    show_panel_border=show_panel_border,
                    panel_border_color=panel_border_color,
                    panel_border_lw=panel_border_lw,
                )

                if r == 0:
                    ax.set_title(col_titles[c], fontsize=8.2, pad=4)

                if c == 0:
                    if row_data["row_sublabel"] is None:
                        ax.text(
                            -0.12,
                            0.5,
                            row_data["row_label"],
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            fontstyle="normal",
                            rotation=90,
                        )
                    else:
                        ax.text(
                            -0.15,
                            0.5,
                            row_data["row_label"],
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            fontstyle="normal",
                            rotation=90,
                        )
                        ax.text(
                            -0.08,
                            0.5,
                            row_data["row_sublabel"],
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=7,
                            fontweight="normal",
                            rotation=90,
                        )

        # ------------------------------------------------------------------
        # layout margins
        # ------------------------------------------------------------------
        fig.subplots_adjust(
            left=0.06,
            right=0.885,
            top=0.88 if title_prefix is None else 0.84,
            bottom=0.12,
            wspace=0.10,
            hspace=0.13,
        )

        # ------------------------------------------------------------------
        # Mass colorbar
        # ------------------------------------------------------------------
        cax = fig.add_axes([0.905, 0.32, 0.014, 0.38])
        cbar = fig.colorbar(mass_mappable, cax=cax)
        cbar.set_label(
            r"Mass $m(t)$",
            fontsize=8.5,
            rotation=270,
            labelpad=12,
        )
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.tick_params(
            axis="y",
            labelsize=7.5,
            length=2.0,
            width=0.5,
            pad=2,
        )
        # ------------------------------------------------------------------
        # Bottom proxy legend
        # ------------------------------------------------------------------
        model_color = plt.get_cmap(mass_cmap)(0.75)

        pred_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=model_color,
            markeredgecolor="none",
            markersize=6.5,
            label="Model prediction",
        )

        obs_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=observed_color,
            markeredgecolor="none",
            markersize=6.5,
            label="Target data",
        )

        fig.legend(
            handles=[pred_handle, obs_handle],
            loc="lower center",
            bbox_to_anchor=(0.48, 0.015),
            ncol=2,
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            columnspacing=1.8,
        )

        if title_prefix is not None:
            fig.suptitle(
                title_prefix,
                y=0.98,
                fontsize=10,
                fontweight="bold",
            )

        if show_panel_labels:
            fig.text(
                0.012,
                0.985,
                panel_label,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
            )

        return fig, axes
