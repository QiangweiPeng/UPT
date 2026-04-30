import copy
import itertools
import ast
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import run_inference_only


# =============================================================================
# 1. Dose / covariate interpolation
# =============================================================================

def make_interp_values(spec: dict) -> list[float]:
    """
    Examples
    --------
    {"values": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
    {"start": 0.01, "end": 10.0, "n": 12, "spacing": "log"}
    {"start": 6.0, "end": 72.0, "n": 6, "spacing": "linear"}
    """
    if "values" in spec:
        values = np.asarray(spec["values"], dtype=float)
    else:
        start = float(spec["start"])
        end = float(spec["end"])
        n = int(spec["n"])
        spacing = spec.get("spacing", "linear")

        if n <= 1:
            values = np.asarray([start], dtype=float)
        elif spacing == "linear":
            values = np.linspace(start, end, n)
        elif spacing == "log":
            if start <= 0 or end <= 0:
                raise ValueError("Log interpolation requires start > 0 and end > 0.")
            values = np.geomspace(start, end, n)
        else:
            raise ValueError(f"Unknown spacing={spacing}. Use 'linear' or 'log'.")

    round_ndigits = spec.get("round_ndigits", None)
    if round_ndigits is not None:
        values = np.round(values, round_ndigits)

    return values.astype(float).tolist()


def _canonicalize_for_grouping(x, ndigits: int = 12):
    if isinstance(x, (float, np.floating)):
        if np.isnan(float(x)):
            return None
        return round(float(x), ndigits)

    if isinstance(x, (int, np.integer)):
        return int(x)

    if isinstance(x, (list, tuple)):
        return tuple(_canonicalize_for_grouping(v, ndigits=ndigits) for v in x)

    return x


def deduplicate_wishlist_excluding_keys(
    wishlist: list[dict],
    exclude_keys: Sequence[str],
    keep: str = "first",
) -> list[dict]:
    """
    Deduplicate biological conditions while ignoring interpolated covariates.

    For dose interpolation, this prevents:
        drug A dose 0.1
        drug A dose 1.0
        drug A dose 10.0

    from each being expanded again into the full dose grid.
    """
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'.")

    exclude_keys = set(exclude_keys)
    grouped = {}

    for wish in wishlist:
        key_items = []

        for k in sorted(wish.keys()):
            if k in exclude_keys:
                continue
            if str(k).startswith("_interp_"):
                continue

            key_items.append((k, _canonicalize_for_grouping(wish[k])))

        group_key = tuple(key_items)

        if keep == "first":
            if group_key not in grouped:
                grouped[group_key] = copy.deepcopy(wish)
        else:
            grouped[group_key] = copy.deepcopy(wish)

    return list(grouped.values())


def build_interpolated_wishlist(
    base_wishlist: list[dict],
    covariate_specs: dict[str, dict],
    mode: str = "cartesian",
    deduplicate_before_interpolation: bool = True,
    attach_interp_meta: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Build interpolated wishlist.

    Typical dose interpolation
    --------------------------
    covariate_specs = {
        "dose_value": {
            "start": 0.01,
            "end": 10.0,
            "n": 16,
            "spacing": "log",
            "round_ndigits": 4,
        }
    }
    """
    if len(covariate_specs) == 0:
        return [copy.deepcopy(w) for w in base_wishlist]

    covariate_names = list(covariate_specs.keys())
    covariate_values = {
        k: make_interp_values(v)
        for k, v in covariate_specs.items()
    }

    if mode == "paired":
        lengths = [len(v) for v in covariate_values.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"In paired mode, all covariates must have equal length. Got {lengths}."
            )

    if deduplicate_before_interpolation:
        unique_base = deduplicate_wishlist_excluding_keys(
            base_wishlist,
            exclude_keys=covariate_names,
            keep="first",
        )
    else:
        unique_base = [copy.deepcopy(w) for w in base_wishlist]

    expanded = []

    for base_idx, wish in enumerate(unique_base):
        if mode == "cartesian":
            combos = itertools.product(*(covariate_values[k] for k in covariate_names))
        elif mode == "paired":
            combos = zip(*(covariate_values[k] for k in covariate_names))
        else:
            raise ValueError(f"Unknown mode={mode}. Use 'cartesian' or 'paired'.")

        for interp_idx, combo in enumerate(combos):
            new_wish = copy.deepcopy(wish)

            for k, v in zip(covariate_names, combo):
                new_wish[k] = float(v)

            if attach_interp_meta:
                new_wish["_interp_base_idx"] = int(base_idx)
                new_wish["_interp_idx"] = int(interp_idx)
                new_wish["_interp_covariates"] = tuple(covariate_names)

            expanded.append(new_wish)

    if verbose:
        print(f"Base wishlist size: {len(base_wishlist)}")
        print(f"Unique base wishlist size: {len(unique_base)}")
        print(f"Expanded wishlist size: {len(expanded)}")

    return expanded


# =============================================================================
# 2. Inference wrapper for interpolated dose grid
# =============================================================================

def run_dose_interpolation_inference(
    model,
    adata_control,
    adata_test,
    wishlist,
    condition_keys,
    condition_rep_keys,
    sample_rep,
    device,
    covariate_specs: dict[str, dict],
    covariate_interp_mode: str = "cartesian",
    n_particles: int = 10000,
    random_seed: int = 42,
    n_steps: int = 50,
    scvi_model_load_path: Optional[str] = None,
    state_model_load_path: Optional[str] = None,
    source_celltype_key: Optional[str] = None,
    return_traj: bool = True,
    save_times: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    t_destiny: float = 1.0,
    return_reducer: bool = False,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.35,
    umap_random_state: int = 0,
    inference_modes: Sequence[str] = ("all",),
    verbose: bool = True,
):
    """
    Minimal inference wrapper for dose interpolation.

    This version intentionally does not do repeated control resampling.
    The uncertainty-like visual element in the final figure comes from
    the predicted particle distribution m_pred at each interpolated dose.
    """
    expanded_wishlist = build_interpolated_wishlist(
        base_wishlist=wishlist,
        covariate_specs=covariate_specs,
        mode=covariate_interp_mode,
        deduplicate_before_interpolation=True,
        attach_interp_meta=True,
        verbose=verbose,
    )

    output = run_inference_only(
        model=model,
        adata_control=adata_control,
        adata_test=adata_test,
        wishlist=expanded_wishlist,
        condition_keys=condition_keys,
        condition_rep_keys=condition_rep_keys,
        sample_rep=sample_rep,
        device=device,
        n_particles=n_particles,
        random_seed=random_seed,
        n_steps=n_steps,
        scvi_model_load_path=scvi_model_load_path,
        state_model_load_path=state_model_load_path,
        source_celltype_key=source_celltype_key,
        return_traj=return_traj,
        save_times=save_times,
        t_destiny=t_destiny,
        return_reducer=return_reducer,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
        inference_modes=tuple(inference_modes),
    )

    if isinstance(output, tuple):
        results = output[0]
        extras = output[1:]
        return results, extras

    return output


# =============================================================================
# 3. Flatten inference result to particle-level dataframe
# =============================================================================

def _unpack_results(results):
    """
    Accept either:
        results_dict
    or:
        (results_dict, extras...)
    """
    if isinstance(results, tuple):
        if len(results) == 0:
            raise ValueError("Got an empty tuple as results.")
        results = results[0]

    if not isinstance(results, dict):
        raise TypeError(
            "Expected results to be a dict or a tuple whose first element is a dict. "
            f"Got {type(results)}."
        )

    return results

def flatten_inference_results_to_particle_df(
    results: dict,
    mode: str = "all",
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    dose_key: str = "dose_value",
    time_key: str = "time",
    particle_subsample: Optional[int] = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Convert inference results to one row per predicted particle.

    Accepts either:
        results
    or:
        (results, extras...)
    """
    results = _unpack_results(results)

    rng = np.random.default_rng(random_seed)
    rows = []

    for cond_name, cond_result in results.items():
        if mode not in cond_result:
            continue

        wish = cond_result.get("wish", {})
        m_pred = np.asarray(cond_result[mode]["m_pred"]).reshape(-1)
        m_pred = m_pred[np.isfinite(m_pred)]

        if len(m_pred) == 0:
            continue

        if particle_subsample is not None and len(m_pred) > particle_subsample:
            keep_idx = rng.choice(len(m_pred), size=int(particle_subsample), replace=False)
            m_pred = m_pred[keep_idx]

        for i, value in enumerate(m_pred):
            rows.append({
                "cond_name": cond_name,
                perturb_key: wish.get(perturb_key),
                cellline_key: wish.get(cellline_key),
                dose_key: wish.get(dose_key),
                time_key: wish.get(time_key),
                "mass_ratio": float(value),
                "particle_idx": int(i),
            })

    df = pd.DataFrame(rows)

    for col in [dose_key, time_key, "mass_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



def summarize_particle_df(
    particle_df: pd.DataFrame,
    x_key: str = "dose_value",
    y_key: str = "mass_ratio",
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    fixed_values: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Summarize particle-level predictions for overlaying trend lines.
    """
    df = particle_df.copy()

    if fixed_values is not None:
        for k, v in fixed_values.items():
            if k in df.columns:
                df = df[df[k].astype(str) == str(v)]

    df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
    df[y_key] = pd.to_numeric(df[y_key], errors="coerce")

    df = df.dropna(subset=[perturb_key, cellline_key, x_key, y_key])

    if len(df) == 0:
        return pd.DataFrame()

    summary = (
        df
        .groupby([perturb_key, cellline_key, x_key], as_index=False)
        .agg(
            pred_mean=(y_key, "mean"),
            pred_median=(y_key, "median"),
            pred_q10=(y_key, lambda x: np.quantile(x, 0.10)),
            pred_q25=(y_key, lambda x: np.quantile(x, 0.25)),
            pred_q75=(y_key, lambda x: np.quantile(x, 0.75)),
            pred_q90=(y_key, lambda x: np.quantile(x, 0.90)),
            n_particles=(y_key, "size"),
        )
        .sort_values([perturb_key, cellline_key, x_key])
    )

    return summary


# =============================================================================
# 4. Optional: observed true mass ratio
# =============================================================================

def _get_matched_control(
    adata_control,
    pert_tuple_str,
    rulebook,
    use_groupwise_control: bool = True,
):
    """
    Use rulebook to find matched controls.
    Falls back to global control if matching fails.
    """
    if not use_groupwise_control or not rulebook:
        return adata_control

    control_groups = rulebook.get("stratification", {}).get("control_groups", [])
    if len(control_groups) == 0:
        return adata_control

    control_groups = list(control_groups)
    schema = list(rulebook.get("condition_tuple_schema", []))

    try:
        actual_tuple = ast.literal_eval(pert_tuple_str)
        mask = np.ones(adata_control.n_obs, dtype=bool)

        for i, col in enumerate(schema):
            if col in control_groups:
                mask &= (
                    adata_control.obs[col].astype(str).to_numpy()
                    == str(actual_tuple[i])
                )

        matched = adata_control[mask]

        if matched.n_obs > 0:
            return matched

        print(f"Warning: no matched control for {pert_tuple_str}; using global control.")
        return adata_control

    except Exception as e:
        print(f"Warning: failed to parse {pert_tuple_str}: {e}; using global control.")
        return adata_control


def _build_obs_mask_from_row(
    adata,
    row_dict: dict,
    match_keys: Sequence[str],
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    mask = np.ones(adata.n_obs, dtype=bool)
    obs = adata.obs

    for k in match_keys:
        if k not in obs.columns or k not in row_dict:
            continue

        value = row_dict[k]

        if pd.isna(value):
            continue

        series = obs[k]
        series_num = pd.to_numeric(series, errors="coerce")
        value_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

        if pd.notna(value_num) and series_num.notna().all():
            mask &= np.isclose(
                series_num.to_numpy(dtype=float),
                float(value_num),
                atol=atol,
                rtol=rtol,
            )
        else:
            mask &= series.astype(str).to_numpy() == str(value)

    return mask


def _row_to_condition_tuple_string(
    row_dict: dict,
    rulebook: dict,
    cond_name_key: str = "cond_name",
):
    if cond_name_key in row_dict and pd.notna(row_dict[cond_name_key]):
        return str(row_dict[cond_name_key])

    schema = list(rulebook.get("condition_tuple_schema", []))
    if len(schema) == 0:
        raise ValueError("rulebook['condition_tuple_schema'] is empty.")

    return str(tuple(row_dict.get(k, None) for k in schema))


def add_true_mass_ratio_to_df(
    df: pd.DataFrame,
    adata_treated,
    adata_control,
    rulebook: dict,
    x_key: str,
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    fixed_values: Optional[dict] = None,
    mass_deduct_key: Optional[str] = None,
    true_col: str = "true_mass_ratio",
    treated_count_col: str = "n_true_cells",
    control_count_col: str = "n_ctrl_cells",
    cond_name_key: str = "cond_name",
    use_groupwise_control: bool = True,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> pd.DataFrame:
    """
    Add observed true mass ratio to a dataframe containing conditions.

    true_mass_ratio = treated total mass / matched-control total mass

    This function is useful for generating the observed points overlaid on the
    violin-style prediction panel.
    """
    out = df.copy()

    if fixed_values is not None:
        for k, v in fixed_values.items():
            if k not in out.columns:
                out[k] = v
            else:
                out[k] = out[k].fillna(v)

    m_source = float(adata_control.uns.get("normalized_m", 1.0))
    m_target = float(adata_treated.uns.get("normalized_m", 1.0))

    fixed_keys = list(fixed_values.keys()) if fixed_values is not None else []

    treated_match_keys = list(dict.fromkeys(
        [perturb_key, cellline_key, x_key] + fixed_keys
    ))
    control_refine_keys = list(dict.fromkeys(
        [cellline_key] + fixed_keys
    ))

    true_values = []
    n_true_values = []
    n_ctrl_values = []

    for _, row in out.iterrows():
        row_dict = row.to_dict()

        treated_mask = _build_obs_mask_from_row(
            adata_treated,
            row_dict=row_dict,
            match_keys=treated_match_keys,
            atol=atol,
            rtol=rtol,
        )
        ad_true = adata_treated[treated_mask]

        pert_tuple_str = _row_to_condition_tuple_string(
            row_dict=row_dict,
            rulebook=rulebook,
            cond_name_key=cond_name_key,
        )

        matched_control = _get_matched_control(
            adata_control=adata_control,
            pert_tuple_str=pert_tuple_str,
            rulebook=rulebook,
            use_groupwise_control=use_groupwise_control,
        )

        ctrl_mask = _build_obs_mask_from_row(
            matched_control,
            row_dict=row_dict,
            match_keys=control_refine_keys,
            atol=atol,
            rtol=rtol,
        )
        ad_ctrl = matched_control[ctrl_mask]

        n_true = int(ad_true.n_obs)
        n_ctrl = int(ad_ctrl.n_obs)

        n_true_values.append(n_true)
        n_ctrl_values.append(n_ctrl)

        if n_true == 0 or n_ctrl == 0:
            true_values.append(np.nan)
            continue

        adjusted_m_target = m_target

        if (
            mass_deduct_key is not None
            and mass_deduct_key in ad_true.obs.columns
            and mass_deduct_key in ad_ctrl.obs.columns
        ):
            n_unique_source = len(ad_ctrl.obs[mass_deduct_key].unique())
            n_unique_target = len(ad_true.obs[mass_deduct_key].unique())
            adjusted_m_target = (
                m_target
                * max(1, n_unique_source)
                / max(1, n_unique_target)
            )

        true_mass_ratio = (n_true * adjusted_m_target) / (n_ctrl * m_source)
        true_values.append(float(true_mass_ratio))

    out[true_col] = true_values
    out[treated_count_col] = n_true_values
    out[control_count_col] = n_ctrl_values

    return out


# =============================================================================
# 5. KDE helpers for violin-style dose panel
# =============================================================================

def _silverman_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    if n <= 1:
        return 1.0

    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = std

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(abs(np.mean(values)) * 0.05, 1e-3)

    return 0.9 * sigma * n ** (-1 / 5)


def _gaussian_kde_1d(
    values: np.ndarray,
    grid: np.ndarray,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """
    Lightweight 1D Gaussian KDE to avoid adding scipy as a dependency.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    grid = np.asarray(grid, dtype=float)

    if len(values) == 0:
        return np.zeros_like(grid)

    if bandwidth is None:
        bandwidth = _silverman_bandwidth(values)

    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = _silverman_bandwidth(values)

    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z ** 2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2 * np.pi)

    return density


def _format_dose_tick(x: float) -> str:
    if not np.isfinite(x):
        return ""

    if x == 0:
        return "0"

    if abs(x) >= 100 or abs(x) < 0.01:
        return f"{x:.0e}"

    if abs(x) < 1:
        return f"{x:.3g}"

    return f"{x:g}"


def _to_plot_x(values: np.ndarray, logx: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if logx:
        if np.any(values <= 0):
            raise ValueError("logx=True requires all x values to be positive.")
        return np.log10(values)

    return values


# =============================================================================
# 6. Main figure function: dose-wise violin density panel
# =============================================================================
