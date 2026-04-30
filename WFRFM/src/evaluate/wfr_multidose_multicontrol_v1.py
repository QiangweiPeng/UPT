import copy
import itertools
import ast
from typing import Dict, List, Sequence, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import run_inference_only




# =============================================================================
# 1. Covariate interpolation utilities
# =============================================================================

def _make_interp_values(spec: dict) -> list[float]:
    """
    spec examples
    -------------
    {"values": [6, 12, 24, 48]}
    {"start": 0.01, "end": 10.0, "n": 8, "spacing": "log"}
    {"start": 6.0, "end": 72.0, "n": 6, "spacing": "linear"}
    """
    if "values" in spec:
        vals = np.asarray(spec["values"], dtype=float)
    else:
        start = float(spec["start"])
        end = float(spec["end"])
        n = int(spec["n"])
        spacing = spec.get("spacing", "linear")

        if n <= 1:
            vals = np.asarray([start], dtype=float)
        elif spacing == "linear":
            vals = np.linspace(start, end, n)
        elif spacing == "log":
            if start <= 0 or end <= 0:
                raise ValueError("log interpolation requires start > 0 and end > 0")
            vals = np.geomspace(start, end, n)
        else:
            raise ValueError(f"Unknown spacing={spacing}. Use 'linear' or 'log'.")

    round_ndigits = spec.get("round_ndigits", None)
    if round_ndigits is not None:
        vals = np.round(vals, round_ndigits)

    return vals.astype(float).tolist()


def build_covariate_interpolated_wishlist(
    base_wishlist: list[dict],
    covariate_specs: dict[str, dict],
    mode: str = "cartesian",
    attach_interp_meta: bool = True,
) -> list[dict]:
    """
    Expand each base wish into multiple wishes with interpolated continuous covariates.

    Parameters
    ----------
    base_wishlist
        Original wishlist. Each wish should already contain perturbation / cell_line /
        dose / time fields as needed.
    covariate_specs
        Example:
        {
            "dose_value": {"start": 0.01, "end": 10.0, "n": 8, "spacing": "log"},
            "time": {"values": [6, 12, 24, 48, 72]}
        }
    mode
        - "cartesian": Cartesian product across all covariates.
        - "paired": Pair the i-th value of each covariate. Lengths must match.
    attach_interp_meta
        Whether to add interpolation metadata to each wish.

    Returns
    -------
    expanded_wishlist
    """
    if len(covariate_specs) == 0:
        return [copy.deepcopy(w) for w in base_wishlist]

    cov_names = list(covariate_specs.keys())
    cov_values_dict = {k: _make_interp_values(v) for k, v in covariate_specs.items()}

    if mode == "paired":
        lengths = [len(v) for v in cov_values_dict.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"In paired mode, all covariates must have the same number of values, got {lengths}"
            )

    expanded = []

    for base_idx, wish in enumerate(base_wishlist):
        if mode == "cartesian":
            combos = itertools.product(*(cov_values_dict[k] for k in cov_names))
        elif mode == "paired":
            combos = zip(*(cov_values_dict[k] for k in cov_names))
        else:
            raise ValueError(f"Unknown mode={mode}. Use 'cartesian' or 'paired'.")

        for interp_idx, combo in enumerate(combos):
            new_wish = copy.deepcopy(wish)

            for cov_name, cov_val in zip(cov_names, combo):
                new_wish[cov_name] = float(cov_val)

            if attach_interp_meta:
                new_wish["_interp_base_idx"] = base_idx
                new_wish["_interp_idx"] = interp_idx
                new_wish["_interp_covariates"] = tuple(cov_names)

            expanded.append(new_wish)

    return expanded



def _canonicalize_value_for_grouping(v, ndigits=12):
    """
    Make values stable for grouping wishlist entries.
    """
    if isinstance(v, float):
        if np.isnan(v):
            return None
        return round(v, ndigits)

    if isinstance(v, np.floating):
        if np.isnan(float(v)):
            return None
        return round(float(v), ndigits)

    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, (list, tuple)):
        return tuple(_canonicalize_value_for_grouping(x, ndigits=ndigits) for x in v)

    return v


def deduplicate_wishlist_excluding_covariates(
    wishlist: list[dict],
    exclude_keys: list[str] | tuple[str, ...],
    keep: str = "first",
    verbose: bool = True,
) -> list[dict]:
    """
    Deduplicate wishlist entries while ignoring specified covariate keys.

    Use case
    --------
    For dose interpolation, the original wishlist may contain multiple observed
    dose values for the same perturbation / cell line / time combination.

    We want:
        one base wish per non-dose condition
        then expand dose_value to desired interpolation values

    Parameters
    ----------
    wishlist
        Original wishlist.
    exclude_keys
        Keys ignored during deduplication, usually ["dose_value"].
    keep
        Currently supports "first" or "last".
    verbose
        Print dedup summary.

    Returns
    -------
    deduped_wishlist
    """
    exclude_keys = set(exclude_keys)

    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'.")

    grouped = {}

    for idx, wish in enumerate(wishlist):
        key_items = []

        for k in sorted(wish.keys()):
            if k in exclude_keys:
                continue

            # Interpolation/control metadata should not define biological identity.
            if str(k).startswith("_interp_") or str(k).startswith("_control_"):
                continue

            key_items.append((k, _canonicalize_value_for_grouping(wish[k])))

        group_key = tuple(key_items)

        if keep == "first":
            if group_key not in grouped:
                grouped[group_key] = copy.deepcopy(wish)
        else:
            grouped[group_key] = copy.deepcopy(wish)

    deduped = list(grouped.values())

    if verbose:
        print(
            f"Deduplicated wishlist excluding {sorted(exclude_keys)}: "
            f"{len(wishlist)} -> {len(deduped)}"
        )

    return deduped

def build_covariate_interpolated_wishlist_from_unique_base(
    base_wishlist: list[dict],
    covariate_specs: dict[str, dict],
    mode: str = "cartesian",
    deduplicate_excluding_interp_covariates: bool = True,
    dedup_keep: str = "first",
    attach_interp_meta: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Build interpolated wishlist after deduplicating original wishlist by
    all fields except the interpolated covariates.

    For dose interpolation:
        original wishlist:
            A549 + drugA + dose 0.1
            A549 + drugA + dose 1.0
            A549 + drugA + dose 10.0

        covariate_specs:
            {"dose_value": {"values": [...]}}

        desired:
            first deduplicate to one base wish:
                A549 + drugA
            then expand to all requested dose values.

    This avoids expanding each observed dose separately.
    """
    interp_covariates = list(covariate_specs.keys())

    if deduplicate_excluding_interp_covariates:
        unique_base_wishlist = deduplicate_wishlist_excluding_covariates(
            wishlist=base_wishlist,
            exclude_keys=interp_covariates,
            keep=dedup_keep,
            verbose=verbose,
        )
    else:
        unique_base_wishlist = [copy.deepcopy(w) for w in base_wishlist]

    expanded_wishlist = build_covariate_interpolated_wishlist(
        base_wishlist=unique_base_wishlist,
        covariate_specs=covariate_specs,
        mode=mode,
        attach_interp_meta=attach_interp_meta,
    )

    if verbose:
        print(f"Unique base wishlist size: {len(unique_base_wishlist)}")
        print(f"Expanded wishlist size: {len(expanded_wishlist)}")

    return expanded_wishlist

# =============================================================================
# 2. Matched control utilities
# =============================================================================

def _get_matched_control(
    adata_control,
    pert_tuple_str,
    rulebook,
    use_groupwise_control=True,
):
    """
    根据 rulebook 动态解析 tuple 字符串，从 adata_control 中筛选精准的 control 子集。
    若匹配失败则回退到全局 control。
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
                mask &= (adata_control.obs[col].astype(str).to_numpy() == str(actual_tuple[i]))

        curr_ctrl_sub = adata_control[mask]

        if curr_ctrl_sub.n_obs > 0:
            return curr_ctrl_sub
        else:
            print(f"Warning: no groupwise control found for {pert_tuple_str}; fallback to global control.")
            return adata_control

    except Exception as e:
        print(f"Warning: failed to parse control tuple {pert_tuple_str} ({e}); fallback to global control.")
        return adata_control


def _build_obs_mask_from_row(
    adata,
    row_dict,
    match_keys,
    atol=1e-8,
    rtol=1e-5,
):
    """
    Build obs mask using row_dict values.

    Numeric columns are matched with np.isclose.
    Non-numeric columns are matched as strings.
    """
    mask = np.ones(adata.n_obs, dtype=bool)
    obs = adata.obs

    for k in match_keys:
        if k not in obs.columns:
            continue
        if k not in row_dict:
            continue

        val = row_dict[k]
        if pd.isna(val):
            continue

        ser = obs[k]
        ser_num = pd.to_numeric(ser, errors="coerce")
        val_num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]

        if pd.notna(val_num) and ser_num.notna().all():
            mask &= np.isclose(
                ser_num.to_numpy(dtype=float),
                float(val_num),
                atol=atol,
                rtol=rtol,
            )
        else:
            mask &= (ser.astype(str).to_numpy() == str(val))

    return mask


def _row_to_pert_tuple_str(
    row_dict,
    rulebook,
    cond_name_key="cond_name",
):
    """
    Priority:
    1. row['cond_name']
    2. tuple constructed from rulebook['condition_tuple_schema']
    """
    if cond_name_key in row_dict and pd.notna(row_dict[cond_name_key]):
        return str(row_dict[cond_name_key])

    schema = list(rulebook.get("condition_tuple_schema", []))
    if len(schema) == 0:
        raise ValueError("rulebook['condition_tuple_schema'] is empty, cannot build pert_tuple_str.")

    tup = tuple(row_dict.get(k, None) for k in schema)
    return str(tup)


def _wish_to_pert_tuple_str(
    wish,
    rulebook,
    cond_name_key="cond_name",
):
    """
    Same logic as _row_to_pert_tuple_str, but for a wish dict.
    """
    if cond_name_key in wish and wish[cond_name_key] is not None:
        return str(wish[cond_name_key])

    schema = list(rulebook.get("condition_tuple_schema", []))
    if len(schema) == 0:
        raise ValueError("rulebook['condition_tuple_schema'] is empty, cannot build pert_tuple_str.")

    tup = tuple(wish.get(k, None) for k in schema)
    return str(tup)


def _sample_adata_obs(
    adata,
    n_obs=None,
    frac=None,
    replace=True,
    random_state=None,
):
    """
    Sample cells from an AnnData object by obs position.

    Defaults:
    - If n_obs and frac are both None, sample adata.n_obs cells.
    - replace=True gives bootstrap-style control resampling.
    """
    if adata.n_obs == 0:
        raise ValueError("Cannot sample from an empty AnnData object.")

    rng = np.random.default_rng(random_state)

    if n_obs is None:
        if frac is not None:
            n_obs = int(np.ceil(adata.n_obs * float(frac)))
        else:
            n_obs = adata.n_obs

    n_obs = int(n_obs)
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}.")

    sampled_pos = rng.choice(
        np.arange(adata.n_obs),
        size=n_obs,
        replace=replace,
    )

    return adata[sampled_pos].copy()


# =============================================================================
# 3. Inference with covariate interpolation + matched-control resampling
# =============================================================================

def run_inference_only_with_covariate_interpolation(
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
    scvi_model_load_path: str = None,
    state_model_load_path: str = None,
    source_celltype_key: str | None = None,

    return_traj: bool = True,
    save_times: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    t_destiny: float = 1.0,
    return_reducer: bool = False,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.35,
    umap_random_state: int = 0,
    inference_modes: tuple[str, ...] = ("all", "v_only", "g_only"),
):
    """
    Original interpolation-only wrapper.
    Kept for compatibility.
    """
    expanded_wishlist = build_covariate_interpolated_wishlist(
        base_wishlist=wishlist,
        covariate_specs=covariate_specs,
        mode=covariate_interp_mode,
        attach_interp_meta=True,
    )

    print(f"Original wishlist size: {len(wishlist)}")
    print(f"Expanded wishlist size: {len(expanded_wishlist)}")

    return run_inference_only(
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
        inference_modes=inference_modes,
    )

def _unpack_run_inference_only_output(output):
    """
    run_inference_only may return either:
        results
    or:
        (results, ...)
    
    This helper extracts the results dict while preserving extra outputs.
    """
    if isinstance(output, tuple):
        if len(output) == 0:
            raise ValueError("run_inference_only returned an empty tuple.")
        results = output[0]
        extras = output[1:]
    else:
        results = output
        extras = None

    if not isinstance(results, dict):
        raise TypeError(
            "The first output of run_inference_only must be a dict-like results object. "
            f"Got type: {type(results)}"
        )

    return results, extras


def run_inference_only_with_covariate_interpolation_and_control_resampling(
    model,
    adata_control,
    adata_test,
    wishlist,
    condition_keys,
    condition_rep_keys,
    sample_rep,
    device,
    rulebook,

    covariate_specs: dict[str, dict],
    covariate_interp_mode: str = "cartesian",

    n_control_repeats: int = 30,
    control_sample_n_obs: int | None = None,
    control_sample_frac: float | None = None,
    control_replace: bool = True,
    use_groupwise_control: bool = True,

    n_particles: int = 10000,
    random_seed: int = 42,
    n_steps: int = 50,
    scvi_model_load_path: str = None,
    state_model_load_path: str = None,
    source_celltype_key: str | None = None,

    return_traj: bool = True,
    save_times: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    t_destiny: float = 1.0,
    return_reducer: bool = False,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.35,
    umap_random_state: int = 0,
    inference_modes: tuple[str, ...] = ("all", "v_only", "g_only"),

    cond_name_key: str = "cond_name",
    verbose: bool = True,
    return_extras: bool = False,
):
    """
    Recommended wrapper for dose interpolation main figure.

    For each interpolated wish:
        1. Find the corresponding matched control.
        2. Resample matched control n_control_repeats times.
        3. Run inference once per sampled control.
        4. Store replicate metadata.

    Returns
    -------
    all_results : dict

    If return_extras=True, returns:
        all_results, all_extras
    """
    expanded_wishlist = build_covariate_interpolated_wishlist_from_unique_base(
        base_wishlist=wishlist,
        covariate_specs=covariate_specs,
        mode=covariate_interp_mode,
        deduplicate_excluding_interp_covariates=True,
        dedup_keep="first",
        attach_interp_meta=True,
        verbose=verbose,
    )


    if verbose:
        print(f"Original wishlist size: {len(wishlist)}")
        print(f"Expanded wishlist size: {len(expanded_wishlist)}")
        print(f"Control repeats per interpolated wish: {n_control_repeats}")

    all_results = {}
    all_extras = {}

    for wish_idx, wish in enumerate(expanded_wishlist):
        pert_tuple_str = _wish_to_pert_tuple_str(
            wish=wish,
            rulebook=rulebook,
            cond_name_key=cond_name_key,
        )

        matched_control = _get_matched_control(
            adata_control=adata_control,
            pert_tuple_str=pert_tuple_str,
            rulebook=rulebook,
            use_groupwise_control=use_groupwise_control,
        )

        if matched_control.n_obs == 0:
            if verbose:
                print(f"Skip wish {wish_idx}: matched control is empty.")
            continue

        for rep in range(int(n_control_repeats)):
            rep_seed = int(random_seed + 100000 * wish_idx + rep)

            sampled_control = _sample_adata_obs(
                matched_control,
                n_obs=control_sample_n_obs,
                frac=control_sample_frac,
                replace=control_replace,
                random_state=rep_seed,
            )

            rep_wish = copy.deepcopy(wish)
            rep_wish["_control_repeat"] = rep
            rep_wish["_control_seed"] = rep_seed
            rep_wish["_n_matched_control"] = int(matched_control.n_obs)
            rep_wish["_n_sampled_control"] = int(sampled_control.n_obs)

            raw_output = run_inference_only(
                model=model,
                adata_control=sampled_control,
                adata_test=adata_test,
                wishlist=[rep_wish],
                condition_keys=condition_keys,
                condition_rep_keys=condition_rep_keys,
                sample_rep=sample_rep,
                device=device,
                n_particles=n_particles,
                random_seed=rep_seed,
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
                inference_modes=inference_modes,
            )

            curr_results, curr_extras = _unpack_run_inference_only_output(raw_output)

            for cond_name, cond_result in curr_results.items():
                new_key = f"{cond_name}__ctrlrep{rep:03d}"

                cond_result = copy.deepcopy(cond_result)
                cond_result["wish"] = rep_wish
                cond_result["_control_sampling"] = {
                    "control_repeat": rep,
                    "control_seed": rep_seed,
                    "n_matched_control": int(matched_control.n_obs),
                    "n_sampled_control": int(sampled_control.n_obs),
                    "replace": bool(control_replace),
                }

                all_results[new_key] = cond_result

                if curr_extras is not None:
                    all_extras[new_key] = curr_extras

        if verbose and (wish_idx + 1) % 10 == 0:
            print(f"Finished {wish_idx + 1}/{len(expanded_wishlist)} interpolated wishes.")

    if verbose:
        print(f"Total resampled result entries: {len(all_results)}")

    if return_extras:
        return all_results, all_extras

    return all_results



# =============================================================================
# 4. True mass ratio calculation
# =============================================================================

    

def add_true_mass_ratio_to_df_groupwise_control(
    df,
    adata_treated,
    adata_control,
    rulebook,
    x_key,
    perturb_key="perturbation",
    cellline_key="cell_line",
    fixed_values=None,
    mass_deduct_keys=None,
    true_col="true_mass_ratio",
    treated_count_col="n_true_cells",
    control_count_col="n_ctrl_cells",
    cond_name_key="cond_name",
    use_groupwise_control=True,
    atol=1e-8,
    rtol=1e-5,
):
    """
    Add observed true mass ratio to df.

    true_mass_ratio = treated total mass / matched-control total mass

    treated matching:
        perturb_key + cellline_key + x_key + fixed_values.keys()

    control matching:
        1. matched control from _get_matched_control
        2. refined by cellline_key + fixed_values.keys()
        3. not filtered by x_key by default
    """
    out = df.copy()

    # Important: make fixed covariates available to row_dict,
    # so _row_to_pert_tuple_str can construct the full condition tuple.
    if fixed_values is not None:
        for k, v in fixed_values.items():
            if k not in out.columns:
                out[k] = v
            else:
                out[k] = out[k].fillna(v)

    m_source_val = float(adata_control.uns.get("normalized_m", 1.0))
    m_target_val = float(adata_treated.uns.get("normalized_m", 1.0))

    fixed_keys = list(fixed_values.keys()) if fixed_values is not None else []

    treated_match_keys = list(dict.fromkeys(
        [perturb_key, cellline_key, x_key] + fixed_keys
    ))
    control_refine_keys = list(dict.fromkeys(
        [cellline_key] + fixed_keys
    ))

    true_vals = []
    n_true_list = []
    n_ctrl_list = []

    for _, row in out.iterrows():
        row_dict = row.to_dict()

        treated_mask = _build_obs_mask_from_row(
            adata_treated,
            row_dict,
            treated_match_keys,
            atol=atol,
            rtol=rtol,
        )
        ad_true = adata_treated[treated_mask]

        pert_tuple_str = _row_to_pert_tuple_str(
            row_dict=row_dict,
            rulebook=rulebook,
            cond_name_key=cond_name_key,
        )

        curr_ctrl = _get_matched_control(
            adata_control=adata_control,
            pert_tuple_str=pert_tuple_str,
            rulebook=rulebook,
            use_groupwise_control=use_groupwise_control,
        )

        ctrl_mask = _build_obs_mask_from_row(
            curr_ctrl,
            row_dict,
            control_refine_keys,
            atol=atol,
            rtol=rtol,
        )
        ad_ctrl = curr_ctrl[ctrl_mask]

        n_true = int(ad_true.n_obs)
        n_ctrl = int(ad_ctrl.n_obs)

        n_true_list.append(n_true)
        n_ctrl_list.append(n_ctrl)

        if n_true == 0 or n_ctrl == 0:
            true_vals.append(np.nan)
            continue

        if (
            mass_deduct_keys is not None
            and mass_deduct_keys in ad_ctrl.obs.columns
            and mass_deduct_keys in ad_true.obs.columns
        ):
            n_unique_source = len(ad_ctrl.obs[mass_deduct_keys].unique())
            n_unique_target = len(ad_true.obs[mass_deduct_keys].unique())
            adj_m_target = m_target_val * max(1, n_unique_source) / max(1, n_unique_target)
        else:
            adj_m_target = m_target_val

        m_true_sum = ad_true.n_obs * adj_m_target
        m_ctrl_sum = ad_ctrl.n_obs * m_source_val

        true_mass_ratio = m_true_sum / m_ctrl_sum if m_ctrl_sum > 0 else np.nan
        true_vals.append(float(true_mass_ratio))

    out[true_col] = true_vals
    out[treated_count_col] = n_true_list
    out[control_count_col] = n_ctrl_list

    return out


# Backward-compatible alias, in case old notebooks still call the old function name.
def add_true_mass_change_to_df_groupwise_control(
    df,
    adata_treated,
    adata_control,
    rulebook,
    x_key,
    perturb_key="perturbation",
    cellline_key="cell_line",
    fixed_values=None,
    mass_deduct_keys=None,
    true_col="true_mass_ratio",
    treated_count_col="n_true_cells",
    control_count_col="n_ctrl_cells",
    cond_name_key="cond_name",
    use_groupwise_control=True,
    atol=1e-8,
    rtol=1e-5,
):
    """
    Backward-compatible wrapper.

    Note
    ----
    Despite the old name, this returns mass ratio, not mass change.
    """
    return add_true_mass_ratio_to_df_groupwise_control(
        df=df,
        adata_treated=adata_treated,
        adata_control=adata_control,
        rulebook=rulebook,
        x_key=x_key,
        perturb_key=perturb_key,
        cellline_key=cellline_key,
        fixed_values=fixed_values,
        mass_deduct_keys=mass_deduct_keys,
        true_col=true_col,
        treated_count_col=treated_count_col,
        control_count_col=control_count_col,
        cond_name_key=cond_name_key,
        use_groupwise_control=use_groupwise_control,
        atol=atol,
        rtol=rtol,
    )


# =============================================================================
# 5. Flatten inference results
# =============================================================================

def flatten_inference_results_to_df(
    results,
    mode: str = "all",
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    dose_key: str = "dose_value",
    time_key: str = "time",
):
    """
    Original flatten function, kept for compatibility.

    This returns one row per condition and treats predicted m as mass ratio.
    """
    rows = []

    for cond_name, cond_result in results.items():
        if mode not in cond_result:
            continue

        wish = cond_result["wish"]
        m_pred = np.asarray(cond_result[mode]["m_pred"]).reshape(-1)

        rows.append({
            "cond_name": cond_name,
            perturb_key: wish.get(perturb_key),
            cellline_key: wish.get(cellline_key),
            dose_key: wish.get(dose_key),
            time_key: wish.get(time_key),
            "mean_mass_ratio": float(m_pred.mean()),
            "median_mass_ratio": float(np.median(m_pred)),
            "n_cells": int(len(m_pred)),
        })

    df = pd.DataFrame(rows)

    for col in [dose_key, time_key]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def flatten_resampled_inference_results_to_df(
    results,
    mode: str = "all",
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    dose_key: str = "dose_value",
    time_key: str = "time",
    mass_summary: str = "mean",
):
    """
    Flatten resampled inference results.

    Each row corresponds to:
        condition × interpolated covariate value × control repeat

    Output y is named mass_ratio.
    """
    rows = []

    for cond_name, cond_result in results.items():
        if mode not in cond_result:
            continue

        wish = cond_result["wish"]
        m_pred = np.asarray(cond_result[mode]["m_pred"]).reshape(-1)

        if len(m_pred) == 0:
            continue

        if mass_summary == "mean":
            mass_ratio = float(np.mean(m_pred))
        elif mass_summary == "median":
            mass_ratio = float(np.median(m_pred))
        else:
            raise ValueError("mass_summary must be 'mean' or 'median'.")

        sampling_meta = cond_result.get("_control_sampling", {})

        rows.append({
            "cond_name": cond_name,
            perturb_key: wish.get(perturb_key),
            cellline_key: wish.get(cellline_key),
            dose_key: wish.get(dose_key),
            time_key: wish.get(time_key),

            "control_repeat": int(wish.get("_control_repeat", sampling_meta.get("control_repeat", -1))),
            "control_seed": int(wish.get("_control_seed", sampling_meta.get("control_seed", -1))),
            "n_matched_control": int(wish.get("_n_matched_control", sampling_meta.get("n_matched_control", -1))),
            "n_sampled_control": int(wish.get("_n_sampled_control", sampling_meta.get("n_sampled_control", -1))),

            "mass_ratio": mass_ratio,
            "n_particles": int(len(m_pred)),
        })

    df = pd.DataFrame(rows)

    for col in [dose_key, time_key, "mass_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# 6. Quantile summary
# =============================================================================

def summarize_resampled_mass_ratio_quantiles(
    df,
    x_key,
    y_key="mass_ratio",
    perturb_key="perturbation",
    cellline_key="cell_line",
    fixed_values=None,
    q_low=0.10,
    q_high=0.90,
):
    """
    Compute quantile band across control-resampling repeats.

    The output contains:
        mass_ratio_median
        mass_ratio_mean
        mass_ratio_low
        mass_ratio_high
    """
    plot_df = df.copy()

    if fixed_values is not None:
        for k, v in fixed_values.items():
            if k not in plot_df.columns:
                continue
            plot_df = plot_df[plot_df[k] == v]

    plot_df[x_key] = pd.to_numeric(plot_df[x_key], errors="coerce")
    plot_df[y_key] = pd.to_numeric(plot_df[y_key], errors="coerce")

    plot_df = plot_df.dropna(
        subset=[perturb_key, cellline_key, x_key, y_key]
    )

    if len(plot_df) == 0:
        return pd.DataFrame()

    summary = (
        plot_df
        .groupby([perturb_key, cellline_key, x_key], as_index=False)
        .agg(
            mass_ratio_median=(y_key, "median"),
            mass_ratio_mean=(y_key, "mean"),
            mass_ratio_low=(y_key, lambda x: np.quantile(x, q_low)),
            mass_ratio_high=(y_key, lambda x: np.quantile(x, q_high)),
            mass_ratio_sd=(y_key, "std"),
            n_repeats=("control_repeat", "nunique"),
            n_points=(y_key, "size"),
        )
        .sort_values([perturb_key, cellline_key, x_key])
    )

    summary["q_low"] = q_low
    summary["q_high"] = q_high

    return summary


# =============================================================================
# 7. Main plotting function: one perturbation × one cell line per figure
# =============================================================================

def plot_mass_ratio_quantile_band_by_x_single_cellline(
    summary_df,
    x_key,
    perturb_key="perturbation",
    cellline_key="cell_line",

    true_df=None,
    true_y_key="true_mass_ratio",

    y_center="mass_ratio_median",
    y_low="mass_ratio_low",
    y_high="mass_ratio_high",

    logx=True,
    logy=False,
    connect_points=True,

    figsize=(3.2, 2.6),
    band_alpha=0.22,
    line_width=2.0,
    marker_size=28,

    title_prefix=None,
    ylabel="Mass ratio",
    xlabel=None,

    baseline=1.0,
    show_baseline=True,

    show=True,
):
    """
    Plot one figure per perturbation × cell line.

    Recommended for Nature Methods style main figure panel:
        - center line: predicted median mass ratio
        - shaded band: control-resampling quantile band
        - observed points: true mass ratio
        - y-axis: linear by default
        - x-axis: log by default for dose
    """
    plot_df = summary_df.copy()
    plot_df[x_key] = pd.to_numeric(plot_df[x_key], errors="coerce")

    needed = [perturb_key, cellline_key, x_key, y_center, y_low, y_high]
    plot_df = plot_df.dropna(subset=needed)

    if len(plot_df) == 0:
        print("No data left after filtering.")
        return {}

    if xlabel is None:
        xlabel = x_key

    true_plot_df = None
    if true_df is not None and true_y_key in true_df.columns:
        true_plot_df = true_df.copy()
        true_plot_df[x_key] = pd.to_numeric(true_plot_df[x_key], errors="coerce")
        true_plot_df[true_y_key] = pd.to_numeric(true_plot_df[true_y_key], errors="coerce")
        true_plot_df = true_plot_df.dropna(
            subset=[perturb_key, cellline_key, x_key, true_y_key]
        )

    fig_dict = {}

    for (pert, cellline), cur in plot_df.groupby([perturb_key, cellline_key]):
        cur = cur.sort_values(x_key)

        fig, ax = plt.subplots(figsize=figsize)

        x = cur[x_key].to_numpy(dtype=float)
        yc = cur[y_center].to_numpy(dtype=float)
        yl = cur[y_low].to_numpy(dtype=float)
        yh = cur[y_high].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(yc) & np.isfinite(yl) & np.isfinite(yh)

        if logx:
            valid &= x > 0
        if logy:
            valid &= (yc > 0) & (yl > 0) & (yh > 0)

        x = x[valid]
        yc = yc[valid]
        yl = yl[valid]
        yh = yh[valid]

        if len(x) == 0:
            plt.close(fig)
            continue

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        if show_baseline:
            ax.axhline(
                baseline,
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
                zorder=0,
            )

        ax.fill_between(
            x,
            yl,
            yh,
            alpha=band_alpha,
            linewidth=0,
            label="Control-resampling band",
            zorder=1,
        )

        if connect_points and len(x) > 1:
            ax.plot(
                x,
                yc,
                linewidth=line_width,
                alpha=0.95,
                zorder=3,
                label="Prediction",
            )

        ax.scatter(
            x,
            yc,
            s=marker_size,
            alpha=0.95,
            zorder=4,
        )

        if true_plot_df is not None:
            true_cur = true_plot_df[
                (true_plot_df[perturb_key].astype(str) == str(pert)) &
                (true_plot_df[cellline_key].astype(str) == str(cellline))
            ].copy()

            if len(true_cur) > 0:
                true_agg = (
                    true_cur
                    .groupby(x_key, as_index=False)[true_y_key]
                    .mean()
                    .sort_values(x_key)
                )

                tx = true_agg[x_key].to_numpy(dtype=float)
                ty = true_agg[true_y_key].to_numpy(dtype=float)

                valid_true = np.isfinite(tx) & np.isfinite(ty)
                if logx:
                    valid_true &= tx > 0
                if logy:
                    valid_true &= ty > 0

                tx = tx[valid_true]
                ty = ty[valid_true]

                if len(tx) > 0:
                    ax.scatter(
                        tx,
                        ty,
                        s=marker_size * 1.35,
                        marker="x",
                        linewidths=1.7,
                        alpha=0.95,
                        zorder=5,
                        label="Observed",
                    )

        title = f"{pert} | {cellline}"
        if title_prefix is not None:
            title = f"{title_prefix}: {title}"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            frameon=False,
            fontsize=7,
            loc="best",
            handlelength=1.2,
        )

        plt.tight_layout()

        fig_dict[(pert, cellline)] = fig

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig_dict


# =============================================================================
# 8. Backward-compatible plotting alias
# =============================================================================

def plot_mean_mass_by_x(
    df,
    x_key,
    perturb_key="perturbation",
    cellline_key="cell_line",
    y_key="mass_ratio",
    true_y_key=None,
    fixed_values=None,
    logx=True,
    logy=False,
    connect_points=True,
    figsize=(3.2, 2.6),
    show=True,
):
    """
    Backward-compatible plotting wrapper.

    If df is replicate-level, this function first computes quantile summaries
    and then plots one perturbation × one cell line per figure.

    For the new main figure workflow, prefer calling:
        summarize_resampled_mass_ratio_quantiles(...)
        plot_mass_ratio_quantile_band_by_x_single_cellline(...)
    """
    summary_df = summarize_resampled_mass_ratio_quantiles(
        df=df,
        x_key=x_key,
        y_key=y_key,
        perturb_key=perturb_key,
        cellline_key=cellline_key,
        fixed_values=fixed_values,
        q_low=0.10,
        q_high=0.90,
    )

    true_df = None
    if true_y_key is not None and true_y_key in df.columns:
        true_df = df

    return plot_mass_ratio_quantile_band_by_x_single_cellline(
        summary_df=summary_df,
        x_key=x_key,
        perturb_key=perturb_key,
        cellline_key=cellline_key,
        true_df=true_df,
        true_y_key=true_y_key if true_y_key is not None else "true_mass_ratio",
        logx=logx,
        logy=logy,
        connect_points=connect_points,
        figsize=figsize,
        ylabel="Mass ratio",
        xlabel=x_key,
        show=show,
    )
