from .adamson import (
    DEFAULT_ADAMSON_CONTROL_CONDITION,
    build_adamson_condition_rep_dict,
    load_adamson_gene_rep_dict,
    load_adamson_split_conditions,
    load_adamson_training_data,
)


DEFAULT_REPLOGLE_CONTROL_CONDITION = DEFAULT_ADAMSON_CONTROL_CONDITION


def load_replogle_gene_rep_dict(condition_rep_path):
    """Load Replogle gene embeddings from CSV or pickle."""
    return load_adamson_gene_rep_dict(condition_rep_path)


def load_replogle_split_conditions(split_path):
    """Load the Replogle train/val/test condition split definition."""
    return load_adamson_split_conditions(split_path)


def build_replogle_condition_rep_dict(
    condition_names,
    go_path=None,
    gene_rep_dict=None,
    control_condition=DEFAULT_REPLOGLE_CONTROL_CONDITION,
    normalize=True,
):
    """Build Replogle condition embeddings from gene embeddings or GO features."""
    return build_adamson_condition_rep_dict(
        condition_names=condition_names,
        go_path=go_path,
        gene_rep_dict=gene_rep_dict,
        control_condition=control_condition,
        normalize=normalize,
    )


def load_replogle_training_data(
    data_path,
    split_path,
    go_path=None,
    condition_rep_path=None,
    condition_key="condition",
    control_key="control",
    control_condition=DEFAULT_REPLOGLE_CONTROL_CONDITION,
    control_max_cells=None,
    treated_max_cells_per_condition=None,
    max_train_conditions=None,
    max_val_conditions=None,
    max_test_conditions=None,
    materialize_test=False,
    extra_obs_columns=None,
    seed=42,
):
    """Load lightweight Replogle AnnData subsets for training and inference."""
    return load_adamson_training_data(
        data_path=data_path,
        split_path=split_path,
        go_path=go_path,
        condition_rep_path=condition_rep_path,
        condition_key=condition_key,
        control_key=control_key,
        control_condition=control_condition,
        control_max_cells=control_max_cells,
        treated_max_cells_per_condition=treated_max_cells_per_condition,
        max_train_conditions=max_train_conditions,
        max_val_conditions=max_val_conditions,
        max_test_conditions=max_test_conditions,
        materialize_test=materialize_test,
        extra_obs_columns=extra_obs_columns,
        seed=seed,
    )
