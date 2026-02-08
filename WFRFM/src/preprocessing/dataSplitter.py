import numpy as np

def parse_any_pert(p: str):
    return [x for x in str(p).split('+') if x not in ("ctrl", "non-targeting", "")]

def canonical_combo(p: str):
    """把 combo 规范化：A+B 与 B+A 统一；single/ctrl 原样返回"""
    genes = parse_any_pert(p)
    if len(genes) <= 1:
        return str(p)
    return "+".join(sorted(genes))


def gene_pair_split(
    adata,
    pert_key: str,
    rng,
    test_ratio: float,
    combo_seen2_train_frac: float = 0.75,
    control_key: str | None = None,
    ctrl_token: str = "ctrl",
    drop_tokens=("non-targeting", ""),
    make_canonical_col: bool = True,
    canonical_suffix: str = "_canon",
    verbose: bool = True,
    return_details: bool = True,
):
    """
    A) gene-OOD 为主：先 split genes -> train_genes / ood_genes
    同时拥有：
      - unseen_single (ood+ctrl)
      - combo_seen0 (ood+ood)
      - combo_seen1 (ood+train)
      - combo_seen2 (train+train) 但 pair holdout (pair OOD)

    关键改动：默认会在 adata.obs 中创建 canonical pert 列（避免 A+B/B+A 导致 missing）
      - 使用 pert_key_canon 做 split 和切片
      - 你也可以设置 make_canonical_col=False 自己先处理

    返回：
      - train_perts, test_perts
      - （可选）details dict（包含 buckets、genes、pairs、missing、以及用于切片的列名）
      - 如果传了 control_key，会额外返回 adata_control/adata_train/adata_test
    """

    # 0) 选择用于 split 的 pert 列（可选自动 canonicalize）
    if make_canonical_col:
        pert_key_used = f"{pert_key}{canonical_suffix}"
        if pert_key_used not in adata.obs.columns:
            adata.obs[pert_key_used] = adata.obs[pert_key].astype(str).map(canonical_combo)
    else:
        pert_key_used = pert_key

    # 1) 全部 perts（canonical 表示）
    all_perts = adata.obs[pert_key_used].astype(str).unique().tolist()

    # 2) pert_list：非 ctrl 且解析后非空
    pert_list_raw = [p for p in all_perts if p != ctrl_token]
    pert_list = []
    for p in pert_list_raw:
        genes = [x for x in str(p).split('+') if x not in (ctrl_token, *drop_tokens)]
        if len(genes) == 0:
            continue
        # 注意：如果 make_canonical_col=True，这里理论上已经 canonical 了；再跑一次也无害
        pert_list.append(canonical_combo(p))
    pert_list = np.unique(pert_list).tolist()

    # 3) pert -> genes（tuple(sorted)）
    pert2genes = {p: tuple(sorted(parse_any_pert(p))) for p in pert_list}

    # 4) unique genes
    unique_pert_genes = np.unique([g for gs in pert2genes.values() for g in gs]).tolist()

    # 5) gene split
    train_gene_set_size = 1.0 - test_ratio
    n_train_genes = max(1, int(len(unique_pert_genes) * train_gene_set_size))
    train_gene_candidates = rng.choice(unique_pert_genes, size=n_train_genes, replace=False).tolist()
    train_gene_set = set(train_gene_candidates)
    ood_genes = [g for g in unique_pert_genes if g not in train_gene_set]

    # 6) buckets
    pert_single_train = []
    unseen_single = []
    combo_seen0, combo_seen1, combo_traintrain_pool = [], [], []

    for p, gs in pert2genes.items():
        if len(gs) == 1:
            (g,) = gs
            if g in train_gene_set:
                pert_single_train.append(p)
            else:
                unseen_single.append(p)
        elif len(gs) >= 2:
            k = len(set(gs) & train_gene_set)
            if k == 0:
                combo_seen0.append(p)
            elif k == 1:
                combo_seen1.append(p)
            else:
                combo_traintrain_pool.append(p)

    # 7) seen2 pair holdout（对 train+train combos 按 pair key 抽训练 pair）
    pair2perts = {}
    for p in combo_traintrain_pool:
        key = pert2genes[p]  # tuple(sorted)
        pair2perts.setdefault(key, []).append(p)

    all_pairs = list(pair2perts.keys())
    n_train_pairs = int(len(all_pairs) * combo_seen2_train_frac)

    if len(all_pairs) > 0 and n_train_pairs > 0:
        idx = rng.choice(np.arange(len(all_pairs)), size=n_train_pairs, replace=False)
        train_pairs = {all_pairs[i] for i in idx}
    else:
        train_pairs = set()

    pert_combo_train = []
    combo_seen2 = []
    for pair, perts in pair2perts.items():
        if pair in train_pairs:
            pert_combo_train.extend(perts)
        else:
            combo_seen2.extend(perts)

    # 8) train/test perts
    train_perts = np.unique(pert_single_train + pert_combo_train).tolist()
    test_perts = np.unique(combo_seen0 + combo_seen1 + combo_seen2 + unseen_single).tolist()

    # 9) checks
    train_set, test_set = set(train_perts), set(test_perts)
    assert train_set.isdisjoint(test_set), "train/test perts overlap!"

    # seen2 pair leakage check
    train_pairs_seen = set()
    for p in train_set:
        if p == ctrl_token or p not in pert2genes:
            continue
        gs = pert2genes[p]
        if len(gs) >= 2 and all(g in train_gene_set for g in gs):
            train_pairs_seen.add(gs)

    test_seen2_pairs = set(pert2genes[p] for p in combo_seen2)
    assert len(train_pairs_seen & test_seen2_pairs) == 0, "seen2 pair leakage: some test pairs appear in train!"

    # coverage（在 canonical 空间里检查）
    covered = train_set | test_set
    missing = set(all_perts) - covered

    if verbose:
        print("pert_key_used:", pert_key_used)
        print("train genes:", len(train_gene_candidates), "ood genes:", len(ood_genes))
        print("combo_seen0:", len(combo_seen0), "combo_seen1:", len(combo_seen1),
              "combo_seen2:", len(combo_seen2), "unseen_single:", len(unseen_single))
        print("train_perts:", len(train_perts), "test_perts:", len(test_perts))
        if len(missing) > 0:
            sample = list(missing)[:10]
            print(f"missing perts (canonical): {len(missing)} (show up to 10) -> {sample}")

    # 10) 需要的话，直接给切好的 adata
    adata_control = adata_train = adata_test = None
    if control_key is not None:
        adata_control = adata[adata.obs[control_key] == True].copy()
        adata_train = adata[adata.obs[pert_key_used].astype(str).isin(train_perts)].copy()
        adata_test = adata[adata.obs[pert_key_used].astype(str).isin(test_perts)].copy()

    if not return_details:
        if control_key is None:
            return train_perts, test_perts, pert_key_used
        return train_perts, test_perts, pert_key_used, adata_control, adata_train, adata_test

    details = dict(
        pert_key_used=pert_key_used,
        pert_list=pert_list,
        pert2genes=pert2genes,
        unique_pert_genes=unique_pert_genes,
        train_genes=train_gene_candidates,
        ood_genes=ood_genes,
        buckets=dict(
            pert_single_train=pert_single_train,
            unseen_single=unseen_single,
            combo_seen0=combo_seen0,
            combo_seen1=combo_seen1,
            combo_seen2=combo_seen2,
            pert_combo_train=pert_combo_train,
        ),
        pairs=dict(
            all_pairs=all_pairs,
            train_pairs=list(train_pairs),
        ),
        missing=missing,
    )

    if control_key is None:
        return train_perts, test_perts, details
    return train_perts, test_perts, details, adata_control, adata_train, adata_test
