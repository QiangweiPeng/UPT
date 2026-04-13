

python experiments/zesta/infer_zebrafish_real_source.py \
    --run-dir results/checkpoints/zesta/zesta_independent_real_time_all_control_X_pca_scaled_delta10_regm1_seed42 \
    --condition-mode full

python experiments/zesta/infer_zebrafish_real_source.py \
    --run-dir results/checkpoints/zesta/zesta_independent_real_time_all_control_X_pca_scaled_delta10_regm1_seed42 \
    --condition-mode growth-only
