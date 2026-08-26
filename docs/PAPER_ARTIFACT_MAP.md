# Paper-to-Artifact Map

| Manuscript evidence | Repository artifact |
|---|---|
| Detector architecture | `src/model_definition_hsnnet_locked.py` |
| Ten checkpoints | Release asset + `manifests/checkpoint_inventory.csv` |
| Pair split | `splits/` + `generated/split_leakage_audit.csv` |
| Four main conditions | `results/core/` |
| BOSSBase 0.4-bpp validation response matrix | `results/core/bossbase_04/validation_response_matrix_z.csv` |
| BOSSBase 0.4-bpp held-out test response matrix | `results/core/bossbase_04/test_response_matrix_z.csv` |
| BOSSBase 0.4-bpp per-image rankings and candidate sets | `results/core/bossbase_04/final_predictions_all.csv` |
| Validation-cover normalization statistics | `results/core/bossbase_04/cover_normalization_statistics.json` |
| Q10-Q90 response signatures | `results/core/bossbase_04/q10_q90_response_signatures.json` |
| Table 9 | `results/core/bossbase_04/table_9_candidate_size_conditioned.csv` |
| Baselines | `configs/baselines_locked.json`, `results/baselines/` |
| Multi-seed | `results/multiseed/` |
| Table 9a | `generated/table_9a_workload_conditioned_risk_coverage.csv` |
| Table 9b | `generated/table_9b_proposed_vs_mondrian.csv` |
| Gamma sweep | `generated/gamma_sweep_201_points.csv` |
| Fig. 3 | `generated/fig3_workload_coverage_false_exclusion.*` |

<!-- BEGIN REVIEWER_FIGURES_V1 -->
## Reviewer-facing figures

| Evidence | Archived source | Regeneration script | Output |
|---|---|---|---|
| Q10-Q90 validation-signature heatmap | `results/core/bossbase_04/final_cross_template_signature_table.csv` | `scripts/generate_q10_q90_heatmap.py` | `figures/q10_q90_signature_interval_heatmap.png`, `figures/q10_q90_signature_interval_heatmap.pdf`, `figures/q10_q90_signature_interval_heatmap_metadata.json` |
| Gamma-sweep coverage and conditional risk | `results/risk_conformal/step6_verified/gamma_sweep_201_points.csv` | `scripts/generate_risk_coverage_curves.py` | `figures/gamma_sweep_decision_coverage_false_exclusion.png`, `figures/gamma_sweep_decision_coverage_false_exclusion.pdf` |
| Gamma-sweep workload and search reduction | `results/risk_conformal/step6_verified/gamma_sweep_201_points.csv` | `scripts/generate_risk_coverage_curves.py` | `figures/gamma_sweep_set_size_search_reduction.png`, `figures/gamma_sweep_set_size_search_reduction.pdf` |
| Workload-conditioned risk-coverage diagnostic | `results/risk_conformal/step6_verified/table_9a_workload_conditioned_risk_coverage.csv`, `results/risk_conformal/step6_verified/table_9b_proposed_vs_mondrian.csv` | `scripts/generate_risk_coverage_curves.py` | `figures/workload_conditioned_risk_coverage.png`, `figures/workload_conditioned_risk_coverage.pdf`, `figures/risk_coverage_curves_metadata.json` |
<!-- END REVIEWER_FIGURES_V1 -->
