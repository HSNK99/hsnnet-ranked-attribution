# Expected Outputs

- `verify_release.py`: `FINAL STATUS: PASS — ARCHIVED RESULTS REPRODUCED.`
- `verify_splits.py`: `FINAL STATUS: PASS — PAIR-IDENTITY SPLIT IS LEAKAGE-SAFE.`
- `verify_checkpoints.py`: `FINAL STATUS: PASS — ALL TEN CHECKPOINTS STRICTLY VERIFIED.`
- `regenerate_tables.py`: `FINAL STATUS: PASS — ALL ARCHIVED TABLES WERE REGENERATED.`

<!-- BEGIN REVIEWER_FIGURES_V1 -->
## Reviewer-facing figure regeneration

Expected successful status messages:

- `generate_q10_q90_heatmap.py`: `FINAL STATUS: PASS - THE REAL 5x5 Q10-Q90 SIGNATURE HEATMAP WAS GENERATED.`
- `generate_risk_coverage_curves.py`: `FINAL STATUS: PASS - THREE LOCKED RISK-COVERAGE AND GAMMA-SWEEP FIGURES WERE GENERATED.`

Expected output files:

- `figures/q10_q90_signature_interval_heatmap.png`
- `figures/q10_q90_signature_interval_heatmap.pdf`
- `figures/q10_q90_signature_interval_heatmap_metadata.json`
- `figures/gamma_sweep_decision_coverage_false_exclusion.png`
- `figures/gamma_sweep_decision_coverage_false_exclusion.pdf`
- `figures/gamma_sweep_set_size_search_reduction.png`
- `figures/gamma_sweep_set_size_search_reduction.pdf`
- `figures/workload_conditioned_risk_coverage.png`
- `figures/workload_conditioned_risk_coverage.pdf`
- `figures/risk_coverage_curves_metadata.json`
<!-- END REVIEWER_FIGURES_V1 -->
