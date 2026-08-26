# HSNNET Ranked Attribution — Reproducibility Package

This repository supports the manuscript **Validation-Calibrated Detector Banks for Ranked Attribution of Spatial Image Steganographic Algorithms**.

The manuscript-to-repository numerical audit is summarized in `MANUSCRIPT_ALIGNMENT_REPORT.md`.

The framework is a second-stage forensic decision layer. It interprets the responses of five frozen algorithm-specific binary steganalyzers after an image has already been identified as stego or suspicious. It returns a complete ranking, a validation-calibrated candidate set, and an abstention state.

It is not a universal Cover/WOW/S-UNIWARD/HILL/HUGO/MiPOD closed-set classifier.

## Quick verification

```bash
python -m pip install -r environment/requirements-minimum.txt
python scripts/run_all_tests.py
```

The lightweight repository regenerates the four manuscript conditions, Table 9a, the 201-point gamma sweep, Table 9b, and Fig. 3 from released per-image outputs. The BOSSBase 0.4-bpp benchmark is rebuilt directly from its validation and test response matrices and checked row by row against the archived 7,500-image ranking export.

## Checkpoint verification

The ten retained checkpoint files are not embedded in the lightweight repository. Their filenames, sizes, and SHA-256 hashes are fixed in `manifests/checkpoint_inventory.csv`. After the checkpoint asset is attached to GitHub Release `v1.0.0`, download `HSNNET_CHECKPOINTS_V1.zip`, extract `checkpoints/` at the repository root, and run:

```bash
python scripts/verify_checkpoints.py
```

Until that Release asset is publicly attached, checkpoint-level reproduction remains pending; archived-output reproduction is available now.

## Locked operating point

- Candidate order: WOW, S-UNIWARD, HILL, HUGO, MiPOD
- Q10–Q90 cross-detector signatures
- gamma = 0.900
- Eight-transform D4 averaging
- Ranking before candidate filtering
- BOSSBase validation-only calibration
- No BOWS2 tuning

## Verified results

| Condition | Top-1 | Top-2 | Top-3 | Retention | Mean set |
|---|---:|---:|---:|---:|---:|
| BOSSBase 0.4 bpp | 42.813% | 65.853% | 82.280% | 90.400% | 3.429 |
| BOSSBase 0.2 bpp | 30.107% | 53.613% | 72.587% | 77.013% | 3.444 |
| BOWS2 0.4 bpp | 35.466% | 59.260% | 77.502% | 91.262% | 3.759 |
| BOWS2 0.2 bpp | 24.564% | 47.146% | 67.148% | 88.610% | 4.184 |

BOSSBase and BOWS2 images are not redistributed.

## Manuscript-benchmark provenance

Run the following command to reconstruct the complete BOSSBase 0.4-bpp ranking and candidate-set outputs from the released validation and held-out test response matrices:

```bash
python scripts/reproduce_bossbase04_manuscript.py
```

The command verifies all 7,500 stego rows against the archived per-image export. It fails if any complete ranking, candidate set, candidate-set size, or locked aggregate differs from the manuscript. A later candidate-ready rerun is retained only for provenance under `results/archive/bossbase_04_later_rerun/`; it is not used to reproduce the manuscript benchmark.

<!-- BEGIN REVIEWER_FIGURES_V1 -->
## Reviewer-facing figure reproduction

The repository includes archived inputs, regeneration scripts, PNG figures, PDF figures, and provenance metadata for the validation-signature heatmap and decision-layer diagnostics.

No dataset, checkpoint loading, GPU execution, model training, recalibration, or target-domain parameter selection is required for these figure-only commands:

```bash
python scripts/generate_q10_q90_heatmap.py
python scripts/generate_risk_coverage_curves.py
```

### Q10-Q90 validation-signature heatmap

Archived source:

- `results/core/bossbase_04/final_cross_template_signature_table.csv`

Reviewer-facing artifacts:

- `scripts/generate_q10_q90_heatmap.py`
- `figures/q10_q90_signature_interval_heatmap.png`
- `figures/q10_q90_signature_interval_heatmap.pdf`
- `figures/q10_q90_signature_interval_heatmap_metadata.json`

### Gamma-sweep and risk-coverage diagnostics

Archived sources:

- `results/risk_conformal/step6_verified/gamma_sweep_201_points.csv`
- `results/risk_conformal/step6_verified/table_9a_workload_conditioned_risk_coverage.csv`
- `results/risk_conformal/step6_verified/table_9b_proposed_vs_mondrian.csv`

Reviewer-facing artifacts:

- `scripts/generate_risk_coverage_curves.py`
- `figures/gamma_sweep_decision_coverage_false_exclusion.png`
- `figures/gamma_sweep_decision_coverage_false_exclusion.pdf`
- `figures/gamma_sweep_set_size_search_reduction.png`
- `figures/gamma_sweep_set_size_search_reduction.pdf`
- `figures/workload_conditioned_risk_coverage.png`
- `figures/workload_conditioned_risk_coverage.pdf`
- `figures/risk_coverage_curves_metadata.json`

These figures are descriptive decision-layer diagnostics. They do not establish finite-sample conformal coverage guarantees.
<!-- END REVIEWER_FIGURES_V1 -->
