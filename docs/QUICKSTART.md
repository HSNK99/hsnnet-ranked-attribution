# Quick Start

## Level 1 — Archived-result reproduction

No dataset and no GPU are required.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r environment\requirements-minimum.txt
python scripts\run_all_tests.py
```

Expected final line:

```text
FINAL STATUS: PASS — ALL AVAILABLE REPOSITORY TESTS PASSED.
```

## Level 2 — Checkpoint integrity

Download `HSNNET_CHECKPOINTS_V1.zip` from [GitHub Release `v1.0.0`](https://github.com/HSNK99/hsnnet-ranked-attribution/releases/tag/v1.0.0), verify SHA-256 `037723ee08b31c344ff0d63011fc05d4adad807a6ce283c5add262a6122a68f7`, extract its `checkpoints/` directory at the repository root, and run:

```bat
python scripts\verify_checkpoints.py
```

The release archive passed CRC validation, and all ten checkpoints passed the locked SHA-256 inventory and strict-load audit.

## Level 3 — Full inference

Obtain BOSSBase 1.01 and BOWS2 separately. Copy `configs/paths.example.json` to `configs/paths.local.json`, enter local paths, and run the locked candidate pipeline or notebook. Dataset images are intentionally not bundled.

<!-- BEGIN REVIEWER_FIGURES_V1 -->
## Level 1 - Reviewer-facing figure reproduction

No dataset, GPU, detector inference, training, or recalibration is required.

From the repository root, run:

```bash
python scripts/generate_q10_q90_heatmap.py
python scripts/generate_risk_coverage_curves.py
```

The first command validates the complete BOSSBase 0.4 bpp 5x5 target-algorithm/detector-dimension grid and regenerates the Q10-Q90 validation-signature heatmap.

The second command validates the 201-point gamma sweep, workload thresholds k=1 through k=5, and the archived Proposed-Mondrian comparison before regenerating the three decision-layer diagnostic figures.

Generated PNG, PDF, and metadata files are written to `figures/`.
<!-- END REVIEWER_FIGURES_V1 -->
