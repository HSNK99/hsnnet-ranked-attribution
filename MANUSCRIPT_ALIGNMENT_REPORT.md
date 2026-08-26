# Manuscript Alignment Report

This package is aligned with the manuscript benchmark reported for BOSSBase 1.01 at 0.4 bpp.

## Locked manuscript results

| Metric | Reproduced value |
|---|---:|
| Stego samples | 7,500 |
| Ranked Top-1 | 42.8133% |
| Ranked Top-2 | 65.8533% |
| Ranked Top-3 | 82.2800% |
| Candidate retention | 90.4000% |
| Mean candidate-set size | 3.4285 |
| Expected search-space reduction | 31.4293% |
| Stego abstention | 2.0400% |
| Cover-like rejection | 52.1333% |
| Cover false attribution | 47.8667% |

## Evidence chain

`scripts/reproduce_bossbase04_manuscript.py` rebuilds Q10-Q90 signatures from the released validation matrix, scores the released held-out test matrix at the locked threshold gamma = 0.900, and verifies all 7,500 stego rankings and candidate sets against the archived per-image export. The comparison currently has zero mismatches.

The later BOSSBase 0.4-bpp rerun that produced Top-1 = 43.7467% is retained under `results/archive/bossbase_04_later_rerun/` for provenance only. It is not used by the manuscript reproduction path.

## Checkpoint release verification

The ten retained checkpoints are distributed as `HSNNET_CHECKPOINTS_V1.zip` through [GitHub Release `v1.0.0`](https://github.com/HSNK99/hsnnet-ranked-attribution/releases/tag/v1.0.0). The archive passed CRC validation (SHA-256 `037723ee08b31c344ff0d63011fc05d4adad807a6ce283c5add262a6122a68f7`), and `scripts/verify_checkpoints.py` verified the inventory hashes, architecture, parameter counts, and strict loading for all ten conditions (10/10 pass).
