# Reproducibility Levels

## Level 1: Archived-output reproduction
`reproduce_bossbase04_manuscript.py` rebuilds the manuscript's original BOSSBase 0.4-bpp benchmark and verifies all 7,500 stego rows against the archived export. `regenerate_tables.py` then rebuilds the four condition summaries, Table 9a, the 201-point gamma sweep, Table 9b, the exact two-fold Mondrian diagnostic, and Fig. 3.

## Level 2: Model-integrity reproduction
`verify_checkpoints.py` checks SHA-256 hashes, strict state-dictionary compatibility, parameter counts, and a CPU forward pass for all ten checkpoints.

## Level 3: Image-to-output reproduction
The locked pipeline can be run on locally obtained BOSSBase/BOWS2 images. No BOWS2 recalibration is permitted.
