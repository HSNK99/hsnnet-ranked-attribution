from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run(script):
    process=subprocess.run(
        [sys.executable,str(ROOT/"scripts"/script)],
        cwd=str(ROOT)
    )
    if process.returncode!=0:
        raise SystemExit(process.returncode)

def main():
    run("reproduce_bossbase04_manuscript.py")
    run("regenerate_tables.py")
    run("verify_release.py")
    run("verify_splits.py")
    checkpoints=list((ROOT/"checkpoints").glob("*.pth"))
    if len(checkpoints)==10:
        run("verify_checkpoints.py")
    else:
        print("SKIP | verify_checkpoints.py | Extract the checkpoint Release asset first.")
    print("\nFINAL STATUS: PASS — ALL AVAILABLE REPOSITORY TESTS PASSED.")

if __name__=="__main__":
    main()
