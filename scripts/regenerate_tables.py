from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repro_utils import (
    ROOT, EXPECTED_RESULTS, condition_check, compute_condition_metrics,
    exact_mondrian, locate_prediction, make_table8a, prepare_bows2_04,
    read_table, set_metrics
)

EXPECTED_TABLE9A_COUNTS = [3231,7635,15173,31793,49153]
EXPECTED_MONDRIAN = {
    "retention_percent":89.978,
    "mean_set_size":3.677,
    "search_reduction_percent":26.461,
    "empty_percent":3.472,
    "decision_coverage_percent":96.528,
    "conditional_false_exclusion_percent":6.786,
}

def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_rows, verification_rows = [], []
    all_pass = True

    for condition in EXPECTED_RESULTS:
        source = locate_prediction(condition)
        observed = compute_condition_metrics(read_table(source))
        passed, rows = condition_check(condition, observed)
        all_pass = all_pass and passed
        verification_rows.extend(rows)
        condition_rows.append({
            "condition":condition,
            "source":str(source.relative_to(ROOT)),
            **observed,
            "status":"PASS" if passed else "FAIL"
        })

    pd.DataFrame(condition_rows).to_csv(
        output_dir/"condition_summary_regenerated.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(verification_rows).to_csv(
        output_dir/"condition_verification.csv",
        index=False, encoding="utf-8-sig"
    )

    source, frame, labels, true_indices, scores = prepare_bows2_04()
    locked_metrics = set_metrics(scores >= .900, true_indices)

    table9a = make_table8a(scores, true_indices, .900)
    table9a["expected_accepted_n"] = EXPECTED_TABLE9A_COUNTS
    table9a["count_status"] = np.where(
        table9a["accepted_n"].to_numpy()==np.array(EXPECTED_TABLE9A_COUNTS),
        "PASS","FAIL"
    )
    table9a.to_csv(
        output_dir/"table_9a_workload_conditioned_risk_coverage.csv",
        index=False, encoding="utf-8-sig"
    )

    gamma_rows = []
    for gamma in np.round(np.arange(0,1.0001,.005),3):
        m = set_metrics(scores >= gamma, true_indices)
        gamma_rows.append({
            "gamma":float(gamma),
            **{k:v for k,v in m.items() if k not in {"set_sizes","contains_true"}}
        })
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(
        output_dir/"gamma_sweep_201_points.csv",
        index=False, encoding="utf-8-sig"
    )

    mondrian, sets, folds, thresholds = exact_mondrian(
        labels,true_indices,scores,.10,42
    )
    mondrian_checks = []
    for metric, expected in EXPECTED_MONDRIAN.items():
        observed = float(mondrian[metric])
        passed = round(observed,3)==round(expected,3)
        mondrian_checks.append({
            "metric":metric,"observed":observed,"expected":expected,
            "status":"PASS" if passed else "FAIL"
        })

    table9b = pd.DataFrame([
        {
            "method":"Proposed, frozen gamma=0.900",
            **{k:v for k,v in locked_metrics.items() if k not in {"set_sizes","contains_true"}},
            "calibration_source":"BOSSBase validation only; frozen on BOWS2"
        },
        {
            "method":"Two-fold Mondrian conformal, alpha=0.10",
            **{k:v for k,v in mondrian.items() if k not in {"set_sizes","contains_true"}},
            "calibration_source":"Cross-fitted BOWS2 labels; shared random.Random(42)"
        }
    ])
    table9b.to_csv(
        output_dir/"table_9b_proposed_vs_mondrian.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(mondrian_checks).to_csv(
        output_dir/"mondrian_verification.csv",
        index=False, encoding="utf-8-sig"
    )
    thresholds.to_csv(
        output_dir/"mondrian_class_thresholds.csv",
        index=False, encoding="utf-8-sig"
    )

    per_image = frame[["sample_id","image_name","true_algo"]].copy()
    per_image["mondrian_fold"] = folds
    per_image["mondrian_set_size"] = mondrian["set_sizes"]
    per_image["mondrian_contains_true"] = mondrian["contains_true"].astype(int)
    per_image["mondrian_candidate_set"] = [
        "|".join(a for a,inc in zip(
            ["WOW","S-UNIWARD","HILL","HUGO","MiPOD"],row
        ) if inc)
        for row in sets
    ]
    per_image.to_csv(
        output_dir/"mondrian_per_image_regenerated.csv.gz",
        index=False, compression="gzip", encoding="utf-8"
    )

    plt.figure(figsize=(7.2,5.2))
    plt.plot(
        table9a["coverage_percent"],
        table9a["conditional_false_exclusion_percent"],
        marker="o", label="Proposed workload budgets"
    )
    for _, row in table9a.iterrows():
        plt.annotate(
            f"k={int(row['max_set_k'])}",
            (row["coverage_percent"],row["conditional_false_exclusion_percent"]),
            xytext=(5,5), textcoords="offset points"
        )
    plt.scatter(
        [mondrian["decision_coverage_percent"]],
        [mondrian["conditional_false_exclusion_percent"]],
        marker="s", label="Mondrian conformal, alpha=0.10"
    )
    plt.xlabel("Decision coverage (%)")
    plt.ylabel("Conditional false-exclusion risk (%)")
    plt.title("Workload-conditioned coverage-risk diagnostic")
    plt.grid(True,alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir/"fig3_workload_coverage_false_exclusion.png",dpi=300,bbox_inches="tight")
    plt.savefig(output_dir/"fig3_workload_coverage_false_exclusion.pdf",bbox_inches="tight")
    plt.close()

    table9a_pass = table9a["count_status"].eq("PASS").all()
    mondrian_pass = all(x["status"]=="PASS" for x in mondrian_checks)
    overall = all_pass and table9a_pass and mondrian_pass and len(gamma_df)==201

    summary = {
        "status":"PASS" if overall else "FAIL",
        "source_bows2_04":str(source.relative_to(ROOT)),
        "four_conditions_pass":bool(all_pass),
        "table9a_counts_pass":bool(table9a_pass),
        "gamma_points":int(len(gamma_df)),
        "mondrian_pass":bool(mondrian_pass),
    }
    (output_dir/"regeneration_summary.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8"
    )

    lines = ["HSNNET TABLE REGENERATION REPORT","="*76]
    for row in condition_rows:
        lines.append(
            f"{row['status']:4s} | {row['condition']:15s} | "
            f"n={row['stego_n']:,} | Top-1={row['top1']:.3f}% | "
            f"retention={row['retention']:.3f}%"
        )
    lines += ["","TABLE 9a","-"*76]
    for _,row in table9a.iterrows():
        lines.append(
            f"{row['count_status']:4s} | k={int(row['max_set_k'])} | "
            f"accepted={int(row['accepted_n']):,}"
        )
    lines += [
        "","TABLE 9b — MONDRIAN","-"*76,
        f"Retention                   : {mondrian['retention_percent']:.3f}%",
        f"Mean set                    : {mondrian['mean_set_size']:.3f}/5",
        f"Search reduction            : {mondrian['search_reduction_percent']:.3f}%",
        f"Empty                       : {mondrian['empty_percent']:.3f}%",
        f"Decision coverage           : {mondrian['decision_coverage_percent']:.3f}%",
        f"Conditional false-exclusion : {mondrian['conditional_false_exclusion_percent']:.3f}%",
        "",
        "FINAL STATUS: " + (
            "PASS — ALL ARCHIVED TABLES WERE REGENERATED."
            if overall else
            "FAIL — REGENERATION MISMATCH DETECTED."
        )
    ]
    report = "\n".join(lines)
    (output_dir/"REGENERATION_REPORT.txt").write_text(report,encoding="utf-8")
    print(report)
    if not overall:
        raise RuntimeError("Table regeneration failed.")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",default=str(ROOT/"generated"))
    args = parser.parse_args()
    run(Path(args.output))

if __name__=="__main__":
    main()
