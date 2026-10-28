"""Map the produced artefacts to the 34 required research outputs.

Reads only files the pipeline wrote under ``--out`` and emits
``34_OUTPUT_REPORT.csv`` / ``34_OUTPUT_REPORT.json``. A task is COMPUTED only
when every artefact it requires is present; otherwise it is PARTIAL or MISSING
with the missing file names. Nothing is inferred or fabricated.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# (task id, rq, theme, [required artefacts relative to the output root])
TASKS = [
    ("RQ01-01", "RQ01", "P4 four-class state model (N/D/G/BOTH)",
     ["RQ01/RQ01_DEV_CONFUSION4_M00.csv", "RQ01/RQ01_DEV_CONFUSION4_M11.csv", "RQ01/RQ01_OOS_CONFUSION4.csv"]),
    ("RQ01-02", "RQ01", "Individual short votes", ["RQ01/RQ01_DEV_INDIVIDUAL.csv"]),
    ("RQ01-03", "RQ01", "Joint long/short vote mechanism", ["RQ01/RQ01_DEV_JOINT_PAIRS.csv"]),
    ("RQ01-04", "RQ01", "Pool vote mechanisms", ["RQ01/RQ01_DEV_POOLS.csv"]),
    ("RQ01-05", "RQ01", "Continuous-score binary classifiers",
     ["RQ01/RQ01_DEV_FOUR_M_METRICS.csv"]),
    ("RQ01-06", "RQ01", "F1 fitting diagnostic", ["RQ01/RQ01_DEV_F1_F5.csv"]),
    ("RQ01-07", "RQ01", "F2 cluster-aware uncertainty gate", ["RQ01/RQ01_DEV_F1_F5.csv"]),
    ("RQ01-08", "RQ01", "F3 penalty benchmark / solver diagnostic", ["RQ01/RQ01_DEV_F1_F5.csv"]),
    ("RQ01-09", "RQ01", "Official F1-F5 certification", ["RQ01/RQ01_DEV_F1_F5.csv"]),
    ("RQ01-10", "RQ01", "Owner certification sheet / F4-F5", ["RQ01/RQ01_DEV_F1_F5.csv"]),
    ("RQ01-11", "RQ01", "Independent classifiers", ["RQ01/RQ01_DEV_IJP_PRE.csv"]),
    ("RQ01-12", "RQ01", "Final per-country threshold selection",
     ["RQ01/RQ01_DEV_THRESHOLD_REGISTRY.json", "RQ01/RQ01_DEV_COUNTRY_STRESS.csv"]),

    ("RQ02-01", "RQ02", "Onset heads across horizons", ["RQ02/RQ02_DEV_ONSET_METRICS.csv"]),
    ("RQ02-02", "RQ02", "ENTRY prediction",
     ["RQ02/RQ02_DEV_ONSET_METRICS.csv", "RQ02/RQ02_OOS_ONSET_METRICS.csv"]),
    ("RQ02-03", "RQ02", "EXIT-A vs EXIT-B",
     ["RQ02/RQ02_DEV_EXIT_AB.csv", "RQ02/RQ02_OOS_EXIT_AB.csv"]),
    ("RQ02-04", "RQ02", "Selected EXIT configuration", ["RQ02/RQ02_DEV_FUTURE_STATE_METRICS.csv"]),
    ("RQ02-05", "RQ02", "EXIT without eligible negatives", ["RQ02/RQ02_OOS_EXIT_AB.csv"]),
    ("RQ02-06", "RQ02", "Pn/Ps source coverage",
     ["RQ02/RQ02_DEV_SOURCE_MATRIX.csv", "RQ02/RQ02_OOS_SOURCE_MATRIX.csv"]),
    ("RQ02-07", "RQ02", "EXIT horizon sensitivity", ["RQ02/RQ02_DEV_EXIT_AB.csv"]),
    ("RQ02-08", "RQ02", "Bottom zone prediction",
     ["RQ02/RQ02_DEV_BOTTOM_ZONES.csv", "RQ02/RQ02_OOS_BOTTOM.csv"]),
    ("RQ02-09", "RQ02", "Bottom scoring factors", ["RQ02/RQ02_DEV_BOTTOM_TROUGHS.csv"]),
    ("RQ02-10", "RQ02", "Pre-onset signal accumulation", ["RQ02/RQ02_DEV_LONG_SHORT_MATRICES.csv"]),
    ("RQ02-11", "RQ02", "Bottom evidence on OOS", ["RQ02/RQ02_OOS_BOTTOM.csv"]),

    ("RQ03-01", "RQ03", "Volatility model (GARCH)", ["RQ03/RQ03_DEV_RISK_MODEL.json"]),
    ("RQ03-02", "RQ03", "Covariance model (DCC)", ["RQ03/RQ03_DEV_RISK_MODEL.json"]),
    ("RQ03-03", "RQ03", "Portfolio ledger and terminal wealth", ["RQ03/RQ03_DEV_NAV_NET.csv"]),
    ("RQ03-04", "RQ03", "VaR / ES evaluation", ["RQ03/RQ03_DEV_VAR_ES_72.csv"]),
    ("RQ03-05", "RQ03", "Allocation path by state", ["RQ03/RQ03_DEV_WEIGHTS.csv"]),
    ("RQ03-06", "RQ03", "Main model vs five benchmarks",
     ["RQ03/RQ03_DEV_PERFORMANCE.csv", "RQ03/RQ03_OOS_PERFORMANCE.csv"]),
    ("RQ03-07", "RQ03", "Dynamic restructuring and feasibility",
     ["RQ03/RQ03_DEV_ROBUSTNESS.csv", "RQ03/RQ03_DEV_CAPITAL_FEASIBILITY.csv"]),
    ("RQ03-08", "RQ03", "Trading lock counterfactual", ["RQ03/RQ03_DEV_PERMISSION.csv"]),
    ("RQ03-09", "RQ03", "Policy options comparison", ["RQ03/RQ03_DEV_ROBUSTNESS.csv"]),
    ("RQ03-10", "RQ03", "Risk constraints and permission gates", ["RQ03/RQ03_DEV_WEIGHTS.csv"]),
    ("RQ03-11", "RQ03", "Short-side / bottom overlay", ["RQ03/RQ03_DEV_SHORT_BOTTOM.csv"]),
]


def build(out_root: str | Path) -> dict:
    root = Path(out_root)
    rows = []
    for tid, rq, theme, needs in TASKS:
        present = [n for n in needs if (root / n).is_file()]
        missing = [n for n in needs if not (root / n).is_file()]
        status = "COMPUTED" if not missing else ("PARTIAL" if present else "MISSING")
        rows.append({
            "task": tid, "rq": rq, "theme": theme, "status": status,
            "evidence_present": len(present), "evidence_required": len(needs),
            "missing": "; ".join(missing),
        })

    summary = {
        "generated_from": str(root),
        "n_tasks": len(rows),
        "computed": sum(r["status"] == "COMPUTED" for r in rows),
        "partial": sum(r["status"] == "PARTIAL" for r in rows),
        "missing": sum(r["status"] == "MISSING" for r in rows),
        "tasks": rows,
    }

    (root / "34_OUTPUT_REPORT.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "34_OUTPUT_REPORT.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary
