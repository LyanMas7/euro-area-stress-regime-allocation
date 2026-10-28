"""rev363 corrected orchestrator: one deterministic command for the whole V8 repair run.

    python run_all.py --dataset-root <locked data> --out <release dir>

Produces RAW RQ01/RQ02/RQ03 corrected outputs (DEV, OOS full, OOS ESRB-scorable,
OOS no-ESRB prediction-only, continuous NAV + tail table) plus the 34 output
report cards, acceptance CSV and run manifest.
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rev363 import rq01_next as R1
from rev363 import rq02_next as R2
from rev363 import rq03_next as R3
from rev363 import estimators as EST
from rev363.report import build_reports
from rev363.util import jdump, run_id, sha256_file, utcnow


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fast", action="store_true", help="legacy setting; does not bypass contract")
    p.add_argument("--oos-authorization", default=None,
                   help="explicit DEC-015 user token; default remains fail-closed")
    p.add_argument("--w01-dev-audit", action="store_true",
                   help="run only lawful W01 full-DEV source, label and math audits")
    a = p.parse_args(argv)

    if a.w01_dev_audit:
        from rev363.w01_dev_audit import execute as audit_w01
        return audit_w01(a.dataset_root, Path(a.out) / "RQ01_DEV_EVIDENCE")

    # Do not construct OOS output directories or run any code before hard gate.
    from rev363.w01_contract import enforce_execution
    enforce_execution(oos_requested=True, oos_authorization=a.oos_authorization)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rq01_dir = out / "RQ01_RESULTS"; rq01_dir.mkdir(parents=True, exist_ok=True)
    rq02_dir = out / "RQ02_RESULTS"; rq02_dir.mkdir(parents=True, exist_ok=True)
    rq03_dir = out / "RQ03_RESULTS"; rq03_dir.mkdir(parents=True, exist_ok=True)

    print("[rev363] RQ01 ...")
    r1 = R1.run(a.dataset_root, rq01_dir, budget_fast=a.fast,
                oos_authorization=a.oos_authorization)
    panel_prob = r1.pop("_panel")

    print("[rev363] RQ02 ...")
    tape10 = pd.read_csv(rq01_dir / "RQ01_10_ISSUED_TAPE_OOS.csv")
    rq01_tape = pd.DataFrame({
        "country": tape10["country"], "issue": tape10["issue"],
        "pred_stress": (tape10["dec_LOGISTIC_M11"] != "N").astype(int),
        "score": 1.0 - tape10["p_LOGISTIC_M11_N"],
    })
    r2 = R2.run(a.dataset_root, rq02_dir, budget_fast=a.fast, rq01_tape=rq01_tape)

    print("[rev363] RQ03 ...")
    r3 = R3.run(a.dataset_root, rq03_dir, rq01_panel=panel_prob,
                rq01_thr=float(r1["selected_threshold"]), budget_fast=a.fast)

    print("[rev363] reports ...")
    reports = build_reports(out, r1, r2, r3)

    manifest = {
        "run_id": run_id("V8_FINAL_EXECUTED_RELEASE"),
        "created_utc": utcnow(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": EST.XGB_VERSION,
        "xgb_available": EST.XGB_AVAILABLE,
        "dataset_root": str(a.dataset_root),
        "outputs": sorted(str(p.relative_to(out)).replace("\\", "/")
                          for p in out.rglob("*") if p.is_file()),
        "note": "RUN_MANIFEST has no self-hash by contract (P1-20).",
    }
    jdump(manifest, out / "RUN_MANIFEST.json")
    print(f"[rev363] done -> {out}")
    return {"RQ01": r1, "RQ02": r2, "RQ03": r3, "reports": reports}


if __name__ == "__main__":
    main()
