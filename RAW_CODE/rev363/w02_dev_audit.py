"""DEV-only W02 eligibility/source audit. No fitted coefficients or OOS access.

Usage: python -m rev363.w02_dev_audit --dataset-root <ASSET03> --out <output>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EA7, RELEASE_RULES, SOURCES7
from .panel import build_panel
from .w02_integrity import (make_dev_onset_keys, make_dev_exit_keys,
                            NATIVE_FREQUENCY, LONG_SOURCES, SHORT_SOURCES)

RUN_ID = "W02-DEV-TARGET-NATIVE-AUDIT-20261023-001"


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(dataset_root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Deliberately NEVER ingest after DEV_END. Pre-DEV 2004 only for lookback.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        obj = build_panel(dataset_root, row_start="2004-01", row_end="2019-12",
                          data_end="2019-12")
    panel = obj["panel"]
    episodes = obj["episodes"]
    all_dev = panel.loc[panel.month.between("2005-01", "2019-12")]
    onset = make_dev_onset_keys(panel, episodes)
    exits = make_dev_exit_keys(panel, episodes)
    assert panel.month.max() == "2019-12"
    assert set(panel.country).issubset(set(EA7))
    assert not onset.empty and not exits.empty
    assert all(onset.target_end <= "2019-12")
    assert all(exits.target_end <= "2019-12")
    assert all(onset.research_label_mature_at <= "2019-12")
    assert all(exits.research_label_mature_at <= "2019-12")
    assert not onset.duplicated(["country", "issue", "horizon"]).any()
    assert not exits.duplicated(["country", "issue", "exit_horizon"]).any()
    onset.to_csv(out / "RQ02_DEV_ONSET_TARGET_KEYS.csv", index=False)
    exits.to_csv(out / "RQ02_DEV_EXIT_TARGET_KEYS.csv", index=False)
    hv = (onset.groupby("horizon", sort=True)
          .agg(n_keys=("first_onset", "size"), onset_positives=("first_onset", "sum"),
               future_stress_positives=("future_stress", "sum"),
               n_countries=("country", "nunique"), first_issue=("issue", "min"),
               last_issue=("issue", "max")) .reset_index())
    hv["onset_negatives"] = hv.n_keys - hv.onset_positives
    hv["n_complete_dev_country_month"] = 7 * 180
    hv.to_csv(out / "RQ02_DEV_ONSET_ELIGIBILITY.csv", index=False)
    ev = (exits.groupby("exit_horizon", sort=True)
          .agg(n_keys=("first_n", "size"), first_n_positives=("first_n", "sum"),
               n_countries=("country", "nunique"), first_issue=("issue", "min"),
               last_issue=("issue", "max")) .reset_index())
    ev["first_n_negatives"] = ev.n_keys - ev.first_n_positives
    ev.to_csv(out / "RQ02_DEV_EXIT_ELIGIBILITY.csv", index=False)
    reg = []
    for s in SOURCES7:
        core = pd.to_numeric(all_dev[f"z_{s}"], errors="coerce")
        rule = RELEASE_RULES[s]
        reg.append(dict(source=s, branch="LONG_SHORT" if s in LONG_SOURCES else "SHORT",
                        native_frequency=NATIVE_FREQUENCY[s],
                        research_release_rule=rule["basis"],
                        issuer_release_status="NOT_VERIFIED", threshold_L=None,
                        threshold_U=None, threshold_STEP=None,
                        n_dev_monthly_rows=int(len(core)),
                        n_finite_z_monthly=int(np.isfinite(core).sum()),
                        n_unique_countries=int(all_dev.country.nunique()),
                        note="Monthly as-of rows are NOT independent native releases"))
    reg = pd.DataFrame(reg)
    reg.to_csv(out / "RQ02_NATIVE_SOURCE_READINESS.csv", index=False)
    report = {
        "run_id": RUN_ID,
        "claim": "DEV_ONLY_TARGET_ELIGIBILITY_AUDIT_NOT_FIT_OR_ISSUED_FORECAST",
        "root": str(obj["root"]),
        "n_dev_calendar_rows": int(len(all_dev)),
        "n_panel_warmup_plus_dev": int(len(panel)),
        "countries": sorted(all_dev.country.unique().tolist()),
        "dev_start": "2005-01", "dev_end": "2019-12",
        "n_onset_horizon_keys": int(len(onset)), "n_exit_horizon_keys": int(len(exits)),
        "research_label_lag": "3 MONTHS ASSUMPTION ONLY; EX-POST PUBLICATION NOT VERIFIED",
        "oos_data_accessed": False,
        "fit_eligible": False,
        "gates": {"native_issuer_pit": "NOT_VERIFIED",
                  "country_EA8": "CONTRACT_CONFLICT_EA7_ONLY",
                  "rq02_owner_approved_target": "NOT_VERIFIED",
                  "xgb_gate_a": "DORMANT",
                  "xgb_gate_b": "NOT_CERTIFIED",
                  "expanding_E1_E3": "BLOCKED_BY_CONTRACT",
                  "supervised_fit": "NOT_RUN",
                  "oos_evaluation": "NOT_RUN_SEALED"},
        "files": {}}
    for f in sorted(out.glob("RQ02_*.csv")):
        report["files"][f.name] = {"sha256": _sha_file(f), "rows": int(len(pd.read_csv(f)))}
    (out / "W02_EVIDENCE.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(json.dumps(audit(args.dataset_root, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
