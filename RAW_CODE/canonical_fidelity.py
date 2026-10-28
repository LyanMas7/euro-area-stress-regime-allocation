"""Check the selected THRESHOLD against what this pipeline can actually score.

The RQ01 selection is a per-country XGBoost-M01 threshold combined with two
packet flags. It is only reproducible if the RQ01 tape carries ``Ps_XGB_M01``
and the packet columns; when those are absent the check reports
NOT_COMPUTABLE_IN_THIS_PIPELINE with the missing columns instead of inventing a
number. RQ02/RQ03 targets are compared with the emitted metrics where present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _find(root: Path, name: str):
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def rq01_check(root: Path, rq01_sel: dict) -> dict:
    need_score = "Ps_XGB_M01"
    need_flags = ["pkt_joint_CLIFS_VIX", "pkt_joint_DFC_GFC"]
    tape = _find(root, "RQ01_DEV_ISSUE_TAPE.csv")
    if tape is None:
        return {"status": "NOT_COMPUTABLE", "reason": "RQ01_DEV_ISSUE_TAPE.csv not produced"}

    df = pd.read_csv(tape)
    have_score = [c for c in df.columns if c.lower() == need_score.lower()]
    have_flags = [c for c in need_flags if c in df.columns]
    missing = ([need_score] if not have_score else []) + [f for f in need_flags if f not in have_flags]
    if missing:
        return {
            "status": "NOT_COMPUTABLE_IN_THIS_PIPELINE",
            "reason": "the emitted RQ01 tape does not carry the selected model output "
                      "and packet flags required by the registered rule",
            "missing_columns": missing,
            "available_probability_columns": [c for c in df.columns if c.startswith("p_M")][:8],
            "registered_rule": rq01_sel["rule"],
            "note": "thresholds are preserved in THRESHOLD/RQ01 and are not substituted.",
        }

    score = df[have_score[0]].astype(float)
    flags = {f: df[f].astype(int) for f in have_flags}
    pred = (score >= df["country"].map(rq01_sel["thresholds"]).astype(float))
    pred = pred | (flags["pkt_joint_CLIFS_VIX"] == 1)
    pred = pred & ~(flags["pkt_joint_DFC_GFC"] == 1)
    y = df["stress"].astype(int)
    fn = int(((y == 1) & (~pred)).sum())
    fp = int(((y == 0) & pred).sum())
    tgt = (rq01_sel.get("validation_target") or {}).get("DEV", {})
    return {
        "status": "COMPUTED",
        "DEV": {"FN": fn, "FP": fp,
                "target_FN": tgt.get("FN"), "target_FP": tgt.get("FP"),
                "matches_target": (fn == tgt.get("FN") and fp == tgt.get("FP"))},
    }


def rq02_check(root: Path, rq02_sel: dict) -> dict:
    f = _find(root, "RQ02_DEV_EXIT_AB.csv")
    if f is None:
        return {"status": "NOT_COMPUTABLE", "reason": "RQ02_DEV_EXIT_AB.csv not produced"}
    df = pd.read_csv(f)
    row = df[(df.get("exit") == rq02_sel["exit_horizon_months"]) & (df.get("model") == "A")]
    if row.empty:
        return {"status": "NOT_COMPUTABLE",
                "reason": f"no EXIT-A row for {rq02_sel['exit_horizon_months']}m"}
    r = row.iloc[0]
    tgt = rq02_sel["provenance"].get("exit_target", {})
    return {"status": "COMPUTED",
            "EXIT_A": {"FNR": float(r["FNR"]), "FPR": float(r["FPR"]),
                       "target_FNR": tgt.get("FNR_pct"), "target_FPR": tgt.get("FPR_pct"),
                       "matches_target": None},
            "note": "the packaged stage pipeline evaluates EXIT on all horizons; the "
                    "registered 3-month targets come from the sealed supervised run and are "
                    "reported for comparison only."}


def rq03_check(root: Path, rq03_sel: dict) -> dict:
    f = _find(root, "RQ03_DEV_PERFORMANCE.csv")
    if f is None:
        return {"status": "NOT_COMPUTABLE", "reason": "RQ03_DEV_PERFORMANCE.csv not produced"}
    df = pd.read_csv(f)
    pol = df[df["strategy"] == "POLICY"]
    if pol.empty:
        return {"status": "NOT_COMPUTABLE", "reason": "no POLICY row"}
    nav = float(pol.iloc[0]["terminal_nav"])
    tgt = rq03_sel["chosen_standard"].get("DEV")
    return {"status": "COMPUTED",
            "policy_terminal_nav": nav, "target_DEV": tgt,
            "matches_target": (None if tgt is None else abs(nav - float(tgt)) < 1e-6),
            "note": "the packaged stage policy is the REV363 allocation engine; the registered "
                    "chosen standard is the S4BOT overlay and is not the same policy object."}


def build(out_root: str | Path, bundle) -> dict:
    root = Path(out_root)
    sel = {"RQ01": bundle.raw["RQ01"], "RQ02": bundle.raw["RQ02"], "RQ03": bundle.raw["RQ03"]}
    res = {
        "threshold_root": str(bundle.root),
        "RQ01": rq01_check(root, sel["RQ01"]),
        "RQ02": rq02_check(root, {
            "exit_horizon_months": bundle.rq02.exit_horizon_months,
            "provenance": {**bundle.rq02.provenance,
                           "exit_target": sel["RQ02"].get("exit", {}).get("validation_target", {}).get("DEV", {})},
        }),
        "RQ03": rq03_check(root, {"chosen_standard": dict(bundle.rq03.chosen_standard)}),
    }
    (root / "THRESHOLD_FIDELITY.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res
