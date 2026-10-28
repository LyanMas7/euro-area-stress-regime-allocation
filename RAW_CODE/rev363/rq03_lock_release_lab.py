"""Lock-release / fallback-Bottom + constrained-risk research lab (NA-046 implementation).

RQ01/RQ02 frozen selections are untouched. No protected file is edited: the lock state
machine is wrapped in memory (original gates are produced first, then a research
fallback-release rule is applied), and risk configurations/profiles are patched in
memory and restored. Every variant records its effective profile and rule.

Rules tested:
  * REL_NOBOTTOM_N — release Main to Common after N locked weeks **if the episode has no
    Bottom event at all** (targets EP006 2014-08 / EP004 2023-06, 278 DEV / 78 OOS weeks).
  * REL_ANY_N — release after N locked weeks regardless of a later Bottom.
  * EWMA constrained risk — EWMA_GAUSS / EWMA_T5 with ES caps 2.5–4% to reduce the DEV
    hard-cap violations that rejected EWMA at the 2% cap.

DEV validity veto (before any OOS): ES-cap, Normal-band and unlocked hard-violation weeks
must not exceed the canonical baseline; Bottom overlay violations must stay zero; MDD<=20%.
OOS finalists (top valid by DEV NAV) are run once with benchmarks. OOS is post-exposure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rev363 import rq03_07_bottom_permission as BP
from rev363.rq03_e2e_engine import RiskConfig

CACHE = "RQ03_WEIGHT_GATE_RUN_20261025/_cache/inputs_FULL.pkl"
DEV_PACKET = "RQ02_DEV_ONLY_F3B_20261025/RQ02_11/RQ02_11_RQ03_ISSUED_PACKET.json"
OOS_PACKET = "DEV_RUN_20261025_FNR_FPR_BAND_CALIBRATION/RQ02_FULL_V3_F3B/RQ02_11/RQ02_11_RQ03_ISSUED_PACKET.json"
OOS_TOKEN = "USER_AUTHORIZED_OOS_20261025_DEC015"
OUT = "AI_HANDOFF/RQ03_LOCK_RELEASE_LAB_20261025"


def _root():
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "AI_HANDOFF").is_dir() and (p / "Code chính").is_dir():
            return p
    raise RuntimeError("PROJECT_ROOT_NOT_FOUND")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_release(orig, weeks: int, mode: str):
    def patched(P, events):
        gates = orig(P, events)
        bottom_eps = {e["episode_id"] for e in events
                      if e["type"] == "BOTTOM_CONFIRMED" and e.get("episode_id")}
        starts, out = {}, []
        for t, g in enumerate(gates):
            eid = g.get("episode_id")
            locked = g.get("locked") is True
            if locked and eid not in starts:
                starts[eid] = t
            if locked and eid in starts and (t - starts[eid]) >= weeks:
                release = (mode == "ANY") or (eid not in bottom_eps)
                if release:
                    g = dict(g)
                    g["locked"] = False
                    g["trade_allowed"] = True
                    g["effective"] = "MAIN_FALLBACK_RELEASED_RESEARCH"
                    g["fallback_release_week"] = t
                    g["fallback_mode"] = mode
                    g["fallback_after_weeks"] = weeks
            out.append(g)
        return out
    return patched


def _risk_counts(out_dir: Path, period: str):
    f = out_dir / f"RQ03_07_RISK_VIOLATIONS_{period}.csv"
    if not f.is_file():
        return {}
    d = {r["metric"]: r["value"] for r in csv.DictReader(f.open(encoding="utf-8"))}
    return {k: int(float(d.get(k, 0))) for k in
            ("ES_CAP_BREACH_WEEKS", "NORMAL_BAND_BREACH_WEEKS", "HARD_VIOLATION_UNLOCKED",
             "HARD_VIOLATION_LOCKED_DRIFT", "HARD_VIOLATION_BOTTOM_OVERLAY")}


def _summary(out_dir: Path, period: str):
    s = json.loads((out_dir / f"RQ03_07_SUMMARY_{period}.json").read_text(encoding="utf-8"))["main"]
    eff, buys = 0.0, 0
    f = out_dir / f"RQ03_07_BOTTOM_EFFECT_{period}.csv"
    if f.is_file():
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("terminal_effect") not in (None, ""):
                eff += float(r["terminal_effect"])
            buys += int(float(r.get("bottom_buy_weeks") or 0))
    return {"terminal_nav": s["terminal_nav"], "MDD": s["MDD"],
            "mean_log_growth": s["mean_log_growth"], "fees_money": s["fees_money"],
            "turnover": s["total_turnover"], "locked_weeks": s.get("n_NO_TRADE_LOCKED", 0),
            "avg_risky_share": s["avg_risky_share"], "n_trades": s["n_trades"],
            "bottom_effect_sum": eff, "bottom_buy_weeks": buys}


def _run(root: Path, spec: dict, out_dir: Path, *, oos: bool, benchmarks: bool):
    H = root / "AI_HANDOFF"
    orig_research = dict(BP.RESEARCH)
    orig_cfgs = BP.E.RISK_CONFIGS
    orig_cfg_name = BP.R.PRIMARY_CFG
    orig_lock = BP.simulate_lock
    try:
        BP.RESEARCH.update(spec.get("overrides") or {})
        cfg_name = spec.get("cfg")
        if cfg_name and "_ESCAP_" in cfg_name:
            base, cap = cfg_name.rsplit("_ESCAP_", 1)
            cov = "EWMA" if base.startswith("EWMA") else "DCC"
            measure = base.split("_")[-1]
            BP.E.RISK_CONFIGS = tuple(c for c in orig_cfgs if c.name != cfg_name) + (
                RiskConfig(cfg_name, cov, measure, 0.95, float(cap)),)
            BP.R.PRIMARY_CFG = cfg_name
        elif cfg_name:
            BP.R.PRIMARY_CFG = cfg_name
        if spec.get("release_weeks") is not None:
            BP.simulate_lock = _make_release(orig_lock, int(spec["release_weeks"]), spec["release_mode"])
        out_dir.mkdir(parents=True, exist_ok=True)
        BP.run(H / CACHE, H / DEV_PACKET, H / OOS_PACKET, out_dir,
               run_benchmarks=benchmarks, skip_oos=not oos,
               oos_authorization=(OOS_TOKEN if oos else None))
    finally:
        BP.RESEARCH.clear(); BP.RESEARCH.update(orig_research)
        BP.E.RISK_CONFIGS = orig_cfgs
        BP.R.PRIMARY_CFG = orig_cfg_name
        BP.simulate_lock = orig_lock


def run(root=None):
    root = Path(root) if root else _root()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    variants = [{"name": "BASELINE", "overrides": {}, "cfg": None, "release_weeks": None, "release_mode": None}]
    for n in (26, 52, 78, 104):
        variants.append({"name": f"REL_NOBOTTOM_{n}", "overrides": {}, "cfg": None,
                         "release_weeks": n, "release_mode": "NOBOTTOM"})
    for n in (26, 52, 78):
        variants.append({"name": f"REL_ANY_{n}", "overrides": {}, "cfg": None,
                         "release_weeks": n, "release_mode": "ANY"})
    for base in ("EWMA_GAUSS", "EWMA_T5"):
        for cap in (0.025, 0.030, 0.040):
            variants.append({"name": f"{base}_ESCAP_{cap}", "overrides": {}, "cfg": f"{base}_ESCAP_{cap}",
                             "release_weeks": None, "release_mode": None})

    def dev(i, spec):
        d = scratch / f"v{i:03d}"
        _run(root, spec, d, oos=False, benchmarks=False)
        s = _summary(d, "DEV"); v = _risk_counts(d, "DEV")
        return {"id": i, "name": spec["name"], "cfg": spec["cfg"] or "DCC_GAUSS",
                "release_weeks": spec["release_weeks"], "release_mode": spec["release_mode"],
                **s, **{f"dev_{k}": x for k, x in v.items()}}

    rows = [dev(i, s) for i, s in enumerate(variants)]
    # combos: best valid release + best valid EWMA cap
    base_counts = {k: rows[0][f"dev_{k}"] for k in
                   ("ES_CAP_BREACH_WEEKS", "NORMAL_BAND_BREACH_WEEKS", "HARD_VIOLATION_UNLOCKED",
                    "HARD_VIOLATION_LOCKED_DRIFT", "HARD_VIOLATION_BOTTOM_OVERLAY")}
    base_total = base_counts["HARD_VIOLATION_UNLOCKED"] + base_counts["HARD_VIOLATION_LOCKED_DRIFT"]

    def valid(r):
        total = r["dev_HARD_VIOLATION_UNLOCKED"] + r["dev_HARD_VIOLATION_LOCKED_DRIFT"]
        return (r["MDD"] <= 0.20 and r["dev_ES_CAP_BREACH_WEEKS"] <= base_counts["ES_CAP_BREACH_WEEKS"]
                and r["dev_NORMAL_BAND_BREACH_WEEKS"] <= base_counts["NORMAL_BAND_BREACH_WEEKS"]
                and total <= base_total and r["dev_HARD_VIOLATION_BOTTOM_OVERLAY"] == 0)

    rel = [r for r in rows if r["release_weeks"] is not None and valid(r)]
    ew = [r for r in rows if r["cfg"] != "DCC_GAUSS" and valid(r)]
    combos = []
    best_rel = max(rel, key=lambda r: r["terminal_nav"]) if rel else None
    best_ew = max(ew, key=lambda r: r["terminal_nav"]) if ew else None
    if best_rel and best_ew:
        combos.append({"name": "COMBO_REL_EWMA", "overrides": {}, "cfg": best_ew["cfg"],
                       "release_weeks": best_rel["release_weeks"], "release_mode": best_rel["release_mode"]})
    for spec in combos:
        variants.append(spec)
        rows.append(dev(len(variants) - 1, spec))
    trials = sorted(rows, key=lambda r: (-r["terminal_nav"], r["MDD"]))
    with (out / "DEV_TRIALS.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trials[0].keys())); w.writeheader(); w.writerows(trials)
    valid_rows = [r for r in trials if valid(r)]
    spec_by_name = {v["name"]: v for v in variants}
    finalists = [r for r in valid_rows if r["name"] != "BASELINE"][:3]
    finalist_specs = [spec_by_name[r["name"]] for r in finalists]
    # OOS finalists + baseline (distinct)
    oos_rows = []
    for spec in ([variants[0]] + finalist_specs):
        d = scratch / f"oos_{spec['name']}"
        _run(root, spec, d, oos=True, benchmarks=True)
        s = _summary(d, "OOS"); v = _risk_counts(d, "OOS")
        dev_row = next(r for r in trials if r["name"] == spec["name"])
        oos_rows.append({"name": spec["name"], "cfg": spec["cfg"] or "DCC_GAUSS",
                         "release": f"{spec['release_mode']}:{spec['release_weeks']}" if spec["release_weeks"] else "",
                         "dev_terminal": dev_row["terminal_nav"], "dev_MDD": dev_row["MDD"],
                         "dev_locked": dev_row["locked_weeks"],
                         "oos_terminal": s["terminal_nav"], "oos_MDD": s["MDD"],
                         "oos_locked": s["locked_weeks"], "oos_bottom_buys": s["bottom_buy_weeks"],
                         **{f"oos_{k}": x for k, x in v.items()}})
    keep = out / "FINALIST_RUNS"
    if keep.exists():
        shutil.rmtree(keep)
    keep.mkdir()
    for spec in ([variants[0]] + finalist_specs):
        src = scratch / f"oos_{spec['name']}"
        if src.is_dir():
            shutil.copytree(src, keep / spec["name"], dirs_exist_ok=True)
    shutil.rmtree(scratch)
    result = {"type": "RQ03_LOCK_RELEASE_LAB", "run_version": "RQ03_LOCK_RELEASE_R01_20261025",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "n_variants": len(trials), "valid_n": len(valid_rows),
              "baseline_dev": rows[0], "valid_finalists": finalists, "oos": oos_rows,
              "claim_limits": ["DEV-only selection with hard-violation veto; OOS post-exposure",
                               "additive research rule; protected files unedited",
                               "RQ01/RQ02 frozen selections untouched"]}
    (out / "SUMMARY.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = ["# RQ03 lock-release + constrained-risk lab", "",
             f"Variants: {len(trials)} DEV; valid (hard-violation veto): {len(valid_rows)}.", "",
             "## DEV variants", "",
             "| Variant | Rule | cfg | DEV NAV | MDD | locked | ES | Normal | unlocked | valid |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in trials:
        lines.append(f"| {r['name']} | {r['release_mode'] or ''}{r['release_weeks'] or ''} | {r['cfg']} | "
                     f"{r['terminal_nav']:.6f} | {r['MDD']:.2%} | {r['locked_weeks']} | "
                     f"{r['dev_ES_CAP_BREACH_WEEKS']} | {r['dev_NORMAL_BAND_BREACH_WEEKS']} | "
                     f"{r['dev_HARD_VIOLATION_UNLOCKED']} | {'YES' if valid(r) else 'NO'} |")
    lines += ["", "## OOS", "",
              "| Variant | DEV NAV | OOS NAV | OOS MDD | locked | ES | Normal | unlocked |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in oos_rows:
        lines.append(f"| {r['name']} | {r['dev_terminal']:.6f} | {r['oos_terminal']:.6f} | {r['oos_MDD']:.2%} | "
                     f"{r['oos_locked']} | {r['oos_ES_CAP_BREACH_WEEKS']} | {r['oos_NORMAL_BAND_BREACH_WEEKS']} | "
                     f"{r['oos_HARD_VIOLATION_UNLOCKED']} |")
    lines += ["", "Canonical baseline: DEV 1.424169 / OOS 0.898219; PC-017 DEV 1.515017 / OOS 0.944055; EW OOS 1.086669.", "",
              "RQ01/RQ02 frozen selections unchanged. OOS post-exposure; no GO claim."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    man = {"version_dir": out.name, "created_utc": datetime.now(timezone.utc).isoformat(),
           "files": {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*"))
                     if p.is_file() and p.name != "MANIFEST_SHA256.json"}}
    (out / "MANIFEST_SHA256.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return {"out": str(out), "n_variants": len(trials), "valid_n": len(valid_rows),
            "finalists": [f["name"] for f in finalists], "oos": oos_rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str)[:4000])
