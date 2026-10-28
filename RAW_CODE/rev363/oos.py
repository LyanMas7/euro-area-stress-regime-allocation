"""REV363 one-shot OOS evaluation after the DEV freeze.

The OOS block is 2020-01..2024-12.  This module verifies the DEV freeze hashes
before it reads any OOS label, applies only frozen functions/thresholds, and
records the one-shot evidence.  No threshold is selected on OOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import labels as L
from . import metrics as MET
from . import models as MOD
from . import rq01 as R1
from . import rq02 as R2
from . import rq03 as R3
from .config import (CLASSES, EA7, HORIZONS, HORIZON_MONTHS, OOS_END, OOS_START,
                     RISKY_ASSETS, canonical_json, sha256_json)
from .data import add_months, find_dataset_root
from .features import FeatureSpec

OOS_VERSION = "REV363_OOS_ONE_SHOT_V1"


_OOS_AUTHORIZATION = None


def set_oos_authorization(token):
    """Install the owner token recorded in DEC-015 for this process only."""
    global _OOS_AUTHORIZATION
    _OOS_AUTHORIZATION = token


def _oos_gate(token=None):
    """OOS 2020-2024 stays SEALED: no code path opens it without the DEC-015 owner token.

    The token is resolved from the explicit argument, then the process global, then
    the inherited environment so that worker processes spawned by the orchestrator
    stay authorised without re-declaring it.
    """
    import os
    from .w01_contract import enforce_execution
    resolved = token
    if resolved is None:
        resolved = _OOS_AUTHORIZATION
    if resolved is None:
        resolved = os.environ.get("RQ_OOS_AUTHORIZATION")
    enforce_execution(oos_requested=True, oos_authorization=resolved)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_freeze(rq01_freeze, rq02_freeze, rq03_freeze):
    out = {}
    for name, path in (("RQ01", rq01_freeze), ("RQ02", rq02_freeze), ("RQ03", rq03_freeze)):
        d = _load_json(path)
        sha = d.get("freeze_sha256")
        if not sha:
            raise ValueError(f"{name}_FREEZE_SHA_MISSING")
        # The freeze payload is the object without the sha field.
        payload = {k: v for k, v in d.items() if k != "freeze_sha256"}
        check = sha256_json(payload)
        if check != sha:
            raise ValueError(f"{name}_FREEZE_SHA_MISMATCH")
        out[name] = d
    return out


def _oos_panel(dataset_root, spec):
    _oos_gate()
    return R1.prepare_panel(dataset_root, allow_oos=True, row_start=OOS_START,
                            row_end=OOS_END, data_end=OOS_END, spec=spec)


def run_rq01_oos(dataset_root, out_dir, freeze):
    _oos_gate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec.from_dict(freeze["feature_spec"])
    prep = _oos_panel(dataset_root, spec)
    panel = prep["panel"]
    if len(panel) != 420:
        raise AssertionError(f"RQ01_OOS_420_REQUIRED_GOT_{len(panel)}")
    models = MOD.deserialize_models(freeze["models"])
    p = R1.model_probabilities(prep["design"], models["M11"])
    score = 1.0 - p[:, 0]
    thr = float(freeze["selected_threshold"]["threshold"])
    pred = (score >= thr).astype(int)
    y = (panel["label"] != "N").astype(int).values
    y4 = np.asarray([CLASSES.index(x) for x in panel["label"]], int)
    cm = MET.confusion_binary(y, pred)
    pred4 = np.where(pred == 0, 0, p[:, 1:].argmax(axis=1) + 1)
    conf4 = MET.confusion4(y4, pred4)
    per_country = []
    for c in EA7:
        m = (panel["country"] == c).values
        per_country.append({"country": c, **MET.confusion_binary(y[m], pred[m])})
    per_phase = []
    for ph in ("PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT"):
        m = (panel["phase"] == ph).values
        if m.sum():
            per_phase.append({"phase": ph, "n": int(m.sum()), **MET.confusion_binary(y[m], pred[m])})
    tape = panel[["country", "month", "label", "phase"]].copy()
    tape["score"] = score
    tape["decision_stress"] = pred
    tape["decision_class"] = np.asarray(CLASSES)[pred4]
    for i, c in enumerate(CLASSES):
        tape[f"p_{c}"] = p[:, i]
    tape.to_csv(out / "RQ01_OOS_ISSUE_TAPE.csv", index=False)
    pd.DataFrame(per_country).to_csv(out / "RQ01_OOS_COUNTRY.csv", index=False)
    pd.DataFrame(per_phase).to_csv(out / "RQ01_OOS_PHASE.csv", index=False)
    pd.DataFrame(conf4["matrix"], index=CLASSES, columns=CLASSES).to_csv(
        out / "RQ01_OOS_CONFUSION4.csv")
    results = {
        "run_id": f"REV363_RQ01_OOS_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "version": OOS_VERSION, "scope": "EA7_OOS_2020_2024",
        "claim": "OOS_ONE_SHOT_RESEARCH_EVALUATION_NOT_OFFICIAL_ISSUED",
        "freeze_sha256": freeze["freeze_sha256"],
        "n_keys": len(panel), "class_counts": {c: int((y4 == i).sum()) for i, c in enumerate(CLASSES)},
        "frozen_threshold": thr, "no_oos_threshold_selection": True,
        "stress_confusion": cm, "confusion4": conf4,
        "per_country": per_country, "per_phase": per_phase,
        "auc": MET.auc_score(y, score),
        "logloss": MET.multiclass_logloss(y4, p),
        "brier": MET.multiclass_brier(y4, p),
        "gate_fnr_lt_5": bool(cm["FNR"] is not None and cm["FNR"] < 0.05),
        "gate_fpr_lt_15": bool(cm["FPR"] is not None and cm["FPR"] < 0.15),
        "oos_gate": "PASS" if (cm["FNR"] < 0.05 and cm["FPR"] < 0.15) else "FAIL",
        "scientific_boundary": "Frozen DEV model applied without OOS tuning; OOS failure is a genuine result.",
    }
    (out / "RQ01_OOS_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n",
                                               encoding="utf-8")
    (out / "EXECUTION_EVIDENCE_OOS.json").write_text(json.dumps({
        "run_id": results["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m rev363.oos --rq01", "freeze_sha256": freeze["freeze_sha256"],
        "n_oos_rows": len(panel), "oos_read": True, "threshold_selection_on_oos": False,
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.glob("RQ01_OOS_*")) if p.is_file()},
    }, indent=2) + "\n", encoding="utf-8")
    return results


def run_rq02_oos(dataset_root, out_dir, freeze01, freeze02):
    _oos_gate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec.from_dict(freeze01["feature_spec"])
    models01 = MOD.deserialize_models(freeze01["models"])
    prep = _oos_panel(dataset_root, spec)
    panel = prep["panel"]
    calibration = freeze02.get("long_short_calibration")
    if not calibration:
        raise ValueError("RQ02_FREEZE_CALIBRATION_MISSING")
    df = R2.build_rq02_frame(prep, spec, calibration=calibration)
    df = R2.attach_rq01_probability(df, prep, models01)
    cols = R2._feature_columns(df)
    episodes = prep["episodes"]
    horizon_out = {}
    for hn in HORIZONS:
        h = HORIZON_MONTHS[hn]
        onset, fstate, eligible = R2.horizon_labels(df, episodes, h, allow_oos=True, end=OOS_END)
        mout = {}
        for mn in ("M00", "M01", "M10", "M11"):
            m = R2.HorizonModel.from_dict(freeze02["horizons"][hn][mn])
            p_on, p_st = R2.predict_horizon(m, df, cols)
            sub = df[eligible].reset_index(drop=True)
            y_on = np.asarray(onset[eligible], int)
            y_st = np.asarray([CLASSES.index(x) if x in CLASSES else 0 for x in fstate[eligible]], int)
            s = p_on[eligible]
            # frozen DEV threshold, never re-selected
            dev_thr = freeze02.get("thresholds", {}).get(hn, {}).get(mn)
            if dev_thr is None:
                dev_thr = 0.5
            pred = (s >= dev_thr).astype(int)
            cm = MET.confusion_binary(y_on, pred)
            mout[mn] = {"n_eligible": int(eligible.sum()), "n_events": int(y_on.sum()),
                        "auc": MET.auc_score(y_on, s), "logloss": float(-np.mean(np.log(np.clip(
                            np.where(y_on == 1, s, 1 - s), 1e-15, 1)))),
                        "brier": float(np.mean((s - y_on) ** 2)),
                        "frozen_threshold": float(dev_thr), "confusion": cm,
                        "state_logloss": MET.multiclass_logloss(y_st, p_st[eligible]),
                        "state_brier": MET.multiclass_brier(y_st, p_st[eligible])}
        horizon_out[hn] = {"h": h, "n_eligible": int(eligible.sum()),
                           "n_events": int(onset[eligible].sum()), "models": mout}
    # S5
    h6 = R2.HorizonModel.from_dict(freeze02["horizons"]["H6"]["M11"])
    h6_score, _ = R2.predict_horizon(h6, df, cols)
    h6_thr = freeze02.get("thresholds", {}).get("H6", {}).get("M11", 0.5)
    s5 = R2.s5_tape(df, h6_score, h6_thr)
    s5.to_csv(out / "RQ02_OOS_S5.csv", index=False)
    # EXIT A/B
    exit_out = {}
    for h in (3, 6, 12, 18):
        event, eligible = R2.exit_labels(df, episodes, h, allow_oos=True, end=OOS_END)
        recs = {}
        for tag, key in (("A", "A"), ("B", "B")):
            rec = freeze02["exit"][f"EXIT_{h}"][key]
            use = rec["columns"]
            X = df.loc[eligible, use].astype(float).values
            Xs = np.nan_to_num((X - np.asarray(rec["mean"])) / np.asarray(rec["scale"]),
                               nan=0.0, posinf=0.0, neginf=0.0)
            p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ np.asarray(rec["coef"]) + rec["intercept"], -40, 40)))
            y = np.asarray(event[eligible], int)
            recs[tag] = {"n": int(eligible.sum()), "n_events": int(y.sum()),
                         "n_censored": int(len(y) - y.sum()),
                         "auc": MET.auc_score(y, p) if len(np.unique(y)) > 1 else None,
                         "logloss": float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p),
                                                                  1e-15, 1)))) if len(y) else None,
                         "confusion_0_5": MET.confusion_binary(y, (p >= 0.5).astype(int))}
        exit_out[f"EXIT_{h}"] = recs
    # OOS CSV tables mirroring the DEV schema
    onset_rows, state_rows, f_rows = [], [], []
    for hn, hrec in horizon_out.items():
        base_ll = hrec["models"]["M00"]["logloss"]
        base_br = hrec["models"]["M00"]["brier"]
        for mn, mv in hrec["models"].items():
            onset_rows.append({"horizon": hn, "model": mn, "n": mv["n_eligible"],
                               "events": mv["n_events"], "auc": mv["auc"],
                               "logloss": mv["logloss"], "brier": mv["brier"],
                               "frozen_threshold": mv["frozen_threshold"],
                               **{k: mv["confusion"][k] for k in
                                  ("TP", "FN", "FP", "TN", "FNR", "FPR")}})
            state_rows.append({"horizon": hn, "model": mn, "n": mv["n_eligible"],
                               "state_logloss": mv["state_logloss"],
                               "state_brier": mv["state_brier"]})
            f_rows.append({"horizon": hn, "model": mn, "n": mv["n_eligible"],
                           "events": mv["n_events"], "F1_logloss": mv["logloss"],
                           "F2_brier": mv["brier"], "F3_auc": mv["auc"],
                           "F5_delta_logloss_vs_M00": mv["logloss"] - base_ll,
                           "F5_delta_brier_vs_M00": mv["brier"] - base_br,
                           "frozen_threshold": mv["frozen_threshold"]})
    pd.DataFrame(onset_rows).to_csv(out / "RQ02_OOS_ONSET_METRICS.csv", index=False)
    pd.DataFrame(state_rows).to_csv(out / "RQ02_OOS_FUTURE_STATE_METRICS.csv", index=False)
    pd.DataFrame(f_rows).to_csv(out / "RQ02_OOS_F1_F5.csv", index=False)
    ex_rows = []
    for k, v in exit_out.items():
        for tag in ("A", "B"):
            r = v[tag]
            ex_rows.append({"exit": k, "model": tag, "n": r["n"], "events": r["n_events"],
                            "n_censored": r["n_censored"], "auc": r["auc"],
                            "logloss": r["logloss"],
                            **{kk: r["confusion_0_5"][kk] for kk in
                               ("TP", "FN", "FP", "TN", "FNR", "FPR")}})
    pd.DataFrame(ex_rows).to_csv(out / "RQ02_OOS_EXIT_AB.csv", index=False)
    R2.source_matrix(df).to_csv(out / "RQ02_OOS_SOURCE_MATRIX.csv", index=False)
    R2.pair_motif_matrix(df).to_csv(out / "RQ02_OOS_PAIR_MOTIFS.csv", index=False)
    # override ledger
    thr01 = float(freeze01["selected_threshold"]["threshold"])
    p01 = R1.model_probabilities(prep["design"], models01["M11"])
    score01 = 1 - p01[:, 0]
    ledger = df[["country", "month", "label"]].copy()
    ledger["rq01_score"] = score01
    ledger["pred_stress"] = (score01 >= thr01).astype(int)
    ledger["effective_state"] = np.where(ledger["pred_stress"] == 1, "Stress", "N")
    ledger["override"] = (ledger["label"] == "N") & (ledger["effective_state"] == "Stress")
    ledger["reverse_override"] = (ledger["label"] != "N") & (ledger["effective_state"] == "N")
    ledger.to_csv(out / "RQ02_OOS_OVERRIDE_LEDGER.csv", index=False)
    # Bottom 3F OOS scoring stays SEALED.  The old in-line proxy (composite EA5 F3, groupby(month).max(), zone
    # starting AT the issue) violated the Bottom spec; the only conformant implementation is the DEV harness
    # research_adapters.rq02_bottom_dev_harness, which refuses any tape beyond 2019-12-31.
    trough_month = None
    zrows = []
    pd.DataFrame([{"status": "OOS_BOTTOM_SEALED_NOT_SCORED"}]).to_csv(out / "RQ02_OOS_BOTTOM.csv", index=False)
    results = {
        "run_id": f"REV363_RQ02_OOS_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "version": OOS_VERSION, "claim": "OOS_ONE_SHOT_RESEARCH_EVALUATION_NOT_OFFICIAL_ISSUED",
        "freeze_sha256": freeze02["freeze_sha256"], "horizons": horizon_out,
        "s5": {"n_rows": len(s5), "n_confirmed": None, "n_proxy_trigger": int(s5["s5_proxy_trigger_18m"].sum()),
               "official_issued": False, "proxy": True},
        "exit": exit_out,
        "override": {"n_N_to_stress": int(ledger["override"].sum()),
                     "n_stress_to_N": int(ledger["reverse_override"].sum())},
        "bottom": {"n_zones": len(zrows), "n_hit": sum(1 for r in zrows if r["status"] == "HIT"),
                   "n_miss": sum(1 for r in zrows if r["status"] == "MISS"),
                   "n_no_zone": sum(1 for r in zrows if r["status"] in ("NO_ZONE", "ABSTAIN")),
                   "trough_month": trough_month, "status": "OOS_BOTTOM_SEALED_NOT_SCORED"},
        "no_oos_tuning": True,
    }
    (out / "RQ02_OOS_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n",
                                               encoding="utf-8")
    (out / "EXECUTION_EVIDENCE_OOS.json").write_text(json.dumps({
        "run_id": results["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m rev363.oos --rq02", "freeze_sha256": freeze02["freeze_sha256"],
        "n_oos_panel": len(df), "oos_read": True, "threshold_selection_on_oos": False,
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.glob("RQ02_OOS_*")) if p.is_file()},
    }, indent=2) + "\n", encoding="utf-8")
    return results


def run_rq03_oos(dataset_root, out_dir, freeze01, freeze03):
    _oos_gate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec.from_dict(freeze01["feature_spec"])
    models01 = MOD.deserialize_models(freeze01["models"])
    prep = _oos_panel(dataset_root, spec)
    panel = prep["panel"]
    p01 = R1.model_probabilities(prep["design"], models01["M11"])
    for i, c in enumerate(CLASSES):
        panel[f"p_rq01_{c}"] = p01[:, i]
    tape = R3.build_oos_tape(find_dataset_root(dataset_root))
    tape = R3.attach_regimes(tape, panel)
    permission = R3.bottom_permission(panel, tape)
    tape = tape.merge(permission, on="month", how="left")
    # frozen GARCH/DCC on the combined DEV+OOS return history
    dev = R3.load_dev_tape(find_dataset_root(dataset_root))
    combined = pd.concat([dev[["week_end"] + list(RISKY_ASSETS)],
                          tape[["week_end"] + list(RISKY_ASSETS)]], ignore_index=True)
    R = combined[list(RISKY_ASSETS)].values
    g = freeze03["garch"]
    cov_all = R3.dcc_cov(R, g, a=freeze03["dcc_a"], b=freeze03["dcc_b"])
    cov_oos = cov_all[len(dev):]
    regimes = tape["regime_exante"].values
    perm = tape["permission"].fillna(0).values
    books = R3.backtest(tape, cov_oos, regimes, perm, fee=freeze03["fee"], short_overlay=True)
    books_no_short = R3.backtest(tape, cov_oos, regimes, perm, fee=freeze03["fee"], short_overlay=False)
    perf = [{"strategy": b, **R3.performance(books[b]["nav"], books[b]["turnover"]),
             "short_overlay": b == "POLICY"} for b in R3.BENCHMARKS]
    nav = pd.DataFrame({"week_end": tape["week_end"]})
    for b in R3.BENCHMARKS:
        nav[b] = books[b]["nav"]
    nav["regime_exante"] = tape["regime_exante"]
    nav["regime_expost"] = tape["regime_expost"]
    nav.to_csv(out / "RQ03_OOS_NAV_NET.csv", index=False)
    pd.DataFrame(perf).to_csv(out / "RQ03_OOS_PERFORMANCE.csv", index=False)
    wr = []
    for b in R3.BENCHMARKS:
        for t in range(len(tape)):
            wk = str(tape.loc[t, "week_end"].date())
            wr.append({"week_end": wk, "strategy": b,
                       **{f"w_{a}": float(books[b]["weights"][t, j])
                          for j, a in enumerate(RISKY_ASSETS)},
                       "cash_weight": float(max(0.0, 1.0 - books[b]["weights"][t].sum())),
                       "short_weight": float(books[b]["short"][t]),
                       "turnover": float(books[b]["turnover"][t]),
                       "cost": float(books[b]["cost"][t])})
    pd.DataFrame(wr).to_csv(out / "RQ03_OOS_WEIGHTS.csv", index=False)
    permission.to_csv(out / "RQ03_OOS_PERMISSION.csv", index=False)
    # risk table
    rows = []
    for b in R3.BENCHMARKS:
        w = books[b]["weights"]
        ret = np.r_[0.0, np.diff(books[b]["nav"]) / books[b]["nav"][:-1]]
        for regime in R3.REGIMES:
            mask = (tape["regime_expost"] == regime).values
            for alpha in R3.ALPHAS:
                vr, es, br, essv = [], [], [], []
                for t in np.where(mask)[0]:
                    if not np.isfinite(cov_oos[t]).all():
                        continue
                    ww = w[t] if w[t].sum() > 0 else np.ones(5) / 5
                    v, e, _ = R3.var_es(ww, np.zeros(5), cov_oos[t], alpha)
                    vr.append(v); es.append(e); br.append(int(ret[t] < -v))
                    rc_ = ww * (cov_oos[t] @ ww)
                    q_ = rc_ / rc_.sum() if abs(rc_.sum()) > 1e-18 else np.ones(5) / 5
                    essv.append(float(1.0 / np.sum(q_ ** 2)))
                rows.append({"strategy": b, "regime": regime, "alpha": alpha,
                             "n_weeks": len(vr), "VaR": float(np.mean(vr)) if vr else None,
                             "ES": float(np.mean(es)) if es else None,
                             "ESS": float(np.mean(essv)) if essv else None,
                             "realized_mean": float(np.mean(ret[mask])) if mask.any() else None,
                             "breach_rate": float(np.mean(br)) if br else None,
                             "status": "EXECUTED_OOS" if vr else "NOT_EVALUATED_INSUFFICIENT_REGIME_SUPPORT"})
    pd.DataFrame(rows).to_csv(out / "RQ03_OOS_VAR_ES_72.csv", index=False)
    # capital
    cap = []
    for t in range(len(tape)):
        S = cov_oos[t]
        if not np.isfinite(S).all():
            cap.append({"week_end": str(tape.loc[t, "week_end"].date()), "regime": regimes[t],
                        "feasible": None, "status": "NOT_EVALUATED_WARMUP"})
            continue
        _, es, _ = R3.var_es(np.ones(5) / 5, np.zeros(5), S, 0.95)
        lo, hi = (0.50, 0.75) if regimes[t] == "N" else (0.0, 0.50)
        c_max = min(hi, 0.02 / es) if es > 0 else hi
        cap.append({"week_end": str(tape.loc[t, "week_end"].date()), "regime": regimes[t],
                    "ES95_unit": es, "c_min": lo, "c_max": c_max,
                    "feasible": int(c_max >= lo), "status": "EVALUATED"})
    cap = pd.DataFrame(cap)
    cap.to_csv(out / "RQ03_OOS_CAPITAL_FEASIBILITY.csv", index=False)
    # short bottom
    eq = tape[["E1", "E2", "E3"]].mean(axis=1).values
    short = books["POLICY"]["short"]
    sb = []
    for t in range(len(tape)):
        if short[t] > 0:
            sb.append({"week_end": str(tape.loc[t, "week_end"].date()),
                       "short_notional": float(short[t]),
                       "equity_return": float(eq[t]),
                       "avoided_loss": float(short[t] * (-eq[t])) - freeze03["fee"] * float(short[t])})
    sb = pd.DataFrame(sb)
    if len(sb) == 0:
        sb = pd.DataFrame(columns=["week_end", "short_notional", "equity_return", "avoided_loss"])
    sb.to_csv(out / "RQ03_OOS_SHORT_BOTTOM.csv", index=False)
    results = {
        "run_id": f"REV363_RQ03_OOS_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "version": OOS_VERSION, "claim": "OOS_ONE_SHOT_RESEARCH_EVALUATION_NOT_OFFICIAL_ISSUED",
        "freeze_sha256": freeze03["freeze_sha256"], "n_weeks": len(tape),
        "performance": perf, "risk_rows": len(rows),
        "capital": {"n_evaluated": int((cap["status"] == "EVALUATED").sum()),
                    "n_feasible": int(cap.loc[cap["status"] == "EVALUATED", "feasible"].sum()),
                    "fraction_feasible": float(cap.loc[cap["status"] == "EVALUATED", "feasible"].mean())
                    if (cap["status"] == "EVALUATED").any() else None},
        "short_bottom": {"n_weeks_short": len(sb),
                         "total_avoided_loss": float(sb["avoided_loss"].sum()) if len(sb) else 0.0,
                         "policy_terminal_with_short": float(books["POLICY"]["nav"][-1]),
                         "policy_terminal_without_short": float(books_no_short["POLICY"]["nav"][-1])},
        "no_oos_tuning": True,
    }
    (out / "RQ03_OOS_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n",
                                               encoding="utf-8")
    (out / "EXECUTION_EVIDENCE_OOS.json").write_text(json.dumps({
        "run_id": results["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m rev363.oos --rq03", "freeze_sha256": freeze03["freeze_sha256"],
        "n_oos_weeks": len(tape), "oos_read": True, "threshold_selection_on_oos": False,
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.glob("RQ03_OOS_*")) if p.is_file()},
    }, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--rq01-out", required=True)
    p.add_argument("--rq02-out", required=True)
    p.add_argument("--rq03-out", required=True)
    p.add_argument("--rq01-freeze", required=True)
    p.add_argument("--rq02-freeze", required=True)
    p.add_argument("--rq03-freeze", required=True)
    p.add_argument("--rq01", action="store_true")
    p.add_argument("--rq02", action="store_true")
    p.add_argument("--rq03", action="store_true")
    a = p.parse_args(argv)
    fr = verify_freeze(a.rq01_freeze, a.rq02_freeze, a.rq03_freeze)
    out = {}
    if a.rq01:
        out["RQ01"] = run_rq01_oos(a.dataset_root, a.rq01_out, fr["RQ01"])
    if a.rq02:
        out["RQ02"] = run_rq02_oos(a.dataset_root, a.rq02_out, fr["RQ01"], fr["RQ02"])
    if a.rq03:
        out["RQ03"] = run_rq03_oos(a.dataset_root, a.rq03_out, fr["RQ01"], fr["RQ03"])
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("run_id", "stress_confusion", "oos_gate", "n_weeks",
                                    "performance", "capital")} for k, v in out.items()},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
