"""REV363 W03 / RQ01 end-to-end DEV pipeline and frozen artifact builder.

Run:
    python -m rev363.rq01 --dataset-root <locked data root> --out <RESULTS/RQ01>

The DEV block is 2005-07..2019-12 (1,218 EA7 country-months).  The run fits the
four-M hierarchy, calibrates the source/pair/pool/phase functions, searches the
finite threshold families, emits every required RQ01 table, and writes a
single frozen artifact.  OOS is not read here.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import labels as L
from . import metrics as MET
from . import models as MOD
from . import thresholds as THR
from .config import (CLASSES, DEV_END, DEV_FIT_START, EA7, FEATURE_VERSION,
                     INCUMBENT_RQ01_M11_L, INCUMBENT_RQ01_M11_THRESHOLD,
                     MODEL_BLOCKS, MODEL_VERSION, PAIRS10, RELEASE_RULES, SOURCES5,
                     canonical_json, sha256_json)
from .data import (PANEL_START, build_monthly_raw, find_dataset_root, month_range)


# ---------------------------------------------------------------------------
# input preparation
# ---------------------------------------------------------------------------
class NonCanonicalPhaseEngineError(RuntimeError):
    """The composite-intensity phase controller was presented as the canonical six-phase chain."""


def phase_chain_claim(panel) -> dict:
    """What the `phase` column of this panel may legitimately be called.

    The canonical chain is PRE_ONSET (Individual/Joint/Pool) + the native REV352/REV353 five-state law
    (research_adapters.rq01.five_state).  The composite-intensity controller of `features.compute_phases` is a
    legacy research proxy: it is reported as such and can never carry the canonical label."""
    engine = None
    if "phase_engine" in getattr(panel, "columns", ()) and len(panel):
        vals = set(panel["phase_engine"].astype(str))
        engine = vals.pop() if len(vals) == 1 else "MIXED:" + ",".join(sorted(vals))
    elif hasattr(panel, "attrs"):
        engine = panel.attrs.get("phase_engine")
    canonical = engine == F.CANONICAL_PHASE_ENGINE
    return {"phase_engine": engine or "UNTAGGED_NOT_CANONICAL", "canonical_six_phase_chain": canonical,
            "canonical_engine": F.CANONICAL_PHASE_ENGINE,
            "warning": None if canonical else
            "COMPOSITE_INTENSITY_CONTROLLER_IS_NOT_THE_REV352_REV353_NATIVE_FIVE_STATE_LAW"}


def require_canonical_phase_engine(engine_id) -> str:
    """Guard for any report that wants to say 'canonical six-phase chain'."""
    if engine_id != F.CANONICAL_PHASE_ENGINE:
        raise NonCanonicalPhaseEngineError(f"NOT_CANONICAL_REV353:{engine_id}")
    return engine_id


def prepare_panel(dataset_root, *, allow_oos: bool, row_start: str, row_end: str,
                  data_end: str, spec: F.FeatureSpec | None = None):
    # Hard gate: a run that did not open OOS may not even REQUEST 2020+ rows from the readers.
    if not allow_oos and (str(data_end) > DEV_END or str(row_end) > DEV_END):
        raise PermissionError(f"OOS_SEALED: allow_oos=False but data_end={data_end} row_end={row_end} > {DEV_END}")
    root = find_dataset_root(dataset_root)
    episodes = L.load_episodes(root)
    raw = build_monthly_raw(root, row_start=PANEL_START, row_end=data_end, data_end=data_end)
    lab = L.label_frame(raw, episodes, allow_oos=allow_oos)
    feats = F.build_raw_features(lab)
    rows = feats[(feats["month"] >= row_start) & (feats["month"] <= row_end)].copy()
    rows = rows.reset_index(drop=True)
    if spec is None:
        spec = F.calibrate_spec(rows, rows["label"])
    votes = F.compute_votes(rows, spec)
    votes = F.compute_phases(votes, spec)
    design = F.build_design(votes, spec, fit=(spec.base_columns == []))
    return {"root": root, "episodes": episodes, "raw": raw, "full": feats,
            "panel": design["df"], "spec": design["spec"], "design": design}


def _y4(panel):
    return np.asarray([CLASSES.index(x) for x in panel["label"]], int)


def _selected_decision(panel, p4, threshold):
    score = 1.0 - p4[:, 0]
    pred = (score >= threshold).astype(int)
    return score, pred


# ---------------------------------------------------------------------------
# metric / table helpers
# ---------------------------------------------------------------------------
def _design_for_model(design, name):
    X, cols = MOD.build_matrix(design, MODEL_BLOCKS[name])
    return {"X": X, "columns": cols}


def model_probabilities(design, model: MOD.TwoHeadModel):
    X, _ = MOD.build_matrix(design, model.blocks)
    return MOD.predict_probabilities_raw(X, model.stress_coef, model.stress_intercept,
                                         model.subtype_coef, model.subtype_intercept)


def evaluate_models(design, models, panel, y4, selected_name, selected_threshold):
    out = {}
    base_p = None
    for name, model in models.items():
        p = model_probabilities(design, model)
        score = 1.0 - p[:, 0]
        argmax = p.argmax(axis=1)
        conf4 = MET.confusion4(y4, argmax)
        stress_argmax = MET.confusion_binary((y4 != 0).astype(int), (argmax != 0).astype(int))
        # threshold-optimised stress decision for every M
        tr = THR.exact_threshold_search(score, (y4 != 0).astype(int))
        pred = (score >= tr.threshold).astype(int)
        cstress = MET.confusion_binary((y4 != 0).astype(int), pred)
        f = MET.f1_f5(y4, p, score, name, baseline_p=base_p,
                      n_masks=tr.n_unique_masks, edf=len(model.columns["all"]) * 3)
        if base_p is None:
            base_p = p
        out[name] = {
            "model_version": model.version, "blocks": list(model.blocks),
            "n_features": len(model.columns["all"]), "n_fit": model.n_fit,
            "p4": p, "score": score, "argmax": argmax,
            "confusion4_argmax": conf4,
            "stress_argmax": stress_argmax,
            "threshold": tr.__dict__,
            "stress_threshold": cstress,
            "F1_F5": f,
            "fit_logloss": model.fit_logloss, "fit_brier": model.fit_brier,
        }
        out[name]["selected"] = bool(name == selected_name)
    return out


def choose_selected(evaluated):
    """Pre-registered selection: M11 if it passes the strict gate, else the
    feasible M with the smallest FP, else M11 as the closest-violation
    challenger.  This is a DEV model-selection rule, not a claim of OOS skill.
    """
    order = ["M11", "M01", "M10", "M00"]
    feasible = [m for m in order if evaluated[m]["threshold"]["gate_feasible"]]
    if feasible:
        return feasible[0], "M11_FIRST_FEASIBLE_PRE_REGISTERED"
    # no feasible model: choose the one with the lowest normalised violation
    def viol(m):
        t = evaluated[m]["threshold"]
        return (max(0, t["FNR"] - 0.05) / 0.05 + max(0, t["FPR"] - 0.15) / 0.15, t["FP"])
    return sorted(order, key=viol)[0], "NO_FEASIBLE_M11_CLOSEST_VIOLATION"


# ---------------------------------------------------------------------------
# required output blocks
# ---------------------------------------------------------------------------
def individual_table(panel):
    rows = []
    y = (panel["label"] != "N").astype(int).values
    for s in SOURCES5:
        v = panel[f"vote_{s}"].astype(str)
        # movement letter U = UP; a missing level (M) is NOT_EVALUATED, excluded from the confusion matrix
        ev = (v != "M").values
        pred = (v == "U").astype(int).values
        c = MET.confusion_binary(y[ev], pred[ev])
        rows.append({"source": s, "UP": int((v == "U").sum()), "DOWN": int((v == "D").sum()),
                     "SIDEWAY": int((v == "S").sum()), "MISSING": int((v == "M").sum()),
                     "count_up": int((v == "U").sum()),
                     "strong_count": int(panel[f"strong_{s}"].sum()),
                     "terminal_mean_up": float(panel.loc[v == "U", f"d6_{s}"].mean())
                     if (v == "U").any() else None,
                     "terminal_mean_down": float(panel.loc[v == "D", f"d6_{s}"].mean())
                     if (v == "D").any() else None,
                     "stress_TP": c["TP"], "stress_FP": c["FP"],
                     "stress_FN": c["FN"], "stress_TN": c["TN"],
                     "stress_FNR": c["FNR"], "stress_FPR": c["FPR"],
                     "n_evaluated": int(ev.sum()), "n_not_evaluated": int((~ev).sum())})
    return rows


def joint_table(panel, spec):
    y = (panel["label"] != "N").astype(int).values
    ind = panel["individual_vote"].values
    out = []
    for p in PAIRS10:
        a, b = p
        for motif in F.MOTIFS:
            mask = (panel[f"pair_{a}_{b}_state"] == motif).values
            known = ~((panel[f"vote_{a}"] == "M").values | (panel[f"vote_{b}"] == "M").values)
            mask = mask & known
            rescued = int(((ind == "FAIL") & mask & (y == 1)).sum())
            newfp = int(((ind == "FAIL") & mask & (y == 0)).sum())
            out.append({"pair": f"{a}-{b}", "motif": motif, "chosen": motif == spec.selected_motifs.get(p),
                        "n_pass": int(mask.sum()), "rescued_fn": rescued, "new_fp": newfp,
                        "coexceedance": int(((ind == "PASS") & mask).sum())})
    return out


def pool_table(panel):
    y = (panel["label"] != "N").astype(int).values
    rows = []
    for op in ("IND_ONLY", "JOINT_ONLY", "IND_AND_JOINT", "IND_OR_JOINT",
               "COMPLEMENT_AND_JOINT", "COMPLEMENT_OR_JOINT"):
        v = panel[f"pool_{op}"]
        st = panel[f"pool_{op}_status"] if f"pool_{op}_status" in panel.columns else             pd.Series(np.where(v.isin(["PASS", "FAIL"]), "EXECUTED", "NOT_EVALUATED"), index=panel.index)
        ev = (st == "EXECUTED").values
        pred = (v == "PASS").astype(int).values
        c = MET.confusion_binary(y[ev], pred[ev])          # executed (binary) pool decisions only
        pre = panel[panel["phase"] == "PRE"]
        pre_st = st[panel["phase"] == "PRE"]
        pre_tp = int(((pre[f"pool_{op}"] == "PASS") & (pre["label"] != "N")).sum())
        pre_fp = int(((pre[f"pool_{op}"] == "PASS") & (pre["label"] == "N")).sum())
        pre_ne = int((pre_st != "EXECUTED").sum())
        rows.append({"operator": op, "PASS": int((v == "PASS").sum()),
                     "FAIL": int((v == "FAIL").sum()), "N_NOT_EVALUATED": int((~ev).sum()),
                     "pre_entry_TP": pre_tp, "pre_entry_FP": pre_fp,
                     "pre_entry_N_NOT_EVALUATED": pre_ne,
                     **{k: c[k] for k in ("TP", "FP", "FN", "TN", "FNR", "FPR")}})
    return rows


def phase_table(panel):
    rows = []
    engine = phase_chain_claim(panel)["phase_engine"]
    for c in EA7:
        sub = panel[panel["country"] == c]
        counts = sub["phase"].value_counts().to_dict()
        starts = sub[sub["phase_start"] == 1]["phase"].value_counts().to_dict()
        rows.append({"country": c, **{p: int(counts.get(p, 0)) for p in
                                      ("PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT")},
                     **{f"start_{p}": int(starts.get(p, 0)) for p in
                        ("ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT")},
                     "phase_engine": engine})
    return rows


def phase_transitions(panel):
    """Counts of observed phase-to-phase transitions by country."""
    counts = {}
    for c in EA7:
        sub = panel[panel["country"] == c].sort_values("month")
        seq = list(sub["phase"])
        for a, b in zip(seq, seq[1:]):
            if a != b:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    engine = phase_chain_claim(panel)["phase_engine"]
    return [{"from": a, "to": b, "count": n, "phase_engine": engine}
            for (a, b), n in sorted(counts.items())]


def phase_occurrence(panel):
    """P(k sources confirmed | phase start) with Wilson intervals."""
    starts = panel[panel["phase_start"] == 1]
    out = []
    for phase in ("ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT"):
        sub = starts[starts["phase"] == phase]
        n = len(sub)
        for k in range(1, 6):
            cnt = int((sub["n_up"] >= k).sum())
            lo, hi = MET.wilson_ci(cnt, n) if n else (None, None)
            out.append({"phase": phase, "k_sources": k, "n_phase_starts": n,
                        "count_ge_k": cnt, "probability": (cnt / n if n else None),
                        "ci95_low": lo, "ci95_high": hi,
                        "status": "IDENTIFIABLE" if n >= 10 else "LOW_SUPPORT_NOT_IDENTIFIABLE"})
    return out


def per_country_metrics(panel, pred, y):
    rows = []
    for c in EA7:
        m = (panel["country"] == c).values
        if m.sum() == 0:
            continue
        cm = MET.confusion_binary(y[m], pred[m])
        rows.append({"country": c, "n": int(m.sum()), "n_stress": int(y[m].sum()),
                     **cm})
    return rows


def phase_confusion(panel, pred, y):
    rows = []
    for phase in ("PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT"):
        m = (panel["phase"] == phase).values
        if m.sum() == 0:
            continue
        cm = MET.confusion_binary(y[m], pred[m])
        rows.append({"phase": phase, "n": int(m.sum()), "n_stress": int(y[m].sum()), **cm})
    return rows


def canonical_parity(design, model, panel, threshold):
    """Two independent code paths must produce identical decisions.

    Path A: probability -> score -> threshold.
    Path B: reconstruct score from the serialized linear predictor and the
    frozen scaler, then threshold.  Both use the same public decision contract.
    """
    p = model_probabilities(design, model)
    score_a = 1.0 - p[:, 0]
    dec_a = (score_a >= threshold).astype(int)
    X, _ = MOD.build_matrix(design, model.blocks)
    z = X @ model.stress_coef + model.stress_intercept
    score_b = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    dec_b = (score_b >= threshold).astype(int)
    return {"path_a": "full_main_probability", "path_b": "end_to_end_serialized_linear_predictor",
            "n": len(dec_a), "decisions_identical": bool(np.array_equal(dec_a, dec_b)),
            "scores_max_abs_diff": float(np.max(np.abs(score_a - score_b))),
            "decision_sha256": sha256_json(dec_a.tolist()),
            "status": "PASS" if np.array_equal(dec_a, dec_b) else "FAIL"}


# ---------------------------------------------------------------------------
# main DEV run
# ---------------------------------------------------------------------------
def run_dev(dataset_root, out_dir, *, write_freeze=True, freeze_path=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prep = prepare_panel(dataset_root, allow_oos=False, row_start=DEV_FIT_START,
                         row_end=DEV_END, data_end=DEV_END)
    panel = prep["panel"]
    design = prep["design"]
    spec = prep["spec"]
    episodes = prep["episodes"]
    if len(panel) != 1218:
        raise AssertionError(f"RQ01_DEV_1218_REQUIRED_GOT_{len(panel)}")
    y4 = _y4(panel)
    y = (y4 != 0).astype(int)
    counts = {c: int((y4 == i).sum()) for i, c in enumerate(CLASSES)}
    if counts != {"N": 800, "D": 41, "G": 131, "BOTH": 246}:
        raise AssertionError(f"RQ01_LABEL_CONTRACT_CONFLICT:{counts}")

    models = MOD.fit_all(design, y4, C=1.0)
    # provisional evaluation to select the model
    provisional = {}
    for name, m in models.items():
        p = model_probabilities(design, m)
        s = 1.0 - p[:, 0]
        tr = THR.exact_threshold_search(s, y)
        provisional[name] = {"threshold": tr.__dict__, "p4": p, "score": s}
    selected_name, selection_rule = choose_selected(provisional)
    selected_threshold = provisional[selected_name]["threshold"]["threshold"]
    evaluated = evaluate_models(design, models, panel, y4, selected_name, selected_threshold)
    selected = evaluated[selected_name]
    p4 = selected["p4"]
    score = selected["score"]
    pred = (score >= selected_threshold).astype(int)
    sel_cm = MET.confusion_binary(y, pred)
    bootstrap = MET.block_bootstrap_metrics(y, pred, panel["country"].values)

    # exact threshold families for the selected model
    tr = selected["threshold"]
    dp = THR.country_dp(score, y, panel["country"].values)
    second = panel["intensity"].fillna(panel["intensity"].median()).values
    td = THR.two_dim_search(score, second, y)
    families = {
        "GLOBAL_SCORE_EXACT": tr,
        "SEVEN_COUNTRY_DP": dp,
        "TWO_DIM_BOUNDED_INTENSITY": td,
    }

    # required tables
    ind_rows = individual_table(panel)
    joint_rows = joint_table(panel, spec)
    pool_rows = pool_table(panel)
    phase_rows = phase_table(panel)
    transition_rows = phase_transitions(panel)
    occurrence_rows = phase_occurrence(panel)
    country_rows = per_country_metrics(panel, pred, y)
    phase_cm = phase_confusion(panel, pred, y)

    # 4x4 confusion at the selected threshold decision: N vs stress + subtype argmax
    sel_pred4 = np.where(pred == 0, 0, p4[:, 1:].argmax(axis=1) + 1)
    conf4_decision = MET.confusion4(y4, sel_pred4)

    # F1-F5 table
    f_rows = []
    for name, ev in evaluated.items():
        f = ev["F1_F5"]
        f_rows.append({"model": name, "selected": ev["selected"],
                       **{k: f[k] for k in ("F1_logloss", "F2_brier", "F3_auc_stress",
                                            "F4_selection_aware", "F4_penalty")},
                       "delta_logloss_vs_M00": (f["F5_incremental"] or {}).get("delta_logloss_vs_baseline"),
                       "delta_brier_vs_M00": (f["F5_incremental"] or {}).get("delta_brier_vs_baseline")})

    phase_claim = phase_chain_claim(panel)

    # keyed issue tape
    tape = []
    for i, r in panel.iterrows():
        row = {"country": r["country"], "issue": r["month"], "label": r["label"],
               "stress": int(y[i]), "score": float(score[i]),
               "decision_stress": int(pred[i]),
               "decision_class": CLASSES[int(sel_pred4[i])],
               "phase": r["phase"], "phase_engine": phase_claim["phase_engine"],
               "individual_vote": r["individual_vote"],
               "individual_vote_status": r.get("individual_vote_status"),
               "n_up": int(r["n_up"]), "n_down": int(r["n_down"]),
               "pool_count": int(r["pool_count"]),
               "threshold_id": "REV363_RQ01_M11_GLOBAL_EXACT",
               "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION}
        for name in MODEL_BLOCKS:
            for k, cl in enumerate(CLASSES):
                row[f"p_{name}_{cl}"] = float(evaluated[name]["p4"][i, k])
        tape.append(row)

    # canonical parity
    parity = canonical_parity(design, models[selected_name], panel, selected_threshold)

    # status matrix (12 RQ01 requirements)
    status = [
        {"id": "R1", "output": "EA7 raw/native/labels and issue-time readiness",
         "status": "COMPLETE_RESEARCH_ASOF" if counts == {"N": 800, "D": 41, "G": 131, "BOTH": 246} else "FAIL",
         "unresolved": "OFFICIAL_ISSUER_FIRST_RELEASE_AND_OWNER_SIGNATURE_NOT_CERTIFIED",
         "evidence": "LABEL_COUNTS_AND_PIT_RULES"},
        {"id": "R2", "output": "Individual five sources UP/DOWN/SIDEWAY, Count, terminal",
         "status": "COMPLETE", "unresolved": None, "evidence": "RQ01_DEV_INDIVIDUAL.csv"},
        {"id": "R3", "output": "Joint 10 pairs x 9 motifs, chosen/rejected, both terminals",
         "status": "COMPLETE", "unresolved": "EXACT_LAG_ZERO_ONLY_FOR_RQ01",
         "evidence": "RQ01_DEV_JOINT_PAIRS.csv"},
        {"id": "R4", "output": "Six Pool operators and pre-ENTRY N-to-stress corrections",
         "status": "COMPLETE", "unresolved": "PRE_ENTRY_EVENT_DENOMINATOR_REPORTED_NOT_OWNER_CERTIFIED",
         "evidence": "RQ01_DEV_POOLS.csv"},
        {"id": "R5", "output": "M00/M01/M10/M11 four-class probabilities, 4x4, ROC, FNR/FPR",
         "status": "COMPLETE" if sel_cm["FNR"] < 0.05 and sel_cm["FPR"] < 0.15 else "PARTIAL_GATE_FAIL",
         "unresolved": None if sel_cm["FNR"] < 0.05 and sel_cm["FPR"] < 0.15 else
         "STRICT_FNR_FPR_GATE_NOT_MET",
         "evidence": "RQ01_DEV_FOUR_M_METRICS.csv"},
        # The composite-intensity controller is a legacy proxy; it is not the REV352/REV353 native chain that the
        # design lock prescribes, so R6 can only be PARTIAL until the canonical engine feeds this table.
        {"id": "R6", "output": "Six states PRE/ENTRY/PEAK/DEESC/RECOVERY/EXIT",
         "status": ("COMPLETE" if phase_claim["canonical_six_phase_chain"]
                    else "PARTIAL_COMPOSITE_LEGACY_ENGINE_NOT_CANONICAL_REV353"),
         "unresolved": ("PHASE_LABELS_ARE_CAUSAL_RESEARCH_STATES_NOT_OWNER_GOLD" if phase_claim["canonical_six_phase_chain"]
                        else "PHASES_COME_FROM_" + str(phase_claim["phase_engine"]) +
                        "_USE_research_adapters.rq01.five_state_FOR_THE_CANONICAL_CHAIN"),
         "evidence": "RQ01_DEV_PHASES.csv"},
        {"id": "R7", "output": "Per-source phase-start conditional 1-to-5 source occurrence",
         "status": "COMPLETE_WITH_LOW_SUPPORT", "unresolved": "SIX_DEV_EPISODES_LIMIT_IDENTIFICATION",
         "evidence": "RQ01_DEV_PHASE_OCCURRENCE.csv"},
        {"id": "R8", "output": "Source+pair+pool threshold family L/U/step and unique masks",
         "status": "COMPLETE", "unresolved": "2D_FAMILY_BOUNDED_NOT_EXHAUSTIVE",
         "evidence": "RQ01_DEV_THRESHOLD_REGISTRY.json"},
        {"id": "R9", "output": "F1-F5, penalty and matched country/event uncertainty",
         "status": "COMPLETE", "unresolved": "BOOTSTRAP_IS_COUNTRY_BLOCK_NOT_INDEPENDENT_EVENTS",
         "evidence": "RQ01_DEV_F1_F5.csv"},
        {"id": "R10", "output": "Deterministic issued-style keyed source-model-phase tape",
         "status": "COMPLETE", "unresolved": "RESEARCH_ASOF_NOT_AUTHENTIC_ISSUER_TAPE",
         "evidence": "RQ01_DEV_ISSUE_TAPE.csv"},
        {"id": "R11", "output": "One canonical four-M path full_main vs end_to_end parity",
         "status": "COMPLETE" if parity["status"] == "PASS" else "FAIL",
         "unresolved": None if parity["status"] == "PASS" else "PARITY_MISMATCH",
         "evidence": "RQ01_DEV_CANONICAL_PARITY.json"},
        {"id": "R12", "output": "OOS final issue tape and confusion per country/stress/phase",
         "status": "PENDING_OOS_ONE_SHOT", "unresolved": "OOS_NOT_OPENED_IN_DEV_RUN",
         "evidence": "RQ01_OOS_RESULTS.json (after freeze)"},
    ]

    results = {
        "run_id": f"REV363_RQ01_DEV_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "scope": "EA7_DEV_2005_07_2019_12", "claim": "DEV_APPARENT_RESEARCH_NOT_OFFICIAL_ISSUED",
        "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
        "population": {"n_keys": len(panel), "class_counts": counts,
                       "stress_n": int(y.sum()), "n_n": int((1 - y).sum()),
                       "countries": list(EA7), "dev_block": "2005-01..2019-12_INDIVISIBLE"},
        "pit": {"status": "RESEARCH_ASOF_NOT_OFFICIAL_ISSUER_PIT",
                "rules": {k: dict(v) for k, v in RELEASE_RULES.items()},
                "oos_read": False},
        "feature_spec": spec.to_dict(),
        "models": {name: {k: v for k, v in ev.items()
                          if k not in ("p4", "score", "argmax")}
                   for name, ev in evaluated.items()},
        "selected_model": selected_name, "selection_rule": selection_rule,
        "selected_threshold": tr,
        "selected_confusion_stress": sel_cm,
        "selected_confusion_4x4": conf4_decision,
        "selected_country_metrics": country_rows,
        "selected_phase_metrics": phase_cm,
        "bootstrap": bootstrap,
        "threshold_families": families,
        "incumbent_preserved": {
            "M11_threshold": INCUMBENT_RQ01_M11_THRESHOLD,
            "M11_L": INCUMBENT_RQ01_M11_L,
            "status": "PRESERVED_NOT_OVERWRITTEN",
        },
        "canonical_parity": parity,
        "phase_chain": phase_claim,
        "requirements": status,
        "n_requirements_complete": sum(1 for s in status if s["status"].startswith("COMPLETE")),
        "unresolved": [s["id"] for s in status if not s["status"].startswith("COMPLETE")],
        "scientific_boundary": "No OOS selection; no owner certification; no production promotion.",
    }

    # CSV outputs
    pd.DataFrame(f_rows).to_csv(out / "RQ01_DEV_FOUR_M_METRICS.csv", index=False)
    for name, ev in evaluated.items():
        pd.DataFrame(ev["confusion4_argmax"]["matrix"],
                     index=CLASSES, columns=CLASSES).to_csv(
            out / f"RQ01_DEV_CONFUSION4_{name}.csv")
    pd.DataFrame(country_rows).to_csv(out / "RQ01_DEV_COUNTRY_STRESS.csv", index=False)
    pd.DataFrame(ind_rows).to_csv(out / "RQ01_DEV_INDIVIDUAL.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(out / "RQ01_DEV_JOINT_PAIRS.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(out / "RQ01_DEV_POOLS.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(out / "RQ01_DEV_PHASES.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(out / "RQ01_DEV_PHASE_TRANSITIONS.csv", index=False)
    pd.DataFrame(occurrence_rows).to_csv(out / "RQ01_DEV_PHASE_OCCURRENCE.csv", index=False)
    pd.DataFrame(f_rows).to_csv(out / "RQ01_DEV_F1_F5.csv", index=False)
    pd.DataFrame(tape).to_csv(out / "RQ01_DEV_ISSUE_TAPE.csv", index=False)
    # Canonical IJP PRE tape (additive; does not replace the phase controller).
    from .ijp_pre import pre_tape as _ijp_pre_tape
    results["ijp_pre"] = _ijp_pre_tape(prep["raw"], panel, out)
    # Canonical REV352/REV353 six-phase tape (additive; legacy phase tape remains).
    from .ijp_phase import phase_tape as _ijp_phase_tape
    results["ijp_phase"] = _ijp_phase_tape(prep["raw"], panel, out)
    (out / "RQ01_DEV_THRESHOLD_REGISTRY.json").write_text(
        json.dumps({"version": "REV363_RQ01_DEV_THRESHOLD_REGISTRY",
                    "selected_model": selected_name,
                    "selected_threshold": tr,
                    "families": families,
                    "incumbent_preserved": results["incumbent_preserved"],
                    "no_oos_tuning": True, "status": "EXPLORATORY_DEV_FROZEN_AFTER_FIT"},
                   indent=2, default=str) + "\n", encoding="utf-8")
    (out / "RQ01_DEV_CANONICAL_PARITY.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    (out / "RQ01_DEV_RESULTS.json").write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "RQ01_DEV_REQUIREMENTS_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # freeze artifact
    freeze = None
    if write_freeze:
        from .config import FREEZE_DIR, FREEZE_FILE, FREEZE_VERSION
        FREEZE_DIR.mkdir(parents=True, exist_ok=True)
        freeze = {
            "freeze_version": FREEZE_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(prep["root"]),
            "dev": {"start": DEV_FIT_START, "end": DEV_END, "n_keys": len(panel),
                    "class_counts": counts, "keys_sha256": sha256_json(
                        [[r["country"], r["month"]] for _, r in panel.iterrows()])},
            "feature_spec": spec.to_dict(),
            "models": MOD.serialize_models(models),
            "selected_model": selected_name,
            "selected_threshold": tr,
            "incumbent_preserved": results["incumbent_preserved"],
            "threshold_families": {k: (v.__dict__ if isinstance(v, THR.ThresholdResult) else v)
                                   for k, v in families.items()},
            "dev_metrics": {"stress_confusion": sel_cm, "bootstrap": bootstrap,
                            "F1_F5": {n: e["F1_F5"] for n, e in evaluated.items()}},
            "requirements_complete": results["n_requirements_complete"],
            "oos": "SEALED_UNTIL_THIS_FREEZE_IS_WRITTEN",
        }
        freeze["freeze_sha256"] = sha256_json(freeze)
        target = Path(freeze_path) if freeze_path else FREEZE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(freeze, indent=2, default=str) + "\n", encoding="utf-8")
        results["freeze_path"] = str(target)
        results["freeze_sha256"] = freeze["freeze_sha256"]
        (out / "RQ01_DEV_FREEZE.json").write_text(
            json.dumps(freeze, indent=2, default=str) + "\n", encoding="utf-8")

    # execution evidence
    evidence = {
        "run_id": results["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m rev363.rq01 --dataset-root <locked> --out <RESULTS/RQ01>",
        "dataset_root": str(prep["root"]),
        "code_version": "REV363", "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "n_rows": len(panel), "class_counts": counts,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "stress_confusion": sel_cm,
        "gate_feasible": sel_cm["FNR"] < 0.05 and sel_cm["FPR"] < 0.15,
        "oos_read": False, "oos_status": "SEALED",
        "outputs_sha256": {p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.glob("RQ01_DEV_*")) if p.is_file()},
    }
    (out / "EXECUTION_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    r = run_dev(a.dataset_root, a.out)
    print(json.dumps({k: r[k] for k in ("run_id", "selected_model", "selected_confusion_stress",
                                       "n_requirements_complete", "unresolved")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()

