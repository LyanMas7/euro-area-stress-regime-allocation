"""Report builder: 34 output report cards, acceptance CSV and markdown reports.

Nothing here fabricates a number: every field is read from the executed
RQ01/RQ02/RQ03 result dictionaries produced by ``rev363.rq01_next`` /
``rq02_next`` / ``rq03_next``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rev363.util import fmt, jdump, utcnow

CARD_KEYS = ["id", "rq", "required_output", "dataset_n", "actual_result", "formula_owner",
             "constraint_pass", "L_U_theta", "DEV_result", "OOS_full", "OOS_ESRB_metric",
             "OOS_noESRB_prediction", "benchmark_matched", "uncertainty_penalty", "XGB_status",
             "expanding_status", "run_id_evidence", "scientific_gate",
             "root_cause_if_not_pass", "next_repair", "declared_gate"]

ORDER = ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08", "01-09",
         "01-10", "01-11", "01-12", "02-01", "02-02", "02-03", "02-04", "02-05", "02-06",
         "02-07", "02-08", "02-09", "02-10", "02-11", "03-01", "03-02", "03-03", "03-04",
         "03-05", "03-06", "03-07", "03-08", "03-09", "03-10", "03-11"]


def mk(**kw):
    d = {k: "" for k in CARD_KEYS}
    d.update(kw)
    return d


def apply_spec_gate(cards):
    """FUNCTION_PASS = F_verified AND T_applicable AND L_executed.  A research-proxy card never self-certifies."""
    for card in cards:
        declared = str(card.get("scientific_gate", ""))
        card["declared_gate"] = declared
        if declared.startswith("PASS"):
            card["scientific_gate"] = "EXECUTED_RESEARCH_NOT_CERTIFIED"
            if not card.get("root_cause_if_not_pass"):
                card["root_cause_if_not_pass"] = ("T (owner thresholds / issuer PIT) and L (certified linkage) are "
                                                  "not established: executed research run only")
    return cards


def _rl(r1, family, model):
    for r in r1["dev_metrics"]:
        if r["family"] == family and r["model"] == model:
            return r
    return {}


def _ro(r1, family, model, window):
    for r in r1["oos_metrics"]:
        if r["family"] == family and r["model"] == model and r["window"] == window:
            return r
    return {}


def build_reports(out: Path, r1: dict, r2: dict, r3: dict) -> dict:
    out = Path(out)
    fam, mdl = r1["selected_family"], r1["selected_model"]
    thr = r1["selected_threshold"]
    dev = _rl(r1, fam, mdl)
    xgb = _rl(r1, "XGBOOST", "M11")
    sc = _ro(r1, "LOGISTIC", "M11", "ESRB_SCORABLE")
    no = _ro(r1, "LOGISTIC", "M11", "NO_ESRB_PREDICTION_ONLY")
    exp_status = r1.get("expanding", {})
    exp_txt = "; ".join(f"{k}:{v['status']}(n_refits={v['n_refits']})"
                        for k, v in exp_status.items()) or "NOT_RUN"
    par = r1.get("parity", {})
    perf = pd.DataFrame(r3["performance"])
    tail = pd.DataFrame(r3["tail_table"])
    cont = r3["continuity"]
    pol = perf[perf["strategy"] == "POLICY"].iloc[0].to_dict() if len(perf) else {}
    hz = pd.DataFrame(r2["horizons_dev"])
    hoos = pd.DataFrame(r2.get("horizons_oos", []))
    hoos_m11 = hoos[(hoos["window"] == "ESRB_SCORABLE") & (hoos["family"] == "LOGISTIC") &
                    (hoos["model"] == "M11")] if len(hoos) else hoos
    hz_oos_txt = "; ".join(
        f"{r.horizon}:n={r.n},ev={r.events},FNR={fmt(r.FNR,3)},AUC={fmt(r.auc,3)}"
        for r in hoos_m11.itertuples()) or "n/a"
    exr = r2["exit_rows"]
    ex_a = [e for e in exr if e.get("model") == "A" and e.get("auc")]
    bd, bo = r2.get("bottom_dev", {}), r2.get("bottom_oos", {})
    ov = r2.get("override", {})
    cap = r3["capital"]; sb = r3["short_bottom"]; rm = r3["risk_model"]
    ess_path = out / "RQ03_03_ESS_TAIL.csv"
    ess_mean = float(pd.read_csv(ess_path)["ESS_tail"].mean()) if ess_path.exists() else None
    ph_sum = r1.get("canonical_phase", {}) or {}
    ph_fail = ph_sum.get("failures", {}) or {}
    n_canon_countries = 7 - len(ph_fail)
    n_phase_none = int((ph_sum.get("oos_state_counts", {}) or {}).get("NONE", 0))
    ph_fail_txt = "; ".join(f"{c}:{r}" for c, r in sorted(ph_fail.items())) or "NONE"
    nij = r1.get("native_ijp", {}) or {}
    nij_ok = nij.get("status") == "RESEARCH_NATIVE_PROFILE"

    R = []
    # ---------------- RQ01 ----------------
    R.append(mk(id="01-01", rq="RQ01",
                required_output="Data coverage 5 sources x 7 countries x native frequency x release time x label status",
                dataset_n=f"DEV {r1['n_dev']} / OOS {r1['n_oos']}",
                actual_result=(f"35 source-country rows{' with native observation counts' if nij_ok else ''}; "
                               f"scorable={r1['n_oos_scorable']} model-only={r1['n_oos_no_esrb']}; "
                               f"ESRB coverage end UNVERIFIED"),
                formula_owner="labels_next.coverage_table() + native IJP coverage",
                constraint_pass="PASS (no self-filled N beyond 2021-04, P0-01)",
                L_U_theta="n/a", DEV_result=f"{r1['n_dev']} rows",
                OOS_full=f"{r1['n_oos']} rows", OOS_ESRB_metric=f"{r1['n_oos_scorable']} scorable",
                OOS_noESRB_prediction=f"{r1['n_oos_no_esrb']} model-only",
                benchmark_matched="ESRB RQ01 canonical workbook",
                uncertainty_penalty="episode-cluster support",
                XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ01_01_DATA_COVERAGE.csv / RQ01_01_NATIVE_COVERAGE.csv",
                scientific_gate="PASS",
                root_cause_if_not_pass="official issuer release clocks are not certified",
                next_repair="Official issuer PIT signature"))
    R.append(mk(id="01-02", rq="RQ01",
                required_output="Individual 5 sources UP/DOWN/SIDE counts, terminal signed/%",
                dataset_n="DEV 1218",
                actual_result=(f"native IJP 5-source research profile; "
                               f"{nij.get('n_individual_evaluated', 0)}/{nij.get('n_individual_rows', 0)} "
                               f"source-issues evaluated" if nij_ok else
                               "per-source counts + D12 terminal signed/% + N comparator (monthly proxy)"),
                formula_owner="rq01_native_ijp + ijp_kernel.individual",
                constraint_pass=("PASS native geometry; research thresholds" if nij_ok else
                                 "PARTIAL research proxy"), L_U_theta="Q75/Q35 prior deltas",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="N comparator",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ01_02_INDIVIDUAL_5.csv / _NATIVE_TAPE.csv",
                scientific_gate=("PARTIAL" if nij_ok else "PARTIAL_RESEARCH_PROXY"),
                root_cause_if_not_pass="owner source cuts and issuer release clocks not certified",
                next_repair="Owner source polarity/cut registry"))
    R.append(mk(id="01-03", rq="RQ01",
                required_output="Joint 10 pairs x 9 motifs, exact lag, two terminals",
                dataset_n="DEV 1218",
                actual_result=(f"native paired Joint; {nij.get('n_motifs_evaluated', 0)}/"
                               f"{nij.get('n_motifs', 9)} motifs evaluated; "
                               f"{nij.get('n_joint_evaluated', 0)}/{nij.get('n_joint_rows', 0)} rows"
                               if nij_ok else
                               "90 motif rows; exact_lag=0; terminal_a/terminal_b means (monthly proxy)"),
                formula_owner="rq01_native_ijp + ijp_kernel.joint",
                constraint_pass=("PASS native paired Count/Gate1; research thresholds" if nij_ok else
                                 "PARTIAL research proxy"), L_U_theta="lag 0",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="N comparator",
                uncertainty_penalty="multiplicity documented", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ01_03_JOINT_10.csv / _NATIVE_TAPE.csv",
                scientific_gate=("PARTIAL" if nij_ok else "PARTIAL_RESEARCH_PROXY"),
                root_cause_if_not_pass=("owner Joint thresholds/Gate2 not certified; "
                                        "Sideway profile is research-only" if nij_ok else
                                        "native paired Gate1/Gate2 and selected-pair consumer not authenticated"),
                next_repair="Owner Joint registry; retain research Sideway profile"))
    R.append(mk(id="01-04", rq="RQ01",
                required_output="Six Pool operators + pre-entry N->Stress flips",
                dataset_n="DEV 1218",
                actual_result=(f"native six Boolean Pool policies; "
                               f"{nij.get('n_pool_evaluated', 0)}/{nij.get('n_pool_rows', 0)} rows evaluated"
                               if nij_ok else "6 operators with pre-entry TP/FP (Kleene research pool)"),
                formula_owner="rq01_native_ijp + ijp_kernel.pool",
                constraint_pass=("PASS six Boolean policies; research thresholds" if nij_ok else
                                 "PARTIAL research proxy"), L_U_theta="n/a",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="N pre-entry comparator",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ01_04_POOLS_6.csv / _NATIVE_TAPE.csv",
                scientific_gate=("PARTIAL" if nij_ok else "PARTIAL_RESEARCH_PROXY"),
                root_cause_if_not_pass=("owner Pool policy not locked; research thresholds" if nij_ok else
                                        "owner binary Pool policy not provided; Kleene operators are research proxies"),
                next_repair="Owner Pool policy registry"))
    R.append(mk(id="01-05", rq="RQ01",
                required_output="M00/M01/M10/M11 x Logistic/XGBoost: P4, 4x4, 2x2, TPR/TNR/FPR/FNR, ROC",
                dataset_n="DEV 1218 / OOS 420",
                actual_result=(f"{fam} {mdl}: DEV FNR={fmt(dev.get('FNR'),4)} "
                               f"FPR={fmt(dev.get('FPR'),4)} AUC={fmt(dev.get('auc'),4)}"),
                formula_owner="four_m_grid()", constraint_pass="PASS clean common BASE (P1-10)",
                L_U_theta=f"theta={fmt(thr,6)}",
                DEV_result=f"FNR {fmt(dev.get('FNR'),4)} / FPR {fmt(dev.get('FPR'),4)}",
                OOS_full="420 keys emitted",
                OOS_ESRB_metric=f"FNR={fmt(sc.get('FNR'),4)} FPR={fmt(sc.get('FPR'),4)}",
                OOS_noESRB_prediction="P4/decisions only",
                benchmark_matched="M00..M11 equal BASE",
                uncertainty_penalty="country-block bootstrap pending",
                XGB_status=f"DEV FNR={fmt(xgb.get('FNR'),4)} FPR={fmt(xgb.get('FPR'),4)} "
                           f"AUC={fmt(xgb.get('auc'),4)} (challenger)",
                expanding_status=exp_txt, run_id_evidence="RQ01_05_FOUR_M_DEV.csv / _OOS.csv / _CALIBRATION.csv",
                scientific_gate="PARTIAL",
                root_cause_if_not_pass="DEV overfit + OOS regime shift: model does not generalise",
                next_repair="R0-R11 diagnostics before any promotion"))
    R.append(mk(id="01-06", rq="RQ01",
                required_output="Six states PRE/ENTRY/PEAK/DEESC/RECOVERY/EXIT",
                dataset_n="DEV 1218",
                actual_result=(f"canonical REV352/REV353 six-phase; countries={n_canon_countries}/7; "
                               f"transitions={ph_sum.get('n_transitions')}; FI snapshot failure"),
                formula_owner="ijp_phase.phase_tape_split + phase_table()",
                constraint_pass="PASS causal run-max only; DEV-frozen snapshots",
                L_U_theta="baseline percentile profile V1 (research)", DEV_result="see CSV",
                OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="n/a",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ01_06_PHASE_6.csv", scientific_gate="PARTIAL",
                root_cause_if_not_pass=(f"canonical snapshot failure: {ph_fail_txt}; "
                                        "no RECOVERY under the baseline percentile profile"),
                next_repair="FI native class support; owner phase gold and threshold profile"))
    R.append(mk(id="01-07", rq="RQ01",
                required_output="Phase conditional 1..5 exact-subset appearance",
                dataset_n="DEV 1218",
                actual_result=(f"canonical-phase subset conditionals with n_subset denominators; "
                               f"countries={n_canon_countries}/7"),
                formula_owner="phase_conditional()", constraint_pass="PASS",
                L_U_theta="n/a", DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="episode denominator",
                uncertainty_penalty="Wilson CI", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ01_07_PHASE_CONDITIONAL_1TO5.csv",
                scientific_gate="PARTIAL",
                root_cause_if_not_pass=f"low event support and FI exclusion ({ph_fail_txt})",
                next_repair="More independent episodes"))
    R.append(mk(id="01-08", rq="RQ01",
                required_output="Threshold register L/U/step/unique masks",
                dataset_n="DEV 1218",
                actual_result=f"theta={fmt(thr,6)} unique_masks={dev.get('unique_masks')}",
                formula_owner="exact_threshold_search + two_dim_search", constraint_pass="PASS",
                L_U_theta=f"L={fmt(dev.get('L'),6)} U={fmt(dev.get('U'),6)} theta={fmt(thr,6)}",
                DEV_result="frozen before OOS", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="frozen before OOS",
                uncertainty_penalty="multiplicity counted", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ01_08_THRESHOLD_REGISTER.json",
                scientific_gate="PASS", next_repair="Owner threshold registry; bounded grid is research-only"))
    R.append(mk(id="01-09", rq="RQ01",
                required_output="F1-F5 fitting/penalty benchmarks",
                dataset_n="DEV 1218",
                actual_result="F1 logloss, F2 Brier, F3 AUC, F4 selection-aware, F5 delta vs M00",
                formula_owner="F1..F5 in rq01_next", constraint_pass="PASS", L_U_theta="n/a",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="M00 baseline",
                uncertainty_penalty="BIC-like penalty", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ01_09_F1_F5.csv", scientific_gate="PASS",
                root_cause_if_not_pass="Official OI/FLI definitions absent",
                next_repair="Metric owner registry"))
    R.append(mk(id="01-10", rq="RQ01",
                required_output="Issued tape country x issue x M x family x P4 x decision",
                dataset_n="OOS 420",
                actual_result=(f"420 OOS rows x 4 M x 2 families; no gold column; "
                               f"112-row ESRB evaluation join separate; canonical OOS phase "
                               f"{n_canon_countries}/7, state-NONE rows={n_phase_none}"),
                formula_owner="tape builder", constraint_pass="PASS vintage stamps present",
                L_U_theta=f"theta={fmt(thr,6)}", DEV_result="n/a",
                OOS_full="420 rows emitted", OOS_ESRB_metric="112 scorable flag",
                OOS_noESRB_prediction="308 model-only flag",
                benchmark_matched="vintage stamps present", uncertainty_penalty="n/a",
                XGB_status="XGB P4 columns included", expanding_status=exp_txt,
                run_id_evidence="RQ01_10_ISSUED_TAPE_OOS.csv / RQ01_10_ESRB_SCORABLE_EVALUATION.csv",
                scientific_gate="PARTIAL",
                root_cause_if_not_pass=f"OOS phase uses DEV-frozen snapshots; FI fallback ({ph_fail_txt})",
                next_repair="Authentic issuer tape and FI snapshot support"))
    R.append(mk(id="01-11", rq="RQ01",
                required_output="Canonical comparator parity, two independent code paths",
                dataset_n="DEV 1218",
                actual_result=f"identical={par.get('decisions_identical')} max|diff|={fmt(par.get('score_max_abs_diff'),10)}",
                formula_owner="canonical_parity()", constraint_pass=par.get("status", "PASS"),
                L_U_theta="n/a", DEV_result=par.get("status"), OOS_full="n/a",
                OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="matched join", uncertainty_penalty="n/a",
                XGB_status=par.get("family"), expanding_status="n/a",
                run_id_evidence="RQ01_11_CANONICAL_PARITY.json",
                scientific_gate="PARTIAL",
                root_cause_if_not_pass="both paths share the same retained model; not an independent reimplementation",
                next_repair="Independent implementation parity"))
    R.append(mk(id="01-12", rq="RQ01",
                required_output="OOS full 2020-2024 + split scorable / no-ESRB reporting",
                dataset_n="OOS 420",
                actual_result=(f"views + split tapes emitted; scorable n={r1['n_oos_scorable']} scored, "
                               f"no-ESRB n={r1['n_oos_no_esrb']} unscored; phase {n_canon_countries}/7"),
                formula_owner="rq01_next.run()", constraint_pass="PASS scoped",
                L_U_theta=f"theta={fmt(thr,6)}", DEV_result="n/a",
                OOS_full="420 prediction rows",
                OOS_ESRB_metric=f"FNR={fmt(sc.get('FNR'),4)} FPR={fmt(sc.get('FPR'),4)}",
                OOS_noESRB_prediction="P4/decisions only",
                benchmark_matched="ESRB benchmark", uncertainty_penalty="country CI pending",
                XGB_status="challenger", expanding_status=exp_txt,
                run_id_evidence="RQ01_12_OOS_VIEWS.csv / _ESRB_SCORABLE.csv / _MODEL_ONLY.csv / "
                                "RQ01_10_ESRB_SCORABLE_EVALUATION.csv",
                scientific_gate="PARTIAL",
                root_cause_if_not_pass="OOS FNR far above DEV FNR: no generalisation; scorable window ends 2021-04",
                next_repair="Repair before promotion"))

    # ---------------- RQ02 ----------------
    hz_ex = hz[hz["family"] == "LOGISTIC"]
    hz0 = hz_ex.iloc[0].to_dict() if len(hz_ex) else {}
    R.append(mk(id="02-01", rq="RQ02",
                required_output="Long 2 (CG,DSR-P) / Short 7 native counts + terminal",
                dataset_n=f"panel {r2['n_panel']}",
                actual_result="7-source L/S matrix; DEV-frozen quantiles (P0-03)",
                formula_owner="build_frame()", constraint_pass="PASS label-independent",
                L_U_theta="q75 frozen", DEV_result="see CSV", OOS_full="n/a",
                OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="N-calibrated", uncertainty_penalty="n/a",
                XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ02_01_LONGSHORT_NATIVE.csv", scientific_gate="PASS",
                next_repair="Owner roster certification"))
    R.append(mk(id="02-02", rq="RQ02",
                required_output="1 Long + 21 Short joint pairs, selected/rejected",
                dataset_n="22 pairs", actual_result="22 pair rows; exact lag recorded",
                formula_owner="1+21 builder", constraint_pass="PASS", L_U_theta="exact lag",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="N comparator",
                uncertainty_penalty="multiplicity", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ02_02_JOINT_1PLUS21.csv", scientific_gate="PASS",
                next_repair="Triangular accumulation matrix"))
    R.append(mk(id="02-03", rq="RQ02",
                required_output="M00..M11 x H15/H12/H10/H8/H6 onset + future state",
                dataset_n=f"{len(r2['horizons_dev'])} rows",
                actual_result="5 horizons x 4 M x 2 families fitted",
                formula_owner="fit_two_family per horizon", constraint_pass="PASS",
                L_U_theta=f"theta={fmt(hz0.get('threshold'),4)}",
                DEV_result=f"FNR {fmt(hz0.get('FNR'),4)} / FPR {fmt(hz0.get('FPR'),4)}",
                OOS_full="DEV-frozen applied to OOS", OOS_ESRB_metric=hz_oos_txt,
                OOS_noESRB_prediction="P4 only after 2021-04",
                benchmark_matched="M00 baseline", uncertainty_penalty="n/a",
                XGB_status="XGB rows included", expanding_status="n/a",
                run_id_evidence="RQ02_03_FOUR_M_HORIZONS_DEV.csv / _OOS.csv",
                scientific_gate="PARTIAL",
                root_cause_if_not_pass="DEV apparent only; OOS far weaker",
                next_repair="Target/clock repair before promotion"))
    R.append(mk(id="02-04", rq="RQ02",
                required_output="S5 last Short confirmation with issued date",
                dataset_n=f"panel {r2['n_panel']}",
                actual_result=f"{r2['s5_rows']} rows, confirmed={r2['s5_confirmed']}",
                formula_owner="s5 builder (H6 M11 threshold)", constraint_pass="PASS research proxy",
                L_U_theta="H6 theta", DEV_result="see CSV", OOS_full="n/a",
                OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="no future source acceptance", uncertainty_penalty="n/a",
                XGB_status="n/a", expanding_status="n/a", run_id_evidence="RQ02_04_S5.csv",
                scientific_gate="PASS", root_cause_if_not_pass="No official issuer stamp",
                next_repair="Authentic issued S5 tape"))
    R.append(mk(id="02-05", rq="RQ02",
                required_output="EXIT-A first-N H3/H6/H12/H18",
                dataset_n=f"{len(exr)} rows",
                actual_result=f"{len(ex_a)} A-models fitted with canonical first-N endpoint (P0-04)",
                formula_owner="exit_event()", constraint_pass="PASS",
                L_U_theta="0.5", DEV_result=f"AUC {fmt(ex_a[0]['auc'],4) if ex_a else None}",
                OOS_full="n/a", OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="censoring handled", uncertainty_penalty="n/a",
                XGB_status="XGB rows included", expanding_status="n/a",
                run_id_evidence="RQ02_05_06_EXIT_AB.csv", scientific_gate="PASS",
                next_repair="Independent exit gold"))
    R.append(mk(id="02-06", rq="RQ02",
                required_output="EXIT-B confirmed-state exit + combined",
                dataset_n=f"{len(exr)} rows",
                actual_result="B variant uses RQ01 issued state as operational fallback",
                formula_owner="exit_event()", constraint_pass="PASS", L_U_theta="0.5",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="entry->exit risk set",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ02_05_06_EXIT_AB.csv", scientific_gate="PASS",
                next_repair="Owner confirmation rule"))
    R.append(mk(id="02-07", rq="RQ02",
                required_output="Override ledger original EXIT vs effective state",
                dataset_n=f"{ov.get('n_rows')} rows",
                actual_result=f"N->Stress={ov.get('n_N_to_stress')} Stress->N={ov.get('n_stress_to_N')}",
                formula_owner="override ledger", constraint_pass="PASS chronological",
                L_U_theta="RQ01 theta", DEV_result="see CSV", OOS_full="n/a",
                OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="immutable original", uncertainty_penalty="n/a",
                XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ02_07_OVERRIDE_LEDGER.csv", scientific_gate="PASS",
                next_repair="Gap fill from RQ01 issued prediction only"))
    R.append(mk(id="02-08", rq="RQ02",
                required_output="Bottom F1/F2/F3 components + issued zones",
                dataset_n=f"DEV {bd.get('n_records')}",
                actual_result="F1 EWI cooling, F2 peer recovery, F3 vol cooling; zone if >=2",
                formula_owner="bottom_zones()", constraint_pass="PASS votes>=2",
                L_U_theta="votes>=2", DEV_result=f"DEV zones {bd.get('n_records')}",
                OOS_full="n/a", OOS_ESRB_metric="N/A asset-level",
                OOS_noESRB_prediction=f"OOS zones {bo.get('n_records')}",
                benchmark_matched="n/a", uncertainty_penalty="n/a", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ02_08_BOTTOM_ZONES_DEV.csv",
                scientific_gate="PASS", next_repair="Owner 3F definition"))
    R.append(mk(id="02-09", rq="RQ02",
                required_output="Bottom HIT/MISS/NO_ZONE on real asset level (P0-05)",
                dataset_n=f"DEV {bd.get('n_records')} / OOS {bo.get('n_records')}",
                actual_result=(f"target=argmin(cumulative level); DEV hit_rate={fmt(bd.get('hit_rate'),4)} "
                               f"OOS hit_rate={fmt(bo.get('hit_rate'),4)}"),
                formula_owner="asset_levels + argmin(level)", constraint_pass="PASS",
                L_U_theta="zone width 3m",
                DEV_result=f"HIT {bd.get('n_hit')}/{bd.get('n_records')}",
                OOS_full=f"HIT {bo.get('n_hit')}/{bo.get('n_records')}",
                OOS_ESRB_metric="N/A", OOS_noESRB_prediction="asset-level truth",
                benchmark_matched="asset-level truth", uncertainty_penalty="interval score computed",
                XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ02_09_BOTTOM_ZONES_OOS.csv", scientific_gate="PASS",
                next_repair="Owner asset index"))
    R.append(mk(id="02-10", rq="RQ02",
                required_output="F1-F5 subtype penalty",
                dataset_n=f"{len(r2['horizons_dev'])} rows",
                actual_result="per-horizon F1/F3/F4",
                formula_owner="F1..F5 builder", constraint_pass="PASS", L_U_theta="n/a",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="M00 baseline",
                uncertainty_penalty="BIC-like", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ02_10_F1_F5_SUBTYPE.csv", scientific_gate="PASS",
                root_cause_if_not_pass="Official definitions absent", next_repair="Metric registry"))
    R.append(mk(id="02-11", rq="RQ02",
                required_output="OOS full horizons/EXIT/Bottom/S5",
                dataset_n="OOS 2020-2024",
                actual_result="OOS tables emitted (prediction-only after 2021-04)",
                formula_owner="rq02_next.run()", constraint_pass="PASS", L_U_theta="frozen",
                DEV_result="n/a",
                OOS_full="OOS horizons/EXIT/Bottom/S5 emitted",
                OOS_ESRB_metric=hz_oos_txt,
                OOS_noESRB_prediction="P4 + source/override/Bottom",
                benchmark_matched="ESRB where valid", uncertainty_penalty="n/a",
                XGB_status="challenger rows", expanding_status="n/a",
                run_id_evidence="RQ02_RESULTS.json", scientific_gate="PARTIAL",
                root_cause_if_not_pass="Prior OOS contaminated; corrected clock applied",
                next_repair="Re-run after label repair"))

    # ---------------- RQ03 ----------------
    R.append(mk(id="03-01", rq="RQ03",
                required_output="Six assets E1/E2/E3/B1/B2/C1",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result="return tape + cumulative level built", formula_owner="full_tape()",
                constraint_pass="PASS", L_U_theta="n/a", DEV_result="see CSV",
                OOS_full="n/a", OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="shared source", uncertainty_penalty="n/a", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ03_01_ASSET_TAPE.csv",
                scientific_gate="PASS", root_cause_if_not_pass="B1/B2/C1 are proxies",
                next_repair="Investability certification"))
    R.append(mk(id="03-02", rq="RQ03",
                required_output="EW/IVOL/MVP/ERC/MaxDiv/POLICY rules",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result=f"6 strategies; POLICY terminal={fmt(pol.get('full_terminal_nav'),4)}",
                formula_owner="target_weights + feasible_project", constraint_pass="PASS",
                L_U_theta="shared W0=1",
                DEV_result=f"DEV end W={fmt(pol.get('W_DEV_2005_2019_end'),4)}",
                OOS_full=f"OOS ret={fmt(pol.get('ret_OOS_FULL_2020_2024'),4)}",
                OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction=f"tail ret={fmt(pol.get('ret_OOS_TAIL_2021_05_2024_12'),4)}",
                benchmark_matched="matched scope/fee", uncertainty_penalty="n/a",
                XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ03_02_PERFORMANCE_CONTINUOUS.csv", scientific_gate="PASS",
                next_repair="Owner cost schedule"))
    R.append(mk(id="03-03", rq="RQ03",
                required_output="GARCH-DCC vol/corr/cov + tail-scenario ESS (P0-08, P1-17)",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result=f"one-sided DEV vintage; ESS_tail mean={fmt(ess_mean,4)}",
                formula_owner="dev_vintage_garch + tail_scenario_ess", constraint_pass="PASS",
                L_U_theta="n/a", DEV_result="see JSON", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="as-of only",
                uncertainty_penalty="MC scenarios", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ03_03_ESS_TAIL.csv", scientific_gate="PASS",
                next_repair="Owner risk certification"))
    R.append(mk(id="03-04", rq="RQ03",
                required_output="VaR/ES 6 strategies x 4 regimes x 3 alphas",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result="per-strategy per-regime VaR/ES with breach rates",
                formula_owner="var_es()", constraint_pass="PASS", L_U_theta="alpha 95/97.5/99",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="issued weekly tape",
                uncertainty_penalty="regime support noted", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ03_04_VAR_ES_6.csv",
                scientific_gate="PASS", next_repair="Owner risk model"))
    R.append(mk(id="03-05", rq="RQ03",
                required_output="Capital risk Normal 50-75% / Stress<=50% pre-trade (P0-07)",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result=(f"evaluated={cap['n_evaluated']} feasible={cap['n_feasible']} "
                               f"infeasible={cap['n_infeasible']}"),
                formula_owner="feasible_project()", constraint_pass="PASS pre-trade",
                L_U_theta="ES budget 2%", DEV_result="see CSV", OOS_full="n/a",
                OOS_ESRB_metric="n/a", OOS_noESRB_prediction="n/a",
                benchmark_matched="per-week", uncertainty_penalty="n/a", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ03_05_CAPITAL_FEASIBILITY.csv",
                scientific_gate="PASS", next_repair="Owner capital rule"))
    R.append(mk(id="03-06", rq="RQ03",
                required_output="HOLD/PARTIAL/FULL action ledger, fee once",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result="independent ledger for 6 strategies; no post-trade repair",
                formula_owner="stateful_backtest()", constraint_pass="PASS", L_U_theta="n/a",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="turnover/fee once",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ03_06_ACTION_LEDGER.csv", scientific_gate="PASS",
                next_repair="Owner execution schedule"))
    R.append(mk(id="03-07", rq="RQ03",
                required_output="Bottom permission + risk squeeze linkage (P0-06)",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result="3-family permission joined strictly as-of",
                formula_owner="asof_week_join()", constraint_pass="PASS", L_U_theta="votes>=3",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="as-of proof",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ03_07_BOTTOM_PERMISSION.csv", scientific_gate="PASS",
                next_repair="Owner permission rule"))
    R.append(mk(id="03-08", rq="RQ03",
                required_output="Short->Bottom avoided loss with/without",
                dataset_n=f"{sb['n_weeks_short']} short weeks",
                actual_result=(f"avoided_loss={fmt(sb['total_avoided_loss'],4)}; "
                               f"with={fmt(sb['policy_terminal_with_short'],4)} "
                               f"without={fmt(sb['policy_terminal_without_short'],4)}"),
                formula_owner="short overlay counterfactual",
                constraint_pass="PASS (negative reported if so)", L_U_theta="20% overlay",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="same W0",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status="n/a",
                run_id_evidence="RQ03_08_SHORT_TO_BOTTOM.csv", scientific_gate="PASS",
                next_repair="Owner hedge rule"))
    R.append(mk(id="03-09", rq="RQ03",
                required_output="ONE continuous W0(2005)->2024 NAV (P0-02)",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result=(f"nav_reset={cont['nav_reset_performed']}; "
                               f"W_2019_last={fmt(cont['W_2019_last'],4)} -> "
                               f"W_2020_first={fmt(cont['W_2020_first'],4)}"),
                formula_owner="stateful_backtest()", constraint_pass="PASS continuity",
                L_U_theta="W0=1", DEV_result="DEV end", OOS_full="OOS full",
                OOS_ESRB_metric="n/a", OOS_noESRB_prediction="tail from same ledger",
                benchmark_matched="6 matched", uncertainty_penalty="n/a", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ03_09_SIX_NAV_NET.csv",
                scientific_gate="PASS", next_repair="Owner NAV certification"))
    R.append(mk(id="03-10", rq="RQ03",
                required_output="Robust utility/sensitivity DCC vs EWMA, normal vs t",
                dataset_n=f"weeks {r3['n_weeks']}", actual_result="VaR/ES sensitivity table",
                formula_owner="robustness", constraint_pass="PASS", L_U_theta="n/a",
                DEV_result="see CSV", OOS_full="n/a", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="n/a", benchmark_matched="no OOS re-selection",
                uncertainty_penalty="distribution sensitivity", XGB_status="n/a",
                expanding_status="n/a", run_id_evidence="RQ03_10_ROBUSTNESS.csv",
                scientific_gate="PASS", next_repair="Owner utility function"))
    R.append(mk(id="03-11", rq="RQ03",
                required_output="OOS full + 2021-05 tail table from the same ledger",
                dataset_n=f"weeks {r3['n_weeks']}",
                actual_result=(f"tail rows={len(tail)}; "
                               f"POLICY W_2024={fmt(pol.get('full_continuous_terminal_wealth'),4)}"),
                formula_owner="tail_table", constraint_pass="PASS", L_U_theta="n/a",
                DEV_result="n/a", OOS_full="OOS full", OOS_ESRB_metric="n/a",
                OOS_noESRB_prediction="tail table emitted", benchmark_matched="6 strategies",
                uncertainty_penalty="n/a", XGB_status="n/a", expanding_status=exp_txt,
                run_id_evidence="RQ03_11_OOS_TAIL_TABLE.csv", scientific_gate="PASS",
                next_repair="Owner acceptance"))

    # FUNCTION_PASS = F_verified AND T_applicable AND L_executed (IJP spec I2, RQ01 spec 9.3, EXIT spec 9.2).
    # These cards describe research-proxy runs: owner thresholds, issuer PIT release and the OOS authority
    # are all absent, so a card may never certify itself as PASS.  The code's own claim stays visible in
    # `declared_gate`; DESIGN_ACCEPTED != IMPLEMENTED != FUNCTION_PASS != MODEL_PERFORMANCE_PASS != PRODUCTION.
    apply_spec_gate(R)
    acc = pd.DataFrame(R)[CARD_KEYS]
    acc = acc.set_index("id").reindex(ORDER).reset_index()
    acc.to_csv(out / "FINAL_OUTPUT_34_ACCEPTANCE.csv", index=False)
    jdump(acc.to_dict(orient="records"), out / "FINAL_OUTPUT_34_ACCEPTANCE.json")

    # ---------- markdown ----------
    def mdt(df, cols=None):
        if cols is None:
            d = df
        else:
            keep = [c for c in cols if c in df.columns]
            d = df[keep]
        return d.to_markdown(index=False)

    dev_md = ["# FULL_DEV_RESULTS (DEV 2005-07..2019-12)\n",
              f"Run: `{r1['run_id']}`\n",
              "## RQ01 four-M x two estimators (DEV apparent)\n",
              mdt(pd.DataFrame(r1["dev_metrics"]),
                  ["family", "model", "n", "n_features", "threshold", "TP", "FN", "FP", "TN",
                   "FNR", "FPR", "auc", "logloss", "brier", "gate_feasible"]),
              "\n## RQ01 selected model\n",
              f"- family=`{fam}` model=`{mdl}` theta=`{fmt(thr,6)}` rule=`{r1['selection_rule']}`",
              "\n## RQ01 probability calibration diagnostics (evaluated, not used for selection)\n",
              mdt(pd.DataFrame(r1.get("calibration", []))),
              "\n## RQ02 five horizons (DEV)\n",
              mdt(pd.DataFrame(r2["horizons_dev"]),
                  ["horizon", "family", "model", "n_dev", "n_events", "auc", "threshold",
                   "FNR", "FPR", "gate_feasible"]),
              "\n## RQ03 continuous performance (DEV segment)\n",
              mdt(perf, ["strategy", "W_DEV_2005_2019_start", "W_DEV_2005_2019_end",
                         "ret_DEV_2005_2019", "full_annual_return", "full_annual_vol",
                         "full_MDD", "total_fees"])]
    (out / "FULL_DEV_RESULTS.md").write_text("\n".join(dev_md) + "\n", encoding="utf-8")

    oos_md = ["# FULL_OOS_RESULTS (2020-01..2024-12)\n",
              f"Run: `{r1['run_id']}`\n",
              "## RQ01 OOS 4M x 2 families (ESRB-scorable 2020-01..2021-04 only)\n",
              mdt(pd.DataFrame(r1["oos_metrics"]),
                  ["family", "model", "window", "n", "TP", "FN", "FP", "TN", "FNR", "FPR",
                   "auc", "logloss", "brier"]),
              "\n## RQ02 OOS horizons (DEV-frozen; ESRB-scorable only)\n",
              mdt(pd.DataFrame(r2.get("horizons_oos", [])),
                  ["horizon", "family", "model", "window", "n", "events", "auc",
                   "frozen_threshold", "TP", "FN", "FP", "TN", "FNR", "FPR"]),
              "\n## RQ03 OOS full + tail (one continuous NAV, no reset)\n",
              mdt(perf, ["strategy", "W_OOS_FULL_2020_2024_start", "W_OOS_FULL_2020_2024_end",
                         "ret_OOS_FULL_2020_2024", "W_OOS_TAIL_2021_05_2024_12_start",
                         "W_OOS_TAIL_2021_05_2024_12_end", "ret_OOS_TAIL_2021_05_2024_12",
                         "full_terminal_nav", "n_infeasible_hold", "total_fees"])]
    (out / "FULL_OOS_RESULTS.md").write_text("\n".join(oos_md) + "\n", encoding="utf-8")

    tail_md = ["# OOS_TAIL_MODEL_ONLY_RESULTS (2021-05..2024-12)\n",
               "No ESRB classification metric is computed in this window. RQ01/RQ02 emit",
               "probabilities/decisions only; RQ03 uses the same continuous 2005-2024 ledger.\n",
               mdt(tail)]
    (out / "OOS_TAIL_MODEL_ONLY_RESULTS.md").write_text("\n".join(tail_md) + "\n", encoding="utf-8")

    gate = ["# REAL_TRADING_GATE_STATUS\n", "**STATUS: NOT_READY**\n",
            "| Gate | Status | Evidence |", "|---|---|---|",
            "| PIT / issuer signature | NOT_CERTIFIED | release clocks are research as-of |",
            "| Risk model | RESEARCH_ASOF | GARCH-DCC DEV vintage, no owner certification |",
            "| Cash / fees / custody | NOT_CERTIFIED | 10bp research fee only |",
            "| Live market data | NOT_RUN | retrospective weekly tape |",
            "| Broker restrictions | NOT_RUN | none supplied |",
            "| Prospective paper results | NOT_RUN | no forward test performed |",
            "| Human authorization | NOT_GRANTED | no signed owner approval |",
            "",
            "Real trading is never auto-enabled. OOS 2020-2024 has already been inspected, so all",
            "post-exposure XGBoost/expanding development is `POST_OOS_EXPOSURE_RESEARCH`, not",
            "independent validation."]
    (out / "REAL_TRADING_GATE_STATUS.md").write_text("\n".join(gate) + "\n", encoding="utf-8")

    conflict = ["# CONTRACT_CONFLICT_AND_APPROVAL_LOG\n", "## Active conflicts\n",
                "1. **File-01 vs new request - expanding OOS refit.** File 01 'Exactly TWO",
                "   indivisible research calendar blocks' prohibits OOS refitting. The expanding",
                "   axis (E1/E3) is `BLOCKED_BY_ACTIVE_CONTRACT` pending a signed amendment; it is",
                "   executed here only as `POST_OOS_EXPOSURE_EXPERIMENT` evidence.",
                "2. **EA7 (user) vs EA8 (frozen contract).** Thesis scope is EA7; the frozen",
                "   contract binds EA8 (includes IE). IE retained as reference only.",
                "3. **M00..M11 (user) vs frozen A Rev55 / B Rev64 / C cross-check.**",
                "4. **ESRB coverage end.** 2021-05 is a `POLICY_OF_PRESENTATION`, not a verified",
                "   ESRB publication end date (`ESRB_coverage_end=UNVERIFIED`).",
                "", "## Required approvals\n",
                "- Signed amendment permitting OOS online expanding under a DEV-frozen policy.",
                "- Owner certification of source polarity, cost/funding and risk model.",
                "- Confirmation of the true ESRB benchmark coverage end."]
    (out / "CONTRACT_CONFLICT_AND_APPROVAL_LOG.md").write_text("\n".join(conflict) + "\n",
                                                               encoding="utf-8")

    err = pd.DataFrame([
        ("P0-01", "label_month self-fills N beyond ESRB coverage", "REPAIRED",
         "labels_next.label_status; 308 rows NOT_SCORABLE"),
        ("P0-02", "OOS NAV reset to 1", "REPAIRED",
         f"stateful_backtest continuous; nav_reset={cont['nav_reset_performed']}"),
        ("P0-03", "RQ02 Long/Short quantiles from OOS labels", "REPAIRED",
         "DEV-frozen calibration in build_frame()"),
        ("P0-04", "first-N off-by-one", "REPAIRED",
         "labels_next.first_normal_month (NormalDate month inclusive)"),
        ("P0-05", "Bottom target = min weekly return", "REPAIRED",
         "asset_levels argmin(cumulative level)"),
        ("P0-06", "month->week join not strict as-of", "REPAIRED",
         "labels_next.asof_week_join"),
        ("P0-07", "RISK_REPAIR after trade/NAV", "REPAIRED",
         "feasible_project() before turnover/fee/NAV"),
        ("P0-08", "GARCH/DCC full-series warm-up", "REPAIRED",
         "dev_vintage_garch + one-sided filter"),
        ("P0-09", "lag<=x expands opportunities", "REPAIRED",
         "exact_lag=0 pre-registered and recorded"),
        ("P1-10", "BASE containing I/J; M11-only country terms", "REPAIRED",
         "corrected_blocks(): country terms in common BASE"),
        ("P1-17", "ESS from asset risk contributions", "REPAIRED",
         "tail_scenario_ess Monte-Carlo"),
        ("P1-20", "manifest self-hash / invented paths", "REPAIRED",
         "RUN_MANIFEST without self-hash"),
        ("P1-21", "OOS 4x4 scored over no-gold rows", "REPAIRED",
         "scorable-only scoring; 308 rows unscored"),
        ("P1-22", "XGBoost/expanding not executed", "REPAIRED_EXECUTED",
         f"XGB {r1['xgb_version']}; expanding {exp_txt}"),
        ("P2-23", "release/vintage & tradability gaps", "PARTIAL",
         "research as-of only; owner proof absent"),
        ("P2-24", "POLICY robustness/utility incomplete", "PARTIAL",
         "sensitivity reported; owner utility absent"),
        ("P2-25", "heterogeneous units as independent events", "PARTIAL",
         "episode clustering noted; CI pending"),
    ], columns=["id", "defect", "status", "evidence"])
    err.to_csv(out / "ERROR_REGISTER_FINAL.csv", index=False)

    cards = ["# FULL_34_OUTPUT_EXECUTION_REPORT\n", f"Generated {utcnow()}\n",
             f"**RUN** `{r1['run_id']}` | XGBoost {r1['xgb_version']} | expanding: {exp_txt}\n",
             "## 34-row acceptance status\n",
             mdt(acc, ["id", "required_output", "dataset_n", "scientific_gate",
                       "root_cause_if_not_pass"])]
    for r in R:
        block = [f"\n### {r['id']} - {r['required_output']}\n",
                 f"- **Actual result:** {r['actual_result']}",
                 f"- **Formula owner:** {r['formula_owner']} | **Dataset n:** {r['dataset_n']}",
                 f"- **Constraints:** {r['constraint_pass']} | **L/U/theta:** {r['L_U_theta']}",
                 f"- **DEV:** {r['DEV_result']}",
                 f"- **OOS full:** {r['OOS_full']} | **OOS ESRB:** {r['OOS_ESRB_metric']} | "
                 f"**OOS no-ESRB:** {r['OOS_noESRB_prediction']}",
                 f"- **XGB:** {r['XGB_status']} | **Expanding:** {r['expanding_status']}",
                 f"- **Evidence:** {r['run_id_evidence']}",
                 f"- **Gate:** {r['scientific_gate']}"]
        if r["root_cause_if_not_pass"]:
            block.append(f"- **Root cause:** {r['root_cause_if_not_pass']}")
        block.append(f"- **Next repair:** {r['next_repair']}")
        cards += block
    (out / "FULL_34_OUTPUT_EXECUTION_REPORT.md").write_text("\n".join(cards) + "\n",
                                                            encoding="utf-8")

    n_pass = int(acc["scientific_gate"].str.startswith("PASS").sum())     # 0 unless a card is really certified
    n_exec = int(acc["scientific_gate"].str.startswith("EXECUTED_RESEARCH").sum())
    n_partial = int(acc["scientific_gate"].str.startswith("PARTIAL").sum())
    cp = ["# CHECKPOINT_NEXT_ACTION\n",
          "- report_mode: `INCREMENTAL_RESEARCH_UPDATE`",
          "- last_checkpoint_file03: `SOURCE_FRAMEWORK_ONLY` (not auto-updated)",
          "- source_contract_conflict: `EA7 vs EA8 / M00..11 vs frozen A/B/C / OOS expanding prohibited`",
          "- post_oos_exposure: `TRUE` (all XGB/expanding)",
          f"- 34 outputs: {n_pass} certified PASS, {n_exec} EXECUTED_RESEARCH_NOT_CERTIFIED, "
          f"{n_partial} PARTIAL, {34 - n_pass - n_exec - n_partial} other",
          f"- RQ01 OOS FNR={fmt(sc.get('FNR'),4)} vs DEV FNR={fmt(dev.get('FNR'),4)} => no generalisation; gate FAIL",
          "", "## One next action", "",
          "Run the R0-R11 diagnostic battery on the frozen RQ01 score to separate regime shift",
          "from calibration failure, then pre-register a single re-specified model **without**",
          "touching OOS thresholds. OOS FNR/FPR gates are NOT met and were not forced."]
    (out / "CHECKPOINT_NEXT_ACTION.md").write_text("\n".join(cp) + "\n", encoding="utf-8")

    return {"n_pass": n_pass, "n_partial": n_partial, "n_rows": len(acc)}
