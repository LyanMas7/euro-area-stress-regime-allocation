"""RQ01 corrected DEV/OOS execution: 12 required outputs (IDs 01-01 .. 01-12).

Everything is computed from the frozen DEV fit; the ESRB-scorable OOS window
(2020-01..2021-04) is the only place where classification metrics are reported.
The 2021-05..2024-12 window emits model probabilities/decisions only
(labelled ``MODEL_ONLY``) and is never scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rev363 import features as F
from rev363 import metrics as MET
from rev363 import thresholds as THR
from rev363.config import CLASSES, EA7, MOTIFS, PAIRS10, SOURCES5, SOURCES7
from rev363 import estimators as EST
from rev363 import expanding as EXP
from rev363 import panel as PNL
from rev363.labels_next import (ESRB_SCORABLE_END, STATUS_SCORABLE,
                                  coverage_table, first_normal_month, all_onset_months)
from rev363.util import jdump, run_id, sha256_json, utcnow
from pathlib import Path

DEV_FIT_START, DEV_END = "2005-07", "2019-12"
OOS_END = "2024-12"
NATIVE_FREQUENCY = {"CISS": "weekly", "CLIFS": "monthly", "DFC": "quarterly",
                    "GFC": "monthly", "VIX": "daily"}
RELEASE_TIME_RULE = {"CISS": "WEEKLY_ASOF_RESEARCH", "CLIFS": "MONTHLY_ASOF_RESEARCH",
                     "DFC": "QUARTERLY_ASOF_RESEARCH", "GFC": "MONTHLY_ASOF_RESEARCH",
                     "VIX": "DAILY_CLOSE_ASOF_RESEARCH"}


# ---------------------------------------------------------------------------
# small tables
# ---------------------------------------------------------------------------
def data_coverage(episodes, panel) -> pd.DataFrame:
    cov = coverage_table(episodes).set_index("country", drop=False)
    dev = panel[(panel["month"] >= DEV_FIT_START) & (panel["month"] <= DEV_END)]
    rows = []
    for c in EA7:
        base = cov.loc[c].to_dict()
        sub = dev[dev["country"] == c]
        for s in SOURCES5:
            v = sub[f"vote_{s}"].astype(str)
            rows.append({
                "country": c, "source": s,
                "native_frequency": NATIVE_FREQUENCY[s],
                "release_time_rule": RELEASE_TIME_RULE[s],
                "n_dev_rows": int(len(sub)),
                "n_label_N": int((sub["label"] == "N").sum()),
                "n_label_stress": int((sub["label"] != "N").sum()),
                # movement-state codes: U = UP, D = DOWN, S = SIDEWAY, M = missing (not votes)
                "n_state_UP": int((v == "U").sum()), "n_state_DOWN": int((v == "D").sum()),
                "n_state_SIDEWAY": int((v == "S").sum()), "n_state_MISSING": int((v == "M").sum()),
                "n_episodes_eligible": base["n_episodes_eligible"],
                "onset_months": base["onset_months"],
                "first_normal_months": base["first_normal_months"],
                "dev_scorable": base["dev_scorable"],
                "oos_esrb_scorable_window": base["oos_esrb_scorable_window"],
                "oos_no_esrb_window": base["oos_no_esrb_window"],
                "esrb_coverage_end": base["esrb_coverage_end"],
                "label_status_policy": base["label_status_policy"],
                "release_rules_status": "RESEARCH_ASOF_NOT_OFFICIAL_ISSUER_PIT",
            })
    return pd.DataFrame(rows)


def individual_table(panel: pd.DataFrame) -> pd.DataFrame:
    y = (panel["label"] != "N").astype(int).values
    rows = []
    for s in SOURCES5:
        v = panel[f"vote_{s}"].astype(str)
        # proxy Individual = source state UP (letter U); a missing level (M) is NOT_EVALUATED and is
        # excluded from the confusion matrix instead of being scored as a negative prediction
        ev = (v != "M").values
        pred = (v == "U").astype(int).values
        c = MET.confusion_binary(y[ev], pred[ev])
        up = panel.loc[v == "U", f"d12_{s}"]
        dn = panel.loc[v == "D", f"d12_{s}"]
        first = panel.loc[v == "U", f"z_{s}"]
        rows.append({
            "source": s, "native_frequency": NATIVE_FREQUENCY[s],
            "count_up": int((v == "U").sum()), "count_down": int((v == "D").sum()),
            "count_side": int((v == "S").sum()), "count_missing": int((v == "M").sum()),
            "n_evaluated": int(ev.sum()), "n_not_evaluated": int((~ev).sum()),
            "terminal_signed_mean_up": float(up.mean()) if len(up) else None,
            "terminal_signed_mean_down": float(dn.mean()) if len(dn) else None,
            "terminal_pct_mean_up": (float(up.mean() * 100.0) if len(up) else None),
            "n_comparator_N": int((y == 0).sum()),
            "stress_TP": c["TP"], "stress_FP": c["FP"], "stress_FN": c["FN"], "stress_TN": c["TN"],
            "stress_FNR": c["FNR"], "stress_FPR": c["FPR"],
        })
    return pd.DataFrame(rows)


def joint_table(panel: pd.DataFrame, spec) -> pd.DataFrame:
    y = (panel["label"] != "N").astype(int).values
    ind = panel["individual_vote"].values
    rows = []
    for p in PAIRS10:
        a, b = p
        for motif in MOTIFS:
            mask = (panel[f"pair_{a}_{b}_state"] == motif).values
            known = ~((panel[f"vote_{a}"] == "M").values | (panel[f"vote_{b}"] == "M").values)
            mask = mask & known
            rescued = int(((ind == "FAIL") & mask & (y == 1)).sum())
            newfp = int(((ind == "FAIL") & mask & (y == 0)).sum())
            term_a = panel[f"d12_{a}"].where(mask)
            term_b = panel[f"d12_{b}"].where(mask)
            rows.append({"pair": f"{a}-{b}", "motif": motif, "exact_lag": 0,
                         "chosen": bool(motif == spec.selected_motifs.get(p)),
                         "n_known": int(known.sum()), "n_pass": int(mask.sum()),
                         "terminal_a_mean_pass": (float(term_a.mean()) if mask.any() else None),
                         "terminal_b_mean_pass": (float(term_b.mean()) if mask.any() else None),
                         "rescued_fn": rescued, "new_fp": newfp,
                         "n_comparator_N": int((y == 0).sum())})
    return pd.DataFrame(rows)


def pool_table(panel: pd.DataFrame) -> pd.DataFrame:
    y = (panel["label"] != "N").astype(int).values
    rows = []
    for op in ("IND_ONLY", "JOINT_ONLY", "IND_AND_JOINT", "IND_OR_JOINT",
               "COMPLEMENT_AND_JOINT", "COMPLEMENT_OR_JOINT"):
        v = panel[f"pool_{op}"]
        st = panel[f"pool_{op}_status"] if f"pool_{op}_status" in panel.columns else             pd.Series(np.where(v.isin(["PASS", "FAIL"]), "EXECUTED", "NOT_EVALUATED"), index=panel.index)
        ev = (st == "EXECUTED").values
        pred = (v == "PASS").astype(int).values
        c = MET.confusion_binary(y[ev], pred[ev])          # executed (binary) decisions only
        pre = panel[panel["phase"] == "PRE"]
        rows.append({"operator": op, "PASS": int((v == "PASS").sum()),
                     "FAIL": int((v == "FAIL").sum()), "N_NOT_EVALUATED": int((~ev).sum()),
                     "pre_entry_TP": int(((pre[f"pool_{op}"] == "PASS") & (pre["label"] != "N")).sum()),
                     "pre_entry_FP": int(((pre[f"pool_{op}"] == "PASS") & (pre["label"] == "N")).sum()),
                     **{k: c[k] for k in ("TP", "FP", "FN", "TN", "FNR", "FPR")}})
    return pd.DataFrame(rows)


def phase_table(panel: pd.DataFrame, *, phase_col: str = "phase", engine: str | None = None,
                status_by_country: dict | None = None) -> pd.DataFrame:
    rows = []
    # the composite-intensity controller is NOT the canonical REV352/REV353 chain: say so in the output itself
    engine = engine or (str(panel["phase_engine"].iloc[0])
                        if "phase_engine" in panel.columns and len(panel) else "UNTAGGED_NOT_CANONICAL")
    for c in EA7:
        sub = panel[panel["country"] == c]
        cnt = sub[phase_col].value_counts().to_dict()
        rows.append({"country": c, **{p: int(cnt.get(p, 0)) for p in
                                      ("PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT")},
                     "n_months": int(len(sub)),
                     "n_canonical_months": int(sub[phase_col].notna().sum()) if phase_col != "phase" else int(len(sub)),
                     "phase_engine": engine,
                     "canonical_status": (status_by_country or {}).get(c, "CANONICAL_OK")})
    return pd.DataFrame(rows)


def phase_conditional(panel: pd.DataFrame, *, phase_col: str = "phase") -> pd.DataFrame:
    """Exact-subset conditional appearance P(source j present | subset S present, phase)."""
    rows = []
    for phase in ("ENTRY", "PEAK", "DEESC", "RECOVERY"):
        sub = panel[panel[phase_col] == phase]
        n = len(sub)
        for s in SOURCES5:
            present = (sub[f"vote_{s}"] == "U").values
            rows.append({"phase": phase, "estimand": "marginal_appearance", "subset": s,
                         "n_phase": n, "n_subset": n, "n_event": int(present.sum()),
                         "probability": (float(present.mean()) if n else None),
                         "status": ("IDENTIFIABLE" if n >= 10 else
                                    "LOW_SUPPORT" if n else "NO_PHASE_SUPPORT"),
                         "ci95": MET.wilson_ci(int(present.sum()), n) if n else [None, None]})
    # conditionals for |S|=1..4 over the 5-source full set
    for k in (1, 2, 3, 4):
        for S in _subsets(SOURCES5, k):
            rest = [j for j in SOURCES5 if j not in S]
            for phase in ("PEAK", "DEESC"):
                sub = panel[panel[phase_col] == phase]
                m = np.ones(len(sub), bool)
                for i in S:
                    m &= (sub[f"vote_{i}"] == "U").values
                nS = int(m.sum())
                for j in rest:
                    cnt = int(((sub[f"vote_{j}"] == "U").values & m).sum())
                    rows.append({"phase": phase, "estimand": f"P({j}|{'&'.join(S)})",
                                 "subset": "&".join(S), "n_phase": len(sub),
                                 "n_subset": nS, "n_event": cnt,
                                 "probability": (cnt / nS if nS else None),
                                 "status": ("IDENTIFIABLE" if nS >= 10 else
                                            "LOW_SUPPORT" if nS else "NO_SUBSET_SUPPORT"),
                                 "ci95": MET.wilson_ci(cnt, nS) if nS else [None, None]})
    # |S|=5 -> joint appearance of all five
    for phase in ("PEAK", "DEESC"):
        sub = panel[panel[phase_col] == phase]
        m = np.ones(len(sub), bool)
        for i in SOURCES5:
            m &= (sub[f"vote_{i}"] == "U").values
        rows.append({"phase": phase, "estimand": "P(all_five)", "subset": "&".join(SOURCES5),
                     "n_phase": len(sub), "n_subset": len(sub), "n_event": int(m.sum()),
                     "probability": (float(m.mean()) if len(sub) else None),
                     "status": ("IDENTIFIABLE" if len(sub) >= 10 else
                                "LOW_SUPPORT" if len(sub) else "NO_PHASE_SUPPORT"),
                     "ci95": MET.wilson_ci(int(m.sum()), len(sub)) if len(sub) else [None, None]})
    return pd.DataFrame(rows)


def _subsets(items, k):
    from itertools import combinations
    return list(combinations(items, k))


# ---------------------------------------------------------------------------
# 4M x estimator x policy grid
# ---------------------------------------------------------------------------
def _score_confusion(y, score, thr):
    pred = (np.asarray(score) >= thr).astype(int)
    return MET.confusion_binary(y, pred), pred


def four_m_grid(panel_dev, panel_oos, spec, family: str, *,
                joint_columns=None, joint_source="PROXY_JOINT_COLUMNS",
                ind_columns=None, ind_source="SPEC_PROXY_IND_COLUMNS") -> dict:
    cols = PNL.corrected_blocks(spec)
    base = list(cols["M00"])
    ind = list(ind_columns if ind_columns is not None else cols["_blocks"]["IND"])
    joint = list(joint_columns if joint_columns is not None else cols["_blocks"]["JOINT"])
    blocks = {"M00": base, "M01": base + ind, "M10": base + joint,
              "M11": base + ind + joint}
    y4_dev = PNL.y4_from(panel_dev)
    y_dev = (y4_dev != 0).astype(int)
    out = {"family": family, "models": {}, "dev": {}, "oos": {},
           "columns": blocks, "joint_source": joint_source,
           "joint_columns": joint, "ind_source": ind_source, "ind_columns": ind}
    for mname in ("M00", "M01", "M10", "M11"):
        Xd = PNL.matrix_for(panel_dev, blocks[mname])
        Xo = PNL.matrix_for(panel_oos, blocks[mname])
        mdl = EST.fit_two_family(Xd, blocks[mname], y4_dev, family=family, name=mname)
        p_dev = mdl.p4(Xd)
        p_oos = mdl.p4(Xo)
        tr = THR.exact_threshold_search(1 - p_dev[:, 0], y_dev)
        out["models"][mname] = mdl
        cm, _ = _score_confusion(y_dev, 1 - p_dev[:, 0], tr.threshold)
        out["dev"][mname] = {
            "n": len(y_dev), "threshold": tr.__dict__, "confusion": cm,
            "confusion4_argmax": MET.confusion4(y4_dev, p_dev.argmax(axis=1)),
            "auc": MET.auc_score(y_dev, 1 - p_dev[:, 0]),
            "logloss": MET.multiclass_logloss(y4_dev, p_dev),
            "brier": MET.multiclass_brier(y4_dev, p_dev),
            "p4": p_dev.tolist(), "n_features": len(blocks[mname]),
            "feature_source": joint_source, "ind_source": ind_source,
        }
        out["oos"][mname] = {"p4": p_oos, "score": 1 - p_oos[:, 0],
                             "threshold": tr.threshold,
                             "feature_source": joint_source,
                             "joint_columns": joint}
    return out


def _scored_subset(panel_oos, y4_oos, p_oos, thr, mask):
    y = (y4_oos != 0).astype(int)
    cm, pred = _score_confusion(y[mask], (1 - p_oos[:, 0])[mask], thr)
    pred4 = np.where(pred == 0, 0, p_oos[mask][:, 1:].argmax(axis=1) + 1)
    return {"n": int(mask.sum()), "confusion": cm,
            "confusion4": MET.confusion4(y4_oos[mask], pred4),
            "auc": MET.auc_score(y[mask], (1 - p_oos[:, 0])[mask]),
            "logloss": (MET.multiclass_logloss(y4_oos[mask], p_oos[mask]) if mask.sum() else None),
            "brier": (MET.multiclass_brier(y4_oos[mask], p_oos[mask]) if mask.sum() else None)}


def run(dataset_root, out_dir, *, budget_fast=False, oos_authorization=None) -> dict:
    from rev363.w01_contract import enforce_execution
    enforce_execution(oos_requested=True, oos_authorization=oos_authorization)
    out = Path(out_dir)
    prep = PNL.build_panel(dataset_root, row_start="2004-01", row_end=OOS_END, data_end=OOS_END)
    panel = prep["panel"]
    spec = prep["spec"]
    episodes = prep["episodes"]

    dev_mask = (panel["month"] >= DEV_FIT_START) & (panel["month"] <= DEV_END)
    oos_mask = (panel["month"] >= "2020-01") & (panel["month"] <= OOS_END)
    scor_mask = oos_mask & (panel["label_status"] == STATUS_SCORABLE)
    noc_mask = oos_mask & (panel["label_status"] != STATUS_SCORABLE)

    panel_dev = panel[dev_mask].reset_index(drop=True)
    panel_oos = panel[oos_mask].reset_index(drop=True)
    scor_mask_oos = (panel_oos["label_status"] == STATUS_SCORABLE).values
    noc_mask_oos = ~scor_mask_oos

    # --- 01-01 coverage
    cov = data_coverage(episodes, panel)

    # --- 01-02 .. 01-04: native IJP research tape (monthly proxy is the fallback)
    native = None
    pkt_ind_cols: list = []
    ind_source = "SPEC_PROXY_IND_COLUMNS"
    native_oos = None
    native_summary = None
    native_result = {"status": "NOT_EVALUATED"}
    native_error = None
    pair_selection = None
    sel_public = None
    feature_dev = None
    feature_oos = None
    joint_cols = []
    selected_pool = None
    packet_mode = "NO_PACKET"
    ind_proxy = individual_table(panel_dev)
    jnt_proxy = joint_table(panel_dev, spec)
    pol_proxy = pool_table(panel_dev)
    if not budget_fast:
        try:
            from . import rq01_native_ijp as NIJ
            from . import rq01_pair_selection_run as RPS
            labels_dev = panel_dev[["country", "month", "label", "phase"]]
            # 1) full 10x9 probe tape: research evidence and DEV-only G0-G4 input.
            native_probe = NIJ.build_native_tape(dataset_root, start=DEV_FIT_START,
                                                 end=DEV_END)
            probe_summary = NIJ.summarize(native_probe, labels_dev)
            native_probe["joint_tape"].to_csv(out / "RQ01_03_JOINT_NATIVE_ALLPAIRS.csv",
                                             index=False)
            native_probe["pool_tape"].to_csv(out / "RQ01_04_POOL_NATIVE_ALLPAIRS.csv",
                                            index=False)
            probe_summary["joint"].to_csv(out / "RQ01_03_JOINT_10_ALLPAIRS.csv", index=False)
            probe_summary["pools"].to_csv(out / "RQ01_04_POOLS_6_ALLPAIRS.csv", index=False)
            # 2) registry decision on DEV only (no OOS row is passed to the selector).
            pair_selection = RPS.select_pairs(native_probe, panel_dev, out_dir=out)
            selected_configs = pair_selection.get("selected_configs", {})
            research_configs = pair_selection.get("research_configs", {})
            selected_pool = pair_selection.get("selection_pool_operator")
            # 3) issued DEV packet consumes exactly the frozen registry selection;
            #    the diagnostic feature packet consumes the declared research roster.
            native = NIJ.build_native_tape(
                dataset_root, start=DEV_FIT_START, end=DEV_END,
                selected_configs=selected_configs, research_configs=research_configs,
                selected_pool=selected_pool)
            native_summary = NIJ.summarize(native, labels_dev)
            native["individual_tape"].to_csv(out / "RQ01_02_INDIVIDUAL_NATIVE_TAPE.csv", index=False)
            native["joint_tape"].to_csv(out / "RQ01_03_JOINT_NATIVE_TAPE.csv", index=False)
            native["pool_tape"].to_csv(out / "RQ01_04_POOL_NATIVE_TAPE.csv", index=False)
            native["legacy_joint_tape"].to_csv(out / "RQ01_03_JOINT_NATIVE_LEGACY.csv", index=False)
            native["legacy_pool_tape"].to_csv(out / "RQ01_04_POOL_NATIVE_LEGACY.csv", index=False)
            native_summary["individual"].to_csv(out / "RQ01_02_INDIVIDUAL_5.csv", index=False)
            native_summary["joint"].to_csv(out / "RQ01_03_JOINT_10.csv", index=False)
            native_summary["pools"].to_csv(out / "RQ01_04_POOLS_6.csv", index=False)
            native_summary["not_evaluated_reasons"].to_csv(
                out / "RQ01_04_NOT_EVALUATED_REASON_LEDGER_DEV.csv", index=False)
            native["config_parity"].to_csv(
                out / "RQ01_03_CONFIG_SELECTION_CONSUMPTION_DEV.csv", index=False)
            native["packet_features"].to_csv(
                out / "RQ01_03_PACKET_FEATURES_DEV.csv", index=False)
            # 4) OOS packet is built with the same frozen configurations.  The
            #    registry is never re-estimated on OOS.
            native_oos = NIJ.build_native_tape(
                dataset_root, start="2020-01", end=OOS_END,
                selected_configs=selected_configs, research_configs=research_configs,
                selected_pool=selected_pool)
            native_oos["packet_features"].to_csv(
                out / "RQ01_03_PACKET_FEATURES_OOS.csv", index=False)
            native_oos["config_parity"].to_csv(
                out / "RQ01_03_CONFIG_SELECTION_CONSUMPTION_OOS.csv", index=False)
            native_oos["joint_tape"].to_csv(
                out / "RQ01_03_JOINT_NATIVE_TAPE_OOS.csv", index=False)
            native_oos["pool_tape"].to_csv(
                out / "RQ01_04_POOL_NATIVE_TAPE_OOS.csv", index=False)
            feature_dev = native["packet_features"]
            feature_oos = native_oos["packet_features"]
            # M10/M11 consume exactly the SELECTED packet features when a registry
            # selection exists; the 10-config research roster is the fallback only.
            selected_joint_cols = list(native["packet_feature_selected_columns"])
            research_joint_cols = list(native["packet_feature_joint_columns"])
            joint_cols = selected_joint_cols or research_joint_cols
            packet_mode = ("SELECTED_REGISTRY_PACKET" if selected_joint_cols
                           else "FULL_ROSTER_RESEARCH_PACKET")
            native_cov = NIJ.coverage(native["native_frame"])
            native_cov.to_csv(out / "RQ01_01_NATIVE_COVERAGE.csv", index=False)
            cov = cov.merge(native_cov, on=["country", "source"], how="left",
                            suffixes=("", "_native"))
            cov = cov.drop(columns=[c for c in ("native_frequency_native",
                                                "release_rules_status_native")
                                    if c in cov.columns])
            jdump(native["profile"], out / "RQ01_NATIVE_IJP_PROFILE.json")
            sel_public = ({k: v for k, v in pair_selection.items()
                           if k not in ("selected_configs", "research_configs")}
                          if pair_selection is not None else None)
            native_result = {
                "status": "RESEARCH_NATIVE_PROFILE",
                "n_individual_rows": int(len(native["individual_tape"])),
                "n_individual_evaluated": int((native["individual_tape"]["status"] == "EVALUATED").sum()),
                "n_joint_rows": int(len(native["joint_tape"])),
                "n_joint_evaluated": int((native["joint_tape"]["status"] == "EVALUATED").sum()),
                "n_joint_rows_oos": int(len(native_oos["joint_tape"])),
                "n_pool_rows": int(len(native["pool_tape"])),
                "n_pool_evaluated": int((native["pool_tape"]["status"] == "EVALUATED").sum()),
                "n_motifs": int(native_summary["joint"]["motif"].nunique()),
                "n_motifs_evaluated": int((native_summary["joint"]["n_evaluated"] > 0).sum()),
                "n_selected_configs": len(selected_configs),
                "n_research_feature_configs": len(research_configs),
                "selection_status": pair_selection.get("status"),
                "selection_pool_operator": selected_pool,
                "packet_feature_mode": packet_mode,
                "packet_feature_columns": joint_cols,
                "packet_selected_columns": selected_joint_cols,
                "packet_research_columns": research_joint_cols,
                "ind_block_source": ind_source,
                "packet_ind_columns": pkt_ind_cols,
                "config_parity_dev": native_summary["config_parity"],
                "config_parity_oos": (
                    native_oos["config_parity"]["status"]
                    == "PASS_SELECTED_EQUALS_CONSUMED_CONFIG_ID").sum(),
                "profile": native["profile"],
                "pair_selection": sel_public,
            }
        except Exception as exc:
            native_error = f"{type(exc).__name__}:{exc}"
    if native_summary is None:
        ind_proxy.to_csv(out / "RQ01_02_INDIVIDUAL_5.csv", index=False)
        jnt_proxy.to_csv(out / "RQ01_03_JOINT_10.csv", index=False)
        pol_proxy.to_csv(out / "RQ01_04_POOLS_6.csv", index=False)
        native_result = {"status": "PARTIAL_RESEARCH_PROXY", "reason": native_error}
    else:
        ind_proxy.to_csv(out / "RQ01_02_INDIVIDUAL_5_PROXY.csv", index=False)
        jnt_proxy.to_csv(out / "RQ01_03_JOINT_10_PROXY.csv", index=False)
        pol_proxy.to_csv(out / "RQ01_04_POOLS_6_PROXY.csv", index=False)
    cov.to_csv(out / "RQ01_01_DATA_COVERAGE.csv", index=False)

    # --- packet feature merge: exactly the issued packet, no proxy JOINT columns
    feature_source = "NO_PACKET_M10_EQUALS_M00"
    if feature_dev is not None and joint_cols:
        keep = ["country", "month"] + joint_cols + ["pkt_joint_not_evaluated_count", "pkt_pool_issued"]
        panel_dev = panel_dev.merge(feature_dev[keep], on=["country", "month"],
                                    how="left", validate="one_to_one")
        panel_oos = panel_oos.merge(feature_oos[keep], on=["country", "month"],
                                    how="left", validate="one_to_one")
        feature_source = packet_mode
    # --- native-packet Individual block: M01/M11 consume the SAME issued Individual
    #     records as the native tape (roadmap 01-02), not the proxy spec IND columns.
    if native is not None and len(native["individual_tape"]):
        it = native["individual_tape"]
        ev = it[it["status"] == "EVALUATED"][["country", "issue", "source", "vote"]].copy()
        piv = ev.pivot_table(index=["country", "issue"], columns="source",
                             values="vote", aggfunc="first")
        piv.columns = [f"pkt_ind_{c}" for c in piv.columns]
        pkt_ind_cols = sorted(piv.columns)
        feat_ind = piv.reset_index().rename(columns={"issue": "month"})
        for col in pkt_ind_cols:
            feat_ind[col] = pd.to_numeric(feat_ind[col], errors="coerce")
        feat_ind["pkt_ind_evaluated_count"] = feat_ind[pkt_ind_cols].notna().sum(axis=1)
        panel_dev = panel_dev.merge(feat_ind, on=["country", "month"], how="left",
                                    validate="one_to_one")
        panel_oos = panel_oos.merge(feat_ind, on=["country", "month"], how="left",
                                    validate="one_to_one")
        ind_source = "NATIVE_PACKET_INDIVIDUAL_VOTES"
        if native_result.get("status") == "RESEARCH_NATIVE_PROFILE":
            native_result["ind_block_source"] = ind_source
            native_result["packet_ind_columns"] = pkt_ind_cols
    if len(panel_dev) != int(dev_mask.sum()) or len(panel_oos) != int(oos_mask.sum()):
        raise AssertionError("PACKET_FEATURE_MERGE_CHANGED_PANEL_ROWS")

    # --- 01-06/07 phases: canonical REV352/REV353 tape where available.
    phase_summary = None
    phase_failures = {}
    try:
        from .ijp_phase import phase_tape_split as _canonical_phase_split
        phase_summary = _canonical_phase_split(prep["raw"], panel_dev, panel_oos, out)
        phase_failures = phase_summary.get("failures", {})
        canonical = pd.read_csv(out / "RQ01_DEV_CANONICAL_PHASE.csv")
        canonical["phase_canonical"] = canonical["state_after"].map(
            {"PRE_ONSET": "PRE", "ENTRY": "ENTRY", "PEAK": "PEAK",
             "DE_ESCALATION": "DEESC", "RECOVERY": "RECOVERY", "EXIT": "EXIT", "NONE": None})
        phase_view = panel_dev.merge(
            canonical[["country", "issue_at", "phase_canonical"]],
            left_on=["country", "month"], right_on=["country", "issue_at"], how="left")
        oos_canonical = pd.read_csv(out / "RQ01_OOS_CANONICAL_PHASE.csv")
        oos_canonical["phase_canonical"] = oos_canonical["state_after"].map(
            {"PRE_ONSET": "PRE", "ENTRY": "ENTRY", "PEAK": "PEAK",
             "DE_ESCALATION": "DEESC", "RECOVERY": "RECOVERY", "EXIT": "EXIT", "NONE": None})
        panel_oos = panel_oos.merge(
            oos_canonical[["country", "issue_at", "phase_canonical"]],
            left_on=["country", "month"], right_on=["country", "issue_at"], how="left")
        has_row = panel_oos["issue_at"].notna()
        # A missing canonical snapshot is NOT_EVALUATED, not the legacy monthly phase.  The
        # legacy value remains available in `phase_legacy` for diagnostics only;
        # it must never be issued as a confirmed canonical RECOVERY.
        panel_oos["phase_source_oos"] = np.where(
            ~has_row, "CANONICAL_MISSING_FAIL_CLOSED",
            np.where(panel_oos["phase_canonical"].notna(), "CANONICAL",
                     "CANONICAL_STATE_NONE"))
        panel_oos["phase_engine_oos"] = np.where(
            has_row, "CANONICAL_RQ01_SIX_PHASE_DEV_FROZEN_OOS",
            "NO_CANONICAL_SNAPSHOT_FAIL_CLOSED")
        panel_oos.loc[~has_row, "phase_canonical"] = None
        panel_oos["phase_not_evaluated_reason"] = np.where(
            panel_oos["phase_canonical"].notna(), None,
            np.where(has_row, "CANONICAL_STATE_NONE",
                     "NO_CANONICAL_SNAPSHOT_FAIL_CLOSED"))
        phase_view["phase_not_evaluated_reason"] = np.where(
            phase_view["phase_canonical"].notna(), None,
            "NO_CANONICAL_DEV_SNAPSHOT_FAIL_CLOSED")
        phase_summary["legacy_fallback_to_confirmed"] = False
        phase_engine = "CANONICAL_RQ01_SIX_PHASE_V1"
        phase_summary["engine"] = phase_engine
    except Exception as exc:
        phase_summary = {"status": "NOT_EVALUATED", "reason": f"{type(exc).__name__}:{exc}",
                         "engine": "NO_CANONICAL_PHASE_FAIL_CLOSED"}
        phase_view = panel_dev.copy()
        phase_view["phase_canonical"] = None
        phase_view["phase_not_evaluated_reason"] = "CANONICAL_ENGINE_FAILED_FAIL_CLOSED"
        panel_oos["phase_canonical"] = None
        panel_oos["phase_source_oos"] = "CANONICAL_FAILED_FAIL_CLOSED"
        panel_oos["phase_engine_oos"] = "NO_CANONICAL_PHASE_FAIL_CLOSED"
        panel_oos["phase_not_evaluated_reason"] = "CANONICAL_ENGINE_FAILED_FAIL_CLOSED"
        phase_engine = "NO_CANONICAL_PHASE_FAIL_CLOSED"
        phase_failures = {"ALL": str(exc)}
    ph = phase_table(phase_view, phase_col="phase_canonical", engine=phase_engine,
                     status_by_country={c: ("OK" if c not in phase_failures else phase_failures[c])
                                        for c in EA7})
    ph.to_csv(out / "RQ01_06_PHASE_6.csv", index=False)
    pc = phase_conditional(phase_view, phase_col="phase_canonical")
    pc.to_csv(out / "RQ01_07_PHASE_CONDITIONAL_1TO5.csv", index=False)
    # --- 01-06 FI class-support diagnostic (roadmap): count labelled native rows per
    #     source/class; an explicit NOT_IDENTIFIABLE route when a class is unsupported.
    fi_diag_rows = []
    try:
        from .ijp_s5 import native_rows_from_raw as _native_rows_from_raw
        for _c in EA7:
            _native = _native_rows_from_raw(prep["raw"], _c)
            _lookup = {str(r["month"]): r["label"] for _, r in
                       panel_dev[panel_dev["country"] == _c].iterrows()}
            for _src in ("CISS", "CLIFS", "DFC", "GFC", "VIX"):
                _n_n = _n_s = 0
                for _row in _native.get(_src, []):
                    _lab = _lookup.get(str(getattr(_row, "period_end", ""))[:7])
                    if _lab is None:
                        continue
                    if _lab == "N":
                        _n_n += 1
                    else:
                        _n_s += 1
                fi_diag_rows.append({
                    "country": _c, "source": _src,
                    "n_native_labelled_N": _n_n, "n_native_labelled_Stress": _n_s,
                    "class_support_route": ("SUPPORT_OK" if (_n_n > 0 and _n_s > 0)
                                            else "NOT_IDENTIFIABLE"),
                    "phase_status": phase_failures.get(_c, "OK"),
                    "note": ("Both N and Stress native references exist for the frozen baseline snapshot"
                             if (_n_n > 0 and _n_s > 0)
                             else "Missing class reference; source-phase cutoffs stay UNDEFINED and votes NOT_EVALUATED (no fabricated Recovery)")})
    except Exception as exc:
        fi_diag_rows = [{"country": "ALL", "source": "ALL", "class_support_route": "DIAGNOSTIC_FAILED",
                         "note": f"{type(exc).__name__}:{exc}"}]
    pd.DataFrame(fi_diag_rows).to_csv(out / "RQ01_PHASE_FI_DIAGNOSTIC.csv", index=False)
    episode_conditional = {"status": "NOT_EVALUATED"}
    try:
        from . import rq01_episode_conditional as EPC
        canonical_csv = out / "RQ01_DEV_CANONICAL_PHASE.csv"
        individual_csv = out / "RQ01_02_INDIVIDUAL_NATIVE_TAPE.csv"
        if canonical_csv.exists() and individual_csv.exists():
            episode_conditional = EPC.run(canonical_csv, individual_csv, out)
            episode_conditional["status"] = "COMPLETE_EPISODE_CONDITIONAL_RESEARCH"
        else:
            episode_conditional = {"status": "NOT_EVALUATED",
                                   "reason": "CANONICAL_PHASE_OR_NATIVE_INDIVIDUAL_MISSING"}
    except Exception as exc:
        episode_conditional = {"status": "NOT_EVALUATED",
                               "reason": f"{type(exc).__name__}:{exc}"}
    jdump(episode_conditional, out / "RQ01_07_EPISODE_CONDITIONAL_SUMMARY.json")

    # --- 01-05 four-M x two families
    grid = {}
    for fam in ("LOGISTIC", "XGBOOST"):
        grid[fam] = four_m_grid(panel_dev, panel_oos, spec, fam,
                                joint_columns=joint_cols,
                                joint_source=feature_source,
                                ind_columns=(pkt_ind_cols or None),
                                ind_source=ind_source)

    # --- 01-02 DEV-only challenger: proxy spec IND block vs native-packet IND block
    #     (pre-registered comparison on DEV; never used to select on OOS)
    ind_challenge_rows = []
    if pkt_ind_cols:
        for fam in ("LOGISTIC", "XGBOOST"):
            try:
                proxy = four_m_grid(panel_dev, panel_oos, spec, fam,
                                    joint_columns=joint_cols,
                                    joint_source=feature_source,
                                    ind_columns=None,
                                    ind_source="SPEC_PROXY_IND_COLUMNS")
                for mname in ("M01", "M11"):
                    for tag, g in (("PROXY_SPEC_IND", proxy), ("NATIVE_PACKET_IND", grid[fam])):
                        d = g["dev"][mname]
                        ind_challenge_rows.append({
                            "family": fam, "model": mname, "ind_block": tag,
                            "n_ind_columns": len(g["ind_columns"]),
                            "auc": d["auc"], "logloss": d["logloss"],
                            "brier": d["brier"], "threshold": d["threshold"]["threshold"],
                            "scope": "DEV_ONLY_PRE_REGISTERED"})
            except Exception as exc:
                ind_challenge_rows.append({"family": fam, "model": "M01/M11",
                                           "ind_block": "CHALLENGER_FAILED",
                                           "reason": f"{type(exc).__name__}:{exc}",
                                           "scope": "DEV_ONLY_PRE_REGISTERED"})
    pd.DataFrame(ind_challenge_rows).to_csv(
        out / "RQ01_05_IND_BLOCK_CHALLENGER_DEV.csv", index=False)

    dev_rows, oos_rows = [], []
    for fam, g in grid.items():
        for mname in ("M00", "M01", "M10", "M11"):
            d = g["dev"][mname]
            thr = d["threshold"]
            dev_rows.append({"family": fam, "model": mname, "update_policy": "DEV_FROZEN",
                             "feature_source": g["joint_source"],
                             "n_joint_columns": len(g["joint_columns"]),
                             "n": d["n"], "n_features": d["n_features"],
                             "threshold": thr["threshold"], "L": thr["L"], "U": thr["U"],
                             "unique_masks": thr["n_unique_masks"], "gate_feasible": thr["gate_feasible"],
                             "TP": d["confusion"]["TP"], "FN": d["confusion"]["FN"],
                             "FP": d["confusion"]["FP"], "TN": d["confusion"]["TN"],
                             "FNR": d["confusion"]["FNR"], "FPR": d["confusion"]["FPR"],
                             "TPR": d["confusion"]["TPR"], "TNR": d["confusion"]["TNR"],
                             "auc": d["auc"], "logloss": d["logloss"], "brier": d["brier"]})
            y4o = PNL.y4_from(panel_oos)
            sc = _scored_subset(panel_oos, y4o, g["oos"][mname]["p4"],
                                g["oos"][mname]["threshold"], scor_mask_oos)
            # Model-only rows are never scored: do not even compute metrics on their
            # unverified labels (the earlier code computed and then discarded them).
            no = {"n": int(noc_mask_oos.sum()), "confusion": None, "confusion4": None,
                  "auc": None, "logloss": None, "brier": None}
            oos_rows.append({"family": fam, "model": mname, "update_policy": "DEV_FROZEN",
                             "feature_source": g["joint_source"],
                             "window": "ESRB_SCORABLE", "n": sc["n"],
                             **{k: sc["confusion"][k] for k in ("TP", "FN", "FP", "TN", "FNR", "FPR")},
                             "auc": sc["auc"], "logloss": sc["logloss"], "brier": sc["brier"]})
            oos_rows.append({"family": fam, "model": mname, "update_policy": "DEV_FROZEN",
                             "feature_source": g["joint_source"],
                             "window": "NO_ESRB_PREDICTION_ONLY", "n": no["n"],
                             "TP": None, "FN": None, "FP": None, "TN": None,
                             "FNR": None, "FPR": None, "auc": None, "logloss": None, "brier": None})
    pd.DataFrame(dev_rows).to_csv(out / "RQ01_05_FOUR_M_DEV.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(out / "RQ01_05_FOUR_M_OOS.csv", index=False)

    # --- selection (DEV pre-registered: M11 first feasible logistic, else xgb)
    def violation(row):
        def rate(v):
            if v is None:
                return 1.0
            try:
                x = float(v)
            except (TypeError, ValueError):
                return 1.0
            return x if np.isfinite(x) else 1.0
        return (max(0.0, rate(row.get("FNR")) - 0.05) / 0.05 +
                max(0.0, rate(row.get("FPR")) - 0.15) / 0.15)
    cand = [r for r in dev_rows if r["family"] == "LOGISTIC"]
    feasible = [r for r in cand if r["gate_feasible"]]
    order = {"M11": 0, "M01": 1, "M10": 2, "M00": 3}
    if feasible:
        sel = sorted(feasible, key=lambda r: order[r["model"]])[0]
        sel_rule = "M11_FIRST_FEASIBLE_PRE_REGISTERED"
    else:
        sel = sorted(cand, key=lambda r: (violation(r), order[r["model"]]))[0]
        sel_rule = "NO_FEASIBLE_LOGISTIC_CLOSEST_VIOLATION"
    sel_family, sel_model = sel["family"], sel["model"]
    sel_thr = float(sel["threshold"])
    sel_cols = list(grid[sel_family]["columns"][sel_model])
    sel_mdl = grid[sel_family]["models"][sel_model]

    # --- 01-08 threshold register
    sel_p_dev = sel_mdl.p4(PNL.matrix_for(panel_dev, sel_cols))
    sel_score_dev = 1.0 - sel_p_dev[:, 0]
    two_dim = THR.two_dim_search(
        sel_score_dev,
        panel_dev["intensity"].fillna(panel_dev["intensity"].median()).values,
        (PNL.y4_from(panel_dev) != 0).astype(int))
    reg = {
        "selected_family": sel_family, "selected_model": sel_model,
        "selected_threshold": sel["threshold"], "selection_rule": sel_rule,
        "families": {"GLOBAL_SCORE_EXACT": sel["threshold"], "TWO_DIM_BOUNDED": two_dim},
        "unique_masks": sel["unique_masks"],
        "no_oos_tuning": True,
        "contract": "THRESHOLD_FROZEN_ON_DEV_BEFORE_OOS_OPENED",
        "owner_status": "RESEARCH_ONLY_NO_OWNER_THRESHOLD_REGISTRY",
        "owner_certified": False,
        "selected_threshold_owner": None,
        "threshold_profile_id": "RQ01_MOTIF_ISSUE_PACKET_RESEARCH_V1",
        "n_dev_rows": int(len(panel_dev)),
        "n_threshold_candidates_dev": int(sum(
            max(1, int(r.get("unique_masks") or 1)) for r in dev_rows)),
        "n_families": len({r["family"] for r in dev_rows}),
        "n_models": len(dev_rows),
        "selection_penalty_note": "8 M-models x 2 families share one DEV block; apparent DEV metrics are selection-biased and are not owner thresholds",
        "feature_source": feature_source,
        "packet_joint_columns": joint_cols,
    }
    jdump(reg, out / "RQ01_08_THRESHOLD_REGISTER.json")

    # --- 01-09 F1-F5 (research diagnostics only; no official owner definition)
    f_rows = []
    for r in dev_rows:
        k = r["n_features"]
        n = r["n"]
        f4 = (r["logloss"] + (k / (2.0 * max(n, 1))) * np.log(max(n, 2)) +
              np.log(max(r["unique_masks"], 1)) / max(n, 1))
        f_rows.append({"family": r["family"], "model": r["model"], "n": n,
                       "F1_logloss": r["logloss"], "F2_brier": r["brier"], "F3_auc": r["auc"],
                       "F4_selection_aware": f4, "F5_delta_logloss_vs_M00":
                       r["logloss"] - next(x["logloss"] for x in dev_rows
                                           if x["family"] == r["family"] and x["model"] == "M00"),
                       "status": "RESEARCH_DIAGNOSTIC_ONLY",
                       "official_owner_status": "NOT_DEFINED"})
    f_frame = pd.DataFrame(f_rows)
    f_frame.to_csv(out / "RQ01_09_F1_F5.csv", index=False)
    f_checks = {
        "F1_logloss_min": float(f_frame["F1_logloss"].min()),
        "F1_logloss_nonnegative": bool((f_frame["F1_logloss"] >= 0).all()),
        "F2_brier_in_0_2": bool(((f_frame["F2_brier"] >= 0) &
                                (f_frame["F2_brier"] <= 2)).all()),
        "F3_auc_in_0_1": bool(((f_frame["F3_auc"] >= 0) &
                              (f_frame["F3_auc"] <= 1)).all()),
        "F4_selection_aware_finite": bool(np.isfinite(f_frame["F4_selection_aware"]).all()),
        "F5_M00_identity_zero": bool(all(
            abs(x) < 1e-15 for x in f_frame.loc[f_frame.model == "M00",
                                                "F5_delta_logloss_vs_M00"])),
        "official_owner_status": "NOT_DEFINED",
        "claim": "RESEARCH_DIAGNOSTIC_ONLY_NOT_OWNER_F1_F5",
        "status": "PASS_RESEARCH_DEFINITION_CHECK",
    }
    jdump(f_checks, out / "RQ01_09_F1_F5_VERIFICATION.json")
    # Two separated namespaces (roadmap 01-09): research diagnostics vs official F1-F5.
    f_authority = {
        "RQ01_RESEARCH_F1F5": {
            "status": "EXECUTED_RESEARCH_ONLY",
            "estimands": {"F1": "multiclass logloss (proper loss)", "F2": "multiclass Brier",
                          "F3": "binary Stress-vs-N AUC", "F4": "selection-aware penalty heuristic",
                          "F5": "logloss delta vs M00"},
            "population": "DEV 1,218 country-months; apparent (fitted) metrics only",
            "evidence": "RQ01_09_F1_F5.csv / RQ01_09_F1_F5_VERIFICATION.json"},
        "RQ01_OFFICIAL_F1F5": {
            "status": "NOT_DEFINED",
            "reason": ("OFFICIAL_F1_F5_OWNER_DEFINITIONS_NOT_AUTHENTICATED; no signed formulas, "
                       "loss orientation, thresholds, data/units, dependence correction or "
                       "effective date were supplied by the owner"),
            "acceptance": "BLOCKED_OWNER_DEFINITION"},
        "claim": "RESEARCH_DIAGNOSTICS_ARE_NOT_OWNER_F1_F5",
    }
    jdump(f_authority, out / "RQ01_F1_F5_AUTHORITY_LEDGER.json")

    # --- 01-11: numerical parity (same model) and independent matched ablation
    Xs = sel_mdl.matrix(PNL.matrix_for(panel_dev, sel_cols))
    p_a = sel_mdl.p4(PNL.matrix_for(panel_dev, sel_cols))
    Xd = PNL.matrix_for(panel_dev, sel_cols)
    score_a = 1 - p_a[:, 0]
    if sel_family == "LOGISTIC":
        z = Xs @ sel_mdl.coef.T + sel_mdl.intercept[None, :]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z); p_b = e / e.sum(axis=1, keepdims=True)
    else:
        p_b = sel_mdl.p4(Xd)
    parity = {"family": sel_family, "model": sel_model, "n": len(Xd),
              "score_max_abs_diff": float(np.max(np.abs(score_a - (1 - p_b[:, 0])))),
              "decisions_identical": bool(np.array_equal(
                  (score_a >= sel_thr).astype(int), (1 - p_b[:, 0] >= sel_thr).astype(int))),
              "decision_sha256": sha256_json((score_a >= sel_thr).astype(int).tolist())}
    parity["status"] = "PASS" if parity["decisions_identical"] else "FAIL"
    jdump(parity, out / "RQ01_11_CANONICAL_PARITY.json")
    from . import rq01_independent_11 as IND11
    independent = IND11.run(
        panel_dev, panel_oos, spec,
        {"model": sel_model, "family": sel_family, "cols": sel_cols,
         "threshold": sel_thr, "p4_oos": grid[sel_family]["oos"][sel_model]["p4"]},
        packet_dev=feature_dev, packet_oos=feature_oos,
        joint_cols=joint_cols, out_dir=out)

    # --- 01-10 issued tape: full OOS 420 rows, no retrospective gold
    panel_oos = panel_oos.reset_index(drop=True)
    tape = pd.DataFrame({
        "country": panel_oos["country"], "issue": panel_oos["month"],
        "phase": panel_oos["phase_canonical"],
        "phase_legacy": panel_oos["phase"],
        "phase_engine": panel_oos["phase_engine_oos"],
        "phase_source": panel_oos["phase_source_oos"],
        "phase_not_evaluated_reason": panel_oos.get("phase_not_evaluated_reason"),
        "individual_vote": panel_oos["individual_vote"],
        "individual_vote_status": panel_oos.get("individual_vote_status"),
        "pool_count": panel_oos["pool_count"],
        "pkt_joint_not_evaluated_count": panel_oos.get("pkt_joint_not_evaluated_count"),
        "pkt_pool_issued": panel_oos.get("pkt_pool_issued"),
    })
    tape["packet_feature_source"] = feature_source
    tape["selection_pool_operator"] = (native_result.get("selection_pool_operator")
                                       if native_result else None)
    tape["n_selected_configs"] = (native_result.get("n_selected_configs")
                                  if native_result else None)
    for fam in ("LOGISTIC", "XGBOOST"):
        for mname in ("M00", "M01", "M10", "M11"):
            p4 = np.asarray(grid[fam]["oos"][mname]["p4"], float)
            score = np.asarray(grid[fam]["oos"][mname]["score"], float)
            thr = float(grid[fam]["oos"][mname]["threshold"])
            pred = (score >= thr).astype(int)
            for k, cl in enumerate(CLASSES):
                tape[f"p_{fam}_{mname}_{cl}"] = p4[:, k]
            tape[f"dec_{fam}_{mname}"] = np.where(
                pred == 0, "N",
                np.asarray(CLASSES)[p4[:, 1:].argmax(axis=1) + 1])
    tape.to_csv(out / "RQ01_10_ISSUED_TAPE_OOS.csv", index=False)
    tape["scorable"] = (panel_oos["label_status"] == STATUS_SCORABLE).astype(int)
    evaluation = tape.copy()
    evaluation["gold_label"] = panel_oos["label"].where(
        panel_oos["label_status"] == STATUS_SCORABLE)
    evaluation["label_status"] = panel_oos["label_status"]
    evaluation[evaluation["scorable"] == 1].to_csv(
        out / "RQ01_10_ESRB_SCORABLE_EVALUATION.csv", index=False)

    # --- 01-12 OOS reporting views: full, ESRB-scorable, model-only; one chronology
    views = []
    for fam in ("LOGISTIC", "XGBOOST"):
        for mname in ("M00", "M01", "M10", "M11"):
            score = np.asarray(grid[fam]["oos"][mname]["score"], float)
            thr = float(grid[fam]["oos"][mname]["threshold"])
            pos = score >= thr
            views.append({
                "family": fam, "model": mname, "update_policy": "DEV_FROZEN",
                "threshold": thr, "n_full": int(len(panel_oos)),
                "n_esrb_scorable": int(scor_mask_oos.sum()),
                "n_model_only": int(noc_mask_oos.sum()),
                "n_positive_full": int(pos.sum()),
                "n_positive_esrb_scorable": int((pos & scor_mask_oos).sum()),
                "n_positive_model_only": int((pos & noc_mask_oos).sum()),
                "phase_engine": ("CANONICAL_RQ01_SIX_PHASE_DEV_FROZEN_OOS"
                                 if phase_summary.get("status") != "NOT_EVALUATED"
                                 else "COMPOSITE_LEGACY_NOT_CANONICAL_REV353"),
                "n_canonical_countries": int(7 - len(phase_failures)),
            })
    pd.DataFrame(views).to_csv(out / "RQ01_12_OOS_VIEWS.csv", index=False)
    evaluation[evaluation["scorable"] == 1].to_csv(
        out / "RQ01_12_OOS_ESRB_SCORABLE.csv", index=False)
    tape[tape["scorable"] == 0].to_csv(out / "RQ01_12_OOS_MODEL_ONLY.csv", index=False)
    canon_suffix = ("FAIL_CLOSED_NOT_EVALUATED" if phase_summary.get("status") == "NOT_EVALUATED"
                    else f"{7 - len(phase_failures)}_OF_7")
    phase_failure_text = ";".join(f"{c}:{r}" for c, r in sorted(phase_failures.items())) or "NONE"
    n_state_none = int((panel_oos["phase_source_oos"] == "CANONICAL_STATE_NONE").sum())
    n_phase_not_evaluated = int(panel_oos["phase_canonical"].isna().sum())
    native_ok = native_result.get("status") == "RESEARCH_NATIVE_PROFILE"
    selection_status = native_result.get("selection_status")
    n_sel = native_result.get("n_selected_configs", 0)
    n_research = native_result.get("n_research_feature_configs", 0)
    parity_dev = native_result.get("config_parity_dev", {}) or {}
    n_parity_ok = parity_dev.get("n_config_parity_pass", 0)
    n_parity_rows = parity_dev.get("n_issue_rows", 0)
    ne_total = (int(native_summary["not_evaluated_reasons"]["count"].sum())
                if native_summary is not None and len(native_summary["not_evaluated_reasons"]) else 0)
    ne_codes = int(len(native_summary["not_evaluated_reasons"])) if native_summary is not None else 0
    sideway = (native_summary["joint"][native_summary["joint"]["motif"].isin(
        ["US", "SU", "DS", "SD", "SS"])] if native_summary is not None else pd.DataFrame())
    sideway_ne = int(sideway["n_not_evaluated"].sum()) if len(sideway) else 0
    sideway_ne_codes = int(len({k for v in sideway["not_evaluated_reasons"]
                                for k in __import__("json").loads(v).keys()})) if len(sideway) else 0
    fi_phase_status = phase_failures.get("FI", "OK")
    ind_status = independent.get("status", "NOT_RUN")
    status12 = [
        {"id": "01-01", "acceptance": "PARTIAL",
         "status": ("COMPLETE_NATIVE_COVERAGE_RESEARCH_ASOF" if native_ok
                    else "COMPLETE_RESEARCH_ASOF"),
         "evidence": "RQ01_01_DATA_COVERAGE.csv / RQ01_01_NATIVE_COVERAGE.csv",
         "unresolved": "OFFICIAL_ISSUER_PIT_NOT_CERTIFIED"},
        {"id": "01-02", "acceptance": "PARTIAL",
         "status": ("PARTIAL_NATIVE_RESEARCH_PROFILE" if native_ok
                    else "PARTIAL_RESEARCH_PROXY"),
         "evidence": "RQ01_02_INDIVIDUAL_5.csv / RQ01_02_INDIVIDUAL_NATIVE_TAPE.csv",
         "unresolved": ("OWNER_SOURCE_CUTS_NOT_CERTIFIED; NATIVE_RELEASE_CLOCK_RESEARCH_ONLY; "
                        "DEV_METRICS_NOT_GATE_CERTIFIED; "
                        "M01_IND_BLOCK=NATIVE_PACKET_INDIVIDUAL_VOTES")},
        {"id": "01-03", "acceptance": "PARTIAL",
         "status": ("COMPLETE_NATIVE_9_MOTIF_G0_G4_RESEARCH" if native_ok
                    else "PARTIAL_RESEARCH_PROXY"),
         "evidence": ("RQ01_03_JOINT_10.csv / RQ01_03_JOINT_NATIVE_TAPE.csv / "
                      "RQ01_PAIR_G0_G4_DEV.csv / RQ01_PAIR_MOTIF_POOL_LEDGER_DEV.csv / "
                      "RQ01_03_CONFIG_SELECTION_CONSUMPTION_DEV.csv"),
         "unresolved": (f"SELECTION={selection_status}; N_SELECTED={n_sel}; "
                        f"N_RESEARCH_FEATURES={n_research}; G1_RELEASE_RESEARCH_ASOF; "
                        f"SIDEWAY_NOT_EVALUATED_WITH_REASON={sideway_ne} ({sideway_ne_codes} codes); "
                        "OWNER_GATE1_GATE2_NOT_CERTIFIED")},
        {"id": "01-04", "acceptance": "PARTIAL",
         "status": ("PARTIAL_NATIVE_POOL_6_PLUS_FALLBACK" if native_ok
                    else "PARTIAL_RESEARCH_PROXY"),
         "evidence": ("RQ01_04_POOLS_6.csv / RQ01_04_POOL_NATIVE_TAPE.csv / "
                      "RQ01_POOL_POLICY_COMPARISON_DEV.csv / RQ01_04_NOT_EVALUATED_REASON_LEDGER_DEV.csv"),
         "unresolved": (f"REGISTRY_POOL={native_result.get('selection_pool_operator')}; SELECTED={n_sel}; "
                        "POOL_FALLBACK_IND_ONLY_WHEN_EMPTY; OWNER_POOL_POLICY_NOT_LOCKED; "
                        f"NOT_EVALUATED_REASON_ROWS={ne_total} ({ne_codes} codes); DEV_FPR_NOT_GATE_CERTIFIED")},
        {"id": "01-05", "acceptance": "PARTIAL", "status": "COMPLETE_PACKET_4M_APPARENT",
         "evidence": "RQ01_05_FOUR_M_DEV.csv / RQ01_05_FOUR_M_OOS.csv / RQ01_05_CALIBRATION.csv",
         "unresolved": (f"FEATURE_SOURCE={feature_source}; JOINT_COLUMNS={len(native_result.get('packet_feature_columns') or [])}; "
                        f"SELECTED_COLUMNS={len(native_result.get('packet_selected_columns') or [])}; "
                        "XGB_OVERFIT_DEV/AUC1; OOS_FNR_NOT_GATE_CERTIFIED")},
        {"id": "01-06", "acceptance": "PARTIAL", "status": f"PARTIAL_CANONICAL_{canon_suffix}",
         "evidence": "RQ01_06_PHASE_6.csv",
         "unresolved": (f"CANONICAL_FAILURES={phase_failure_text}; FI_PHASE={fi_phase_status}; "
                        "LEGACY_FALLBACK_TO_CONFIRMED_DISABLED; NO_RECOVERY_UNDER_BASELINE_PROFILE")},
        {"id": "01-07", "acceptance": "PARTIAL",
         "status": (f"COMPLETE_EPISODE_CONDITIONAL_CANONICAL_{canon_suffix}"
                    if episode_conditional.get("status", "").startswith("COMPLETE")
                    else f"PARTIAL_LOW_SUPPORT_CANONICAL_{canon_suffix}"),
         "evidence": ("RQ01_07_PHASE_CONDITIONAL_1TO5.csv / RQ01_07_EPISODE_SOURCE_CONFIRMATION.csv / "
                      "RQ01_07_EPISODE_CONDITIONAL.csv / RQ01_07_EPISODE_CONDITIONAL_SUMMARY.json"),
         "unresolved": (f"CANONICAL_FAILURES={phase_failure_text}; EPISODES="
                        f"{episode_conditional.get('n_evaluated_episodes')}; LOW_EVENT_SUPPORT")},
        {"id": "01-08", "acceptance": "PARTIAL", "status": "COMPLETE_RESEARCH_BOUNDED_OWNER_UNVERIFIED",
         "evidence": "RQ01_08_THRESHOLD_REGISTER.json",
         "unresolved": ("BOUNDED_2D_GRID_NOT_EXHAUSTIVE_PROOF; OWNER_THRESHOLDS_MISSING; "
                        "THRESHOLD_OWNER_STATUS=RESEARCH_ONLY_NO_OWNER_THRESHOLD_REGISTRY")},
        {"id": "01-09", "acceptance": "BLOCKED", "status": "BLOCKED_OWNER_DEFINITION",
         "evidence": ("RQ01_09_F1_F5.csv / RQ01_09_F1_F5_VERIFICATION.json / "
                      "RQ01_F1_F5_AUTHORITY_LEDGER.json"),
         "unresolved": ("OFFICIAL_F1_F5_OWNER_DEFINITIONS_NOT_AUTHENTICATED; "
                        "RESEARCH_DIAGNOSTICS_ONLY; OFFICIAL_NAMESPACE=NOT_DEFINED")},
        {"id": "01-10", "acceptance": "PARTIAL", "status": f"COMPLETE_OOS_PREDICTION_NO_GOLD_{canon_suffix}",
         "evidence": ("RQ01_10_ISSUED_TAPE_OOS.csv / RQ01_10_ESRB_SCORABLE_EVALUATION.csv / "
                      "RQ01_03_PACKET_FEATURES_OOS.csv"),
         "unresolved": (f"OOS_PHASE_DEV_FROZEN_SNAPSHOT; STATE_NONE_ROWS={n_state_none}; "
                        f"PHASE_NOT_EVALUATED_ROWS={n_phase_not_evaluated}; NO_ISSUER_PIT; FROZEN_SELECTED={n_sel}; "
                        f"CANONICAL_MISSING={phase_failure_text}")},
        {"id": "01-11",
         "acceptance": ("PASS" if ind_status.startswith("PASS") else "PARTIAL"),
         "status": ("COMPLETE_INDEPENDENT_MATCHED_ABLATION"
                    if ind_status.startswith("PASS") else "PARTIAL_INDEPENDENT_ABLATION"),
         "evidence": ("RQ01_11_CANONICAL_PARITY.json / RQ01_11_NUMERICAL_VS_INDEPENDENT_PARITY.json / "
                      "RQ01_11_INDEPENDENT_ABLATION.csv / RQ01_11_OFFICIAL_VS_INDEPENDENT.csv / "
                      "RQ01_11_BLOCK_BOOTSTRAP.csv"),
         "unresolved": (f"INDEPENDENT_STATUS={ind_status}; FEATURES={independent.get('joint_feature_source')}; "
                        "ESTIMATOR_PRIMARY=MNLOGIT; NUMERICAL_PARITY_SEPARATE; "
                        "BASE_IND_BLOCK=PROXY_SPEC_FIXED_FOR_ABLATION; "
                        "COUNTRY_MONTHS_OF_EPISODE_NOT_INDEPENDENT; OWNER_CERTIFICATION_MISSING")},
        {"id": "01-12", "acceptance": "PARTIAL", "status": f"COMPLETE_OOS_VIEWS_SPLIT_NO_GOLD_{canon_suffix}",
         "evidence": ("RQ01_12_OOS_VIEWS.csv / _ESRB_SCORABLE.csv / _MODEL_ONLY.csv / "
                      "RQ01_12_ACCEPTANCE_MATRIX.csv"),
         "unresolved": (f"ESRB_SCORABLE_ONLY_{int(scor_mask_oos.sum())}_KEYS; "
                        f"NOT_EVALUATED_REASON_ROWS={ne_total}; CONFIG_PARITY={n_parity_ok}/{n_parity_rows}; "
                        "OOS_PHASE_DEV_FROZEN; "
                        f"CANONICAL_MISSING={phase_failure_text}")},
    ]
    status_df = pd.DataFrame(status12)
    status_df.to_csv(out / "RQ01_12_OUTPUT_STATUS.csv", index=False)
    status_df[["id", "acceptance", "status", "evidence", "unresolved"]].to_csv(
        out / "RQ01_12_ACCEPTANCE_MATRIX.csv", index=False)

    # --- calibration diagnostics (evaluated, never used for model selection)
    cal_rows = []
    for f2 in ("LOGISTIC", "XGBOOST"):
        for m2 in ("M00", "M01", "M10", "M11"):
            pd_dev = grid[f2]["dev"][m2]["p4"]
            pd_oos = grid[f2]["oos"][m2]["p4"]
            rep = EST.calibration_report(np.asarray(pd_dev), PNL.y4_from(panel_dev),
                                         np.asarray(pd_oos)[scor_mask_oos],
                                         PNL.y4_from(panel_oos)[scor_mask_oos])
            cal_rows.append({"family": f2, "model": m2, **rep})
    pd.DataFrame(cal_rows).to_csv(out / "RQ01_05_CALIBRATION.csv", index=False)

    # --- expanding (E1 logistic / E3 xgb on selected M)
    panel_pkt = panel.copy().reset_index(drop=True)
    if feature_dev is not None and joint_cols:
        extra_cols = (joint_cols + pkt_ind_cols
                      + ["pkt_joint_not_evaluated_count", "pkt_pool_issued",
                         "pkt_ind_evaluated_count"])
        pkt_all = pd.concat([panel_dev[["country", "month"] + extra_cols],
                             panel_oos[["country", "month"] + extra_cols]],
                            ignore_index=True)
        panel_pkt = panel_pkt.merge(pkt_all, on=["country", "month"],
                                    how="left", validate="one_to_one")
    oos_months = sorted(set(panel_oos["month"])) if not budget_fast else \
        sorted(set(panel_oos["month"]))[::3]
    exp_out = {}
    if not budget_fast:
        for fam in ("LOGISTIC", "XGBOOST"):
            try:
                exp_out[fam] = EXP.expanding_refit(
                    panel_pkt, sel_cols, PNL.y4_from(panel_pkt),
                    dev_mask=dev_mask.values, family=fam, name=sel_model, oos_months=oos_months)
                jdump(exp_out[fam]["ledger"], out / f"RQ01_13_EXPANDING_LEDGER_{fam}.json")
            except Exception as exc:
                exp_out[fam] = {"status": "BLOCKED_BY_CONTRACT", "contract_status": "BLOCKED_BY_ACTIVE_CONTRACT",
                                "ledger": [], "reason": f"{type(exc).__name__}:{exc}"}
        try:
            probe = EXP.invariance_probe(
                panel_pkt, sel_cols, PNL.y4_from(panel_pkt),
                dev_mask=dev_mask.values, family="LOGISTIC", name=sel_model, u="2021-06")
            jdump(probe, out / "RQ01_14_EXPANDING_INVARIANCE.json")
        except Exception as exc:
            jdump({"status": "BLOCKED_BY_CONTRACT", "reason": f"{type(exc).__name__}:{exc}"},
                  out / "RQ01_14_EXPANDING_INVARIANCE.json")

    # --- 01-03/04 selection-consumption and Pool policy evidence
    if native is not None and native_oos is not None:
        jdump({
            "claim": "SELECTED_CONFIG_IDS_EQUAL_CONSUMED_CONFIG_IDS",
            "identity_includes": "pair,leader,follower,motif,EXACT,lag,delta,unit,profile,release,owner",
            "dev": native_summary["config_parity"],
            "oos": {
                "n_issue_rows": int(len(native_oos["config_parity"])),
                "n_config_parity_pass": int(
                    (native_oos["config_parity"]["status"]
                     == "PASS_SELECTED_EQUALS_CONSUMED_CONFIG_ID").sum()),
                "n_selected_configs": int(native_oos["config_parity"]["n_selected"].max()),
                "n_consumed_configs": int(native_oos["config_parity"]["n_consumed"].max()),
                "n_research_configs": int(native_oos["config_parity"]["n_research"].max()),
                "pool_fallback_used": bool(native_oos["config_parity"]["pool_fallback_used"].any()),
            },
            "selected_config_ids": (sel_public or {}).get("selected_config_ids"),
            "research_feature_config_ids": (sel_public or {}).get("research_feature_config_ids"),
            "registry_pool_operator": selected_pool,
            "no_oos_selection": True,
        }, out / "RQ01_03_SELECTED_VS_CONSUMED.json")
        jdump({
            "registry_pool_operator": selected_pool,
            "selection_status": selection_status,
            "selected_pairs": (sel_public or {}).get("chosen_pairs"),
            "selected_config_ids": (sel_public or {}).get("selected_config_ids"),
            "pool_policy_comparison": (sel_public or {}).get("pool_policy_comparison"),
            "selection_claim": "RESEARCH_REGISTRY_DECISION_NOT_OWNER_CERTIFIED",
        }, out / "RQ01_04_POOL_SELECTION_POLICY.json")

    results = {
        "run_id": run_id("REV_NEXT_RQ01"), "created_utc": utcnow(),
        "scope": "EA7 DEV 2005-07..2019-12 / OOS 2020-01..2024-12",
        "n_dev": int(dev_mask.sum()), "n_oos": int(oos_mask.sum()),
        "n_oos_scorable": int(scor_mask.sum()), "n_oos_no_esrb": int(noc_mask.sum()),
        "selected_family": sel_family, "selected_model": sel_model,
        "selected_threshold": sel_thr, "selection_rule": sel_rule,
        "dev_metrics": dev_rows, "oos_metrics": oos_rows,
        "dev_class_counts": {c: int((PNL.y4_from(panel_dev) == i).sum())
                             for i, c in enumerate(CLASSES)},
        "expanding": {f: {"status": v["status"], "contract_status": v["contract_status"],
                          "n_refits": len(v["ledger"])} for f, v in exp_out.items()},
        "xgb_available": EST.XGB_AVAILABLE, "xgb_version": EST.XGB_VERSION,
        "canonical_phase": phase_summary,
        "fi_phase_status": fi_phase_status,
        "phase_fail_closed": {
            "legacy_fallback_to_confirmed": False,
            "missing_snapshot_policy": "NOT_EVALUATED_NO_CANONICAL_SNAPSHOT",
            "n_phase_not_evaluated_oos": n_phase_not_evaluated,
            "n_state_none_oos": n_state_none,
            "phase_not_evaluated_reason_counts": (panel_oos["phase_not_evaluated_reason"]
                                                  .fillna("CONFIRMED").value_counts().to_dict()),
        },
        "native_ijp": native_result,
        "packet_feature_source": feature_source,
        "packet_joint_columns": joint_cols,
        "episode_conditional": episode_conditional,
        "not_evaluated_reason_summary": (native_summary["not_evaluated_reasons"].to_dict("records")
                                         if native_summary is not None else []),
        "pair_selection": sel_public,
        "independent_11": independent,
        "calibration": cal_rows,
        "parity": parity, "threshold_register": reg,
    }
    # panel with the selected RQ01 probabilities for downstream RQ03 regimes
    panel_prob = panel_pkt.copy().reset_index(drop=True)
    p_all = sel_mdl.p4(PNL.matrix_for(panel_prob, sel_cols))
    for k, cl in enumerate(CLASSES):
        panel_prob[f"p_rq01_{cl}"] = p_all[:, k]
    results["_panel"] = panel_prob
    jdump({k: v for k, v in results.items() if k != "_panel"}, out / "RQ01_RESULTS.json")
    # --- single-run evidence manifest (roadmap 01-12 / T09): one run ID, per-output
    #     hashes, key code SHAs; never mixes evidence across run IDs.
    import hashlib
    _manifest = {"run_id": results["run_id"], "created_utc": results["created_utc"],
                 "n_outputs": 0, "outputs": {}, "code": {}, "key_results": {
                     "selected_family": sel_family, "selected_model": sel_model,
                     "selected_threshold": sel_thr, "n_dev": int(dev_mask.sum()),
                     "n_oos": int(oos_mask.sum()), "n_oos_scorable": int(scor_mask.sum())}}
    for _p in sorted(out.glob("*")):
        if _p.is_file() and _p.name != "RUN_MANIFEST.json":
            _manifest["outputs"][_p.name] = hashlib.sha256(_p.read_bytes()).hexdigest()
    _manifest["n_outputs"] = len(_manifest["outputs"])
    _root = Path(__file__).resolve().parent.parent
    for _f in ("rev363/rq01_next.py", "rev363/rq01_native_ijp.py", "rev363/features.py",
               "research_adapters/vote_contract.py", "research_adapters/rq01/common.py",
               "research_adapters/rq01/individual.py", "research_adapters/rq01/pools.py",
               "research_adapters/rq01/pairs.py"):
        _fp = _root / _f
        if _fp.is_file():
            _manifest["code"][_f] = hashlib.sha256(_fp.read_bytes()).hexdigest()
    jdump(_manifest, out / "RUN_MANIFEST.json")
    return results
