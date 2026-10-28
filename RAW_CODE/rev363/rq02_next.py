"""RQ02 corrected DEV/OOS execution: Long/Short, five horizons, EXIT-A/B, S5, Bottom.

Defects repaired
----------------
P0-03  Long/Short magnitude quantiles use strictly prior, label-free
       magnitudes; OOS must receive the DEV-frozen calibration table.
P0-04  EXIT uses the canonical first-normal month (NormalDate month inclusive).
P0-05  Bottom target is ``argmin`` of the *cumulative self-financing level*
       rebuilt from returns, never ``argmin`` of a weekly return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rev363.config import CLASSES, EA7, HORIZONS, HORIZON_MONTHS, PAIRS21, SOURCES7
from rev363 import metrics as MET
from rev363 import thresholds as THR
from rev363.data import _exact, find_dataset_root
from rev363 import estimators as EST
from rev363.w02_integrity import select_dev_binary_threshold
from rev363.rq02_features import (add_long_short_changes, apply_long_short_votes,
                                  final_calibration)
from types import SimpleNamespace
from rev363 import panel as PNL
from rev363.labels_next import (ESRB_SCORABLE_END, STATUS_SCORABLE,
                                  all_normal_months, all_onset_months)
from rev363.util import jdump, run_id, utcnow

DEV_FIT_START, DEV_END = "2005-07", "2019-12"
OOS_END = "2024-12"
LONG_SRC = ("CG", "DSR-P")
SHORT_SRC = ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")
MOTIFS9 = tuple(a + b for a in "UDS" for b in "UDS")


def _mi(m):
    return int(m[:4]) * 12 + int(m[5:7]) - 1


def add_months(m, k):
    y = int(m[:4]); mm = int(m[5:7]) - 1 + k
    y += mm // 12; mm %= 12
    return f"{y:04d}-{mm + 1:02d}"


# ---------------------------------------------------------------------------
# frame with DEV-frozen, label-independent Long/Short calibration (P0-03)
# ---------------------------------------------------------------------------
def build_frame(panel: pd.DataFrame, *, dev_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Causal frame shared with rev363.rq02; labels never enter a feature.

    ``dev_mask`` is retained for call compatibility and is recorded in the
    calibration provenance, not used to fit a threshold on future rows.
    """
    df = add_long_short_changes(panel)
    df = apply_long_short_votes(df, long_sources=LONG_SRC)
    calib = dict(df.attrs.get("long_short_calibration", final_calibration(df)))
    calib["_mode"] = df.attrs.get("long_short_calibration_mode")
    calib["_dev_rows"] = int(np.asarray(dev_mask, dtype=bool).sum())
    return df, calib


def feature_columns(df: pd.DataFrame) -> dict:
    base = []
    for s in SOURCES7:
        base += [f"z_{s}", f"d3_{s}", f"d12_{s}", f"dd_{s}", f"long_{s}", f"short_{s}"]
    base += ["intensity", "n_sources_available"]  # n_long/short are Individual-derived; no M00 leakage
    base += [f"c_{c}" for c in EA7]
    base += [f"cv_{c}_{s}" for c in EA7 for s in SOURCES7 if f"cv_{c}_{s}" in df.columns]
    ind = [f"L_{s}" for s in SHORT_SRC] + [f"S_{s}" for s in SHORT_SRC]
    joint = [f"short_pair_{a}_{b}" for a, b in PAIRS21] + ["long_pair_pass", "n_short_pairs"]
    return {"BASE": base, "IND": ind, "JOINT": joint}


def block_cols(cols, mname):
    return {"M00": cols["BASE"], "M01": cols["BASE"] + cols["IND"],
            "M10": cols["BASE"] + cols["JOINT"],
            "M11": cols["BASE"] + cols["IND"] + cols["JOINT"]}[mname]


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def onset_label(row_country, month, starts, h):
    u = _mi(month)
    return int(any(u < _mi(t) <= u + h for t in starts.get(row_country, [])))


def future_state(panel_lookup, country, month, h):
    # unobserved future month -> censored label status (never a state class, never N)
    from research_adapters.vote_contract import LABEL_CENSORED
    return panel_lookup.get((country, add_months(month, h)), LABEL_CENSORED)


def exit_event(label, month, normals, h, end):
    if label == "N":
        return None, False
    u = _mi(month)
    mature = add_months(month, h) <= end
    later = [_mi(t) - u for t in normals if _mi(t) > u]
    if not mature:
        return None, False
    # A matured at-risk window with no observed first N is a genuine negative.
    return int(bool(later) and min(later) <= h), True


# ---------------------------------------------------------------------------
# Bottom on the real asset level (P0-05)
# ---------------------------------------------------------------------------
def asset_levels(root, *, include_oos: bool = False) -> pd.DataFrame:
    """DEV weekly tape with compounded INDEX LEVELS (lvl_*).  The OOS tape is never loaded here
    (hard gate: no OOS loader inside a DEV run); include_oos=True is refused."""
    if include_oos:
        raise PermissionError("OOS_SEALED: Bottom must not load the 2020-2024 tape")
    tape = pd.read_csv(_exact(root, "V8_RQ03_SIX_ASSET_DEV_WEEKLY.csv"))
    tape["week_end"] = pd.to_datetime(tape["week_end"])
    cols = ("E1", "E2", "E3", "B1", "B2", "C1")
    allw = tape[["week_end"] + list(cols)].sort_values("week_end").reset_index(drop=True)
    for a in cols:
        allw[f"lvl_{a}"] = (1.0 + allw[a].astype(float)).cumprod()
    allw["month"] = allw["week_end"].dt.strftime("%Y-%m")
    return allw


def bottom_zones(panel, episodes, levels, *, dev: bool):
    """F1 (EWI cooling) + F2 (peer recovery) + ASSET-SPECIFIC F3; zone if >= 2 votes.

    Thin wrapper over research_adapters.rq02_bottom_dev_harness (spec RQ02 Bottom 3F): causal F3 per
    asset, zone strictly after the issue, trough = lowest index LEVEL in a strictly future window.
    OOS scoring is sealed, so dev=False returns an empty frame.
    """
    from research_adapters import rq02_bottom_dev_harness as BH
    if not dev:
        return pd.DataFrame()
    eps = [e for e in episodes if e["country"] in EA7 and e["systemic"] == 1
           and e["dev_eligibility"] != "PRE_DEV_ONLY" and pd.Timestamp(e["start"]).year <= 2019]
    weekly = levels.set_index("week_end").sort_index()[["E1", "E2", "E3", "B1", "B2"]]
    return BH.bottom_dev_zones(panel, eps, weekly)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(dataset_root, out_dir, *, budget_fast=False, rq01_tape=None,
        rq02_owner_approval_id=None, native_pit_approval_id=None,
        oos_final_evaluation_approval_id=None) -> dict:
    # RQ02 cannot consume retrospective monthly as-of proxies as certified issued
    # history. Prevent accidental execution of the legacy DEV-fit/OOS-score path
    # until the three separate owner/release/OOS gates are externally validated.
    # These strings are necessary but NOT sufficient as certification evidence.
    if not all(isinstance(v, str) and v.strip() for v in (
            rq02_owner_approval_id, native_pit_approval_id,
            oos_final_evaluation_approval_id)):
        raise PermissionError(
            "W02_RQ02_SUPERVISED_ISSUED_PIPELINE_BLOCKED: "
            "owner target contract, seven-source issuer PIT, and sealed OOS "
            "evaluation approval are not supplied. Use w02_native_dev / "
            "w02_dev_audit for legal DEV-only evidence.")
    # In the present W02 patch, there is no verifier for signed owner/issuer
    # evidence; self-declared approval strings cannot unlock an OOS run.
    # Preserve the legacy body below for auditor comparison only, not execution.
    raise PermissionError(
        "W02_RQ02_APPROVAL_STRINGS_NOT_CERTIFICATION: legacy supervised "
        "forecast path remains blocked pending an independently validated "
        "owner/issuer/OOS gate manifest and full issued-tape implementation.")
    out = out_dir
    prep = PNL.build_panel(dataset_root, row_start="2004-01", row_end=OOS_END, data_end=OOS_END)
    panel = prep["panel"]; episodes = prep["episodes"]
    dev_mask = ((panel["month"] >= "2005-01") & (panel["month"] <= DEV_END)).values
    df, calib = build_frame(panel, dev_mask=dev_mask)
    cols = feature_columns(df)

    y4_all = PNL.y4_from(df)
    dev_fit = (df["month"] >= DEV_FIT_START).values & (df["month"] <= DEV_END).values

    # --- 02-01 Long/Short native matrices
    rows = []
    for s in SOURCES7:
        rows.append({"source": s,
                     "roster": "LONG" if s in LONG_SRC else "SHORT",
                     "n_long": int(df[f"L_{s}"].sum()), "n_short": int(df[f"S_{s}"].sum()),
                     "mean_long_stress": float(df.loc[df["stress"] == 1, f"L_{s}"].mean()),
                     "mean_short_stress": float(df.loc[df["stress"] == 1, f"S_{s}"].mean()),
                     "mean_long_N": float(df.loc[df["stress"] == 0, f"L_{s}"].mean()),
                     "mean_short_N": float(df.loc[df["stress"] == 0, f"S_{s}"].mean()),
                     "terminal_long_mean": float(df[f"long_{s}"].mean()),
                     "terminal_short_mean": float(df[f"short_{s}"].mean())})
    pd.DataFrame(rows).to_csv(out / "RQ02_01_LONGSHORT_NATIVE.csv", index=False)

    # --- 02-02 1 + 21 joint pairs
    mat = []
    for a, b in PAIRS21:
        mat.append({"pair": f"{a}-{b}", "type": "SHORT", "n_uu": int(df[f"short_pair_{a}_{b}"].sum()),
                    "mean_stress": float(df.loc[df["stress"] == 1, f"short_pair_{a}_{b}"].mean()),
                    "mean_N": float(df.loc[df["stress"] == 0, f"short_pair_{a}_{b}"].mean())})
    mat.append({"pair": "CG-DSR-P", "type": "LONG", "n_uu": int(df["long_pair_pass"].sum()),
                "mean_stress": float(df.loc[df["stress"] == 1, "long_pair_pass"].mean()),
                "mean_N": float(df.loc[df["stress"] == 0, "long_pair_pass"].mean())})
    pd.DataFrame(mat).to_csv(out / "RQ02_02_JOINT_1PLUS21.csv", index=False)

    # --- 02-03 horizons x 4M x 2 families
    onset_starts = {c: all_onset_months(episodes, c, allow_oos=True) for c in EA7}
    # Evaluation labels are retrospective; no post-DEV event may enter a DEV fit.
    dev_onset_starts = {c: [x for x in onset_starts[c] if x <= DEV_END] for c in EA7}
    lut = {(c, m): l for c, m, l in zip(df["country"], df["month"], df["label"])}
    hz_rows, hz_store = [], {}
    for hn in HORIZONS:
        h = HORIZON_MONTHS[hn]
        onset = np.array([onset_label(c, m, onset_starts, h) for c, m in
                          zip(df["country"], df["month"])])
        onset_dev = np.array([onset_label(c, m, dev_onset_starts, h) for c, m in
                              zip(df["country"], df["month"])])
        eligible = (df["label"] == "N").values & \
                   np.array([add_months(m, h) <= OOS_END for m in df["month"]])
        dev_elig = (eligible & dev_fit &
                    np.array([add_months(m, h + 3) <= DEV_END for m in df["month"]]))
        for mname in ("M00", "M01", "M10", "M11"):
            block = block_cols(cols, mname)
            Xd = PNL.matrix_for(df[dev_elig], block)
            yd = onset_dev[dev_elig]
            for fam in ("LOGISTIC", "XGBOOST"):
                mdl = EST.fit_two_family(Xd, block, yd, family=fam, name=f"{hn}_{mname}")
                tr = SimpleNamespace(**select_dev_binary_threshold(
                    mdl.score(Xd), yd, fnr_max=0.10, fpr_max=0.20))
                cm = MET.confusion_binary(yd, (mdl.score(Xd) >= tr.threshold).astype(int))
                hz_rows.append({"horizon": hn, "family": fam, "model": mname,
                                "n_dev": int(dev_elig.sum()), "n_events": int(yd.sum()),
                                "auc": MET.auc_score(yd, mdl.score(Xd)),
                                "logloss": mdl.fit_logloss, "threshold": tr.threshold,
                                **{k: cm[k] for k in ("TP", "FN", "FP", "TN", "FNR", "FPR")},
                                "gate_feasible": tr.gate_feasible})
                hz_store[(hn, mname, fam)] = {"model": mdl, "block": block,
                                              "threshold": tr.threshold}
    pd.DataFrame(hz_rows).to_csv(out / "RQ02_03_FOUR_M_HORIZONS_DEV.csv", index=False)

    # --- 02-03b OOS horizon scoring (DEV-frozen model, scorable window only)
    oos_rows = []
    for hn in HORIZONS:
        h = HORIZON_MONTHS[hn]
        onset = np.array([onset_label(c, m, onset_starts, h) for c, m in
                          zip(df["country"], df["month"])])
        oos_elig = (df["label"] == "N").values & \
                   (df["month"] >= "2020-01").values & \
                   np.array([add_months(m, h) <= OOS_END for m in df["month"]])
        sc = oos_elig & (df["label_status"] == STATUS_SCORABLE).values
        for mname in ("M00", "M01", "M10", "M11"):
            for fam in ("LOGISTIC", "XGBOOST"):
                key = (hn, mname, fam)
                if key not in hz_store:
                    continue
                st = hz_store[key]
                Xo = PNL.matrix_for(df[oos_elig], st["block"])
                s = st["model"].score(Xo)
                yo = onset[oos_elig]
                mask_sc = (df["label_status"] == STATUS_SCORABLE).values[oos_elig]
                if mask_sc.sum() and len(np.unique(yo[mask_sc])) > 1:
                    cm = MET.confusion_binary(yo[mask_sc], (s[mask_sc] >= st["threshold"]).astype(int))
                    oos_rows.append({"horizon": hn, "family": fam, "model": mname,
                                     "window": "ESRB_SCORABLE", "n": int(mask_sc.sum()),
                                     "events": int(yo[mask_sc].sum()),
                                     "auc": MET.auc_score(yo[mask_sc], s[mask_sc]),
                                     "frozen_threshold": st["threshold"],
                                     **{k: cm[k] for k in ("TP", "FN", "FP", "TN", "FNR", "FPR")}})
                else:
                    oos_rows.append({"horizon": hn, "family": fam, "model": mname,
                                     "window": "ESRB_SCORABLE", "n": int(mask_sc.sum()),
                                     "events": int(yo[mask_sc].sum()) if mask_sc.sum() else 0,
                                     "auc": None, "frozen_threshold": st["threshold"],
                                     "TP": None, "FN": None, "FP": None, "TN": None,
                                     "FNR": None, "FPR": None})
                oos_rows.append({"horizon": hn, "family": fam, "model": mname,
                                 "window": "NO_ESRB_PREDICTION_ONLY",
                                 "n": int(oos_elig.sum() - mask_sc.sum()),
                                 "events": None, "auc": None,
                                 "frozen_threshold": st["threshold"],
                                 "TP": None, "FN": None, "FP": None, "TN": None,
                                 "FNR": None, "FPR": None})
    pd.DataFrame(oos_rows).to_csv(out / "RQ02_03_FOUR_M_HORIZONS_OOS.csv", index=False)

    # --- 02-04 S5
    s5 = []
    sel = hz_store[("H6", "M11", "LOGISTIC")]
    score_all = sel["model"].score(PNL.matrix_for(df, sel["block"]))
    for c in EA7:
        idx = df.index[df["country"] == c]
        prev = False
        last = None
        for i in idx:
            cur = float(score_all[i]) >= sel["threshold"]
            if cur and not prev:
                last = df.loc[i, "month"]
            prev = float(cur)
            age = None if last is None else (_mi(df.loc[i, "month"]) - _mi(last))
            s5.append({"country": c, "issue": df.loc[i, "month"], "label": df.loc[i, "label"],
                       "s5_last_short": last,
                       "s5_confirmed": pd.NA,
                       "s5_status": "NOT_EVALUATED_NATIVE_RELEASE_NOT_VERIFIED",
                       "s5_exploratory_H6_score_trigger": int(last is not None and age is not None and age <= 18),
                       "s5_age_months": age,
                       "n_short_sources": (int(df.loc[i, "n_short_sources"])
                                           if pd.notna(df.loc[i, "n_short_sources"]) else None),
                       "n_short_pairs": (int(df.loc[i, "n_short_pairs"])
                                         if pd.notna(df.loc[i, "n_short_pairs"]) else None)})
    pd.DataFrame(s5).to_csv(out / "RQ02_04_S5.csv", index=False)

    # --- 02-05/06 EXIT A/B
    norms = {c: all_normal_months(episodes, c, allow_oos=True) for c in EA7}
    ex_rows = []
    for h in (3, 6, 12, 18):
        ev, el = [], []
        for c, lab, m in zip(df["country"], df["label"], df["month"]):
            e, ok = exit_event(lab, m, norms.get(c, []), h, OOS_END)
            ev.append(e); el.append(ok)
        ev = np.array([-1 if x is None else x for x in ev]); el = np.array(el)
        elig = (el & (ev >= 0) & dev_fit & (df["label_status"] == STATUS_SCORABLE).values &
                np.array([add_months(m, h + 3) <= DEV_END for m in df["month"]]))
        for fam in ("LOGISTIC", "XGBOOST"):
            blk = block_cols(cols, "M11")
            Xe = PNL.matrix_for(df[elig], blk)
            ye = ev[elig]
            if len(np.unique(ye)) < 2:
                ex_rows.append({"exit": f"H{h}", "family": fam, "model": "A/B",
                                "n": int(elig.sum()), "events": int(ye.sum()),
                                "status": "NOT_EVALUATED_NO_EVENT_SUPPORT"})
                continue
            mdl = EST.fit_two_family(Xe, blk, ye, family=fam, name=f"EXIT_H{h}")
            p = mdl.score(Xe)
            ex_rows.append({"exit": f"H{h}", "family": fam, "model": "A",
                            "n": int(elig.sum()), "events": int(ye.sum()),
                            "auc": MET.auc_score(ye, p),
                            "logloss": mdl.fit_logloss,
                            **{k: v for k, v in MET.confusion_binary(ye, (p >= 0.5).astype(int)).items()
                               if k in ("TP", "FN", "FP", "TN", "FNR", "FPR")}})
    pd.DataFrame(ex_rows).to_csv(out / "RQ02_05_06_EXIT_AB.csv", index=False)

    # --- 02-07 override ledger (RQ01 issued state as operational fallback)
    led = df[["country", "month", "label", "phase"]].copy()
    if rq01_tape is not None and len(rq01_tape):
        led = led.merge(rq01_tape, left_on=["country", "month"], right_on=["country", "issue"],
                        how="left")
        if "confirmed_asof" in led and "issued_at" in led:
            valid_rq01 = led["confirmed_asof"].eq(True).fillna(False) & \
                         led["issued_at"].notna() & (led["issued_at"].astype(str) <= led["month"])
            led["rq01_issued_stress"] = led["pred_stress"].where(valid_rq01).astype("Int64")
        else:
            led["rq01_issued_stress"] = pd.Series(pd.NA, index=led.index, dtype="Int64")
    else:
        led["rq01_issued_stress"] = pd.Series(pd.NA, index=led.index, dtype="Int64")
    led["effective_state"] = np.select([led["rq01_issued_stress"].eq(1).fillna(False).to_numpy(bool),
                                          led["rq01_issued_stress"].eq(0).fillna(False).to_numpy(bool)],
                                         ["Stress", "N"], default="NOT_EVALUATED")   # no issued RQ01 state: not a class
    led["override_N_to_stress"] = pd.Series(pd.NA, index=led.index, dtype="Int64")
    led["reverse_stress_to_N"] = pd.Series(pd.NA, index=led.index, dtype="Int64")
    led["override_status"] = "NOT_EVALUATED_NO_MATCHED_ISSUED_EXIT_A"
    led.to_csv(out / "RQ02_07_OVERRIDE_LEDGER.csv", index=False)
    override_summary = {"n_N_to_stress": None, "n_stress_to_N": None,
                        "n_rows": len(led), "status": "NOT_EVALUATED_NO_MATCHED_ISSUED_EXIT_A"}

    # --- 02-08/09 Bottom zones on real asset level
    levels = asset_levels(prep["root"])
    bz_dev = bottom_zones(df[df["month"] <= DEV_END], episodes, levels, dev=True)
    bz_oos = bottom_zones(df[df["month"] >= "2020-01"], episodes, levels, dev=False)
    bz_dev.to_csv(out / "RQ02_08_BOTTOM_ZONES_DEV.csv", index=False)
    bz_oos.to_csv(out / "RQ02_09_BOTTOM_ZONES_OOS.csv", index=False)

    def bsum(bz):
        if bz.empty:
            return {"n": 0, "status": "OOS_SEALED_OR_NO_ZONE"}
        from research_adapters.rq02_bottom_dev_harness import summarize_bottom
        return summarize_bottom(bz)           # five assets only; EA5 control reported separately

    # --- 02-10 F1-F5 subtype
    f_rows = []
    for r in hz_rows:
        n = r["n_dev"]; k = len(block_cols(cols, r["model"]))
        f_rows.append({"horizon": r["horizon"], "family": r["family"], "model": r["model"],
                       "n": n, "F1_logloss": r["logloss"], "F3_auc": r["auc"],
                       "F4_selection_aware": r["logloss"] + (k / (2 * max(n, 1))) * np.log(max(n, 2))})
    pd.DataFrame(f_rows).to_csv(out / "RQ02_10_F1_F5_SUBTYPE.csv", index=False)

    results = {
        "run_id": run_id("REV_NEXT_RQ02"), "created_utc": utcnow(),
        "n_panel": len(df), "long_calibration": calib,
        "horizons_dev": hz_rows,
        "horizons_oos": oos_rows,
        "s5_rows": len(s5), "s5_confirmed": None,
        "s5_research_proxy_triggers": int(sum(x["s5_exploratory_H6_score_trigger"] for x in s5)),
        "exit_rows": ex_rows,
        "override": override_summary,
        "bottom_dev": bsum(bz_dev), "bottom_oos": bsum(bz_oos),
        "label_independent_features": "DEV_ONLY_CALIBRATION_PROXY; NOT OFFICIAL ISSUER VERIFIED",
        "threshold_objective": "RQ02_DEV_FNR_LT_10_FPR_LT_20_RESEARCH_ONLY",
        "scientific_gate": "NOT_CERTIFIED_NO_AUTHENTIC_NATIVE_RELEASE_AND_EA8_CONFLICT",
        "bottom_target": "ARG_MIN_CUMULATIVE_ASSET_LEVEL",
        "first_normal_endpoint": "NORMALDATE_MONTH_INCLUSIVE",
    }
    jdump(results, out / "RQ02_RESULTS.json")
    return results
