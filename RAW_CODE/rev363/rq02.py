"""REV363 W04 / RQ02: Long/Short entry, five lead horizons, EXIT-A/B, S5, Bottom.

Targets are deliberately separate:
  * P_onset(h)   = first N->stress transition in (u, u+h]
  * P_state(h)   = four-class N/D/G/BOTH state at u+h
  * P_exit(h)    = first N within h months given stress at u (EXIT-A)
The RQ02 fit is one full DEV block; no temporal folds, no OOS tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import labels as L
from . import metrics as MET
from . import models as MOD
from . import rq01 as R1
from . import thresholds as THR
from .config import (CLASSES, DEV_END, DEV_FIT_START, EA7, FEATURE_VERSION,
                     HORIZON_MONTHS, HORIZONS, MODEL_VERSION, PAIRS21, SOURCES7,
                     canonical_json, sha256_json)
from .data import _exact, add_months, find_dataset_root
from .rq02_features import add_long_short_changes, apply_long_short_votes, final_calibration

RQ02_VERSION = "REV363_RQ02_FIVE_HORIZON_LONG_SHORT_V1"
EXIT_HORIZONS = (3, 6, 12, 18)


# ---------------------------------------------------------------------------
# frame / features
# ---------------------------------------------------------------------------
def _rolling_delta(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in EA7:
        idx = df.index[df["country"] == c]
        sub = df.loc[idx, col].astype(float)
        out.loc[idx] = sub - sub.shift(k)
    return out


def build_rq02_frame(prep: dict, spec, *, calibration: dict | None = None) -> pd.DataFrame:
    """Causal RQ02 frame.

    DEV uses expanding prior-only |long|/|short| quantiles; labels never enter a
    feature. OOS must pass the DEV-frozen calibration table; a 2020+ panel with
    no such table is rejected before any feature is emitted.
    """
    df = add_long_short_changes(prep["panel"])
    if calibration is None and len(df) and str(df["month"].min()) >= "2020-01":
        raise ValueError("FROZEN_DEV_CALIBRATION_REQUIRED_FOR_OOS")
    return apply_long_short_votes(df, calibration=calibration)


def attach_rq01_probability(df: pd.DataFrame, prep: dict, models) -> pd.DataFrame:
    out = df.copy()
    m11 = models["M11"]
    p = R1.model_probabilities(prep["design"], m11)
    out["p_rq01_stress"] = 1.0 - p[:, 0]
    out["p_rq01_N"] = p[:, 0]
    out["p_rq01_D"] = p[:, 1]
    out["p_rq01_G"] = p[:, 2]
    out["p_rq01_BOTH"] = p[:, 3]
    return out


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def horizon_labels(df: pd.DataFrame, episodes, h: int, *, allow_oos: bool, end: str):
    onset = L.onset_labels(df, episodes, h, allow_oos=allow_oos)
    fstate = L.future_state_labels(df, episodes, h, allow_oos=allow_oos)
    # GLOBAL NO-U: a censored future state is never a class; such rows are not fit/score eligible
    eligible = ((df["label"] == "N") & (df["month"].map(lambda m: add_months(m, h)) <= end)
                & fstate.isin(CLASSES))
    return onset, fstate, eligible


# ---------------------------------------------------------------------------
# model bundle
# ---------------------------------------------------------------------------
@dataclass
class HorizonModel:
    horizon: str
    h: int
    columns: list
    blocks: dict
    mean: np.ndarray
    scale: np.ndarray
    onset_coef: np.ndarray
    onset_intercept: float
    state_coef: np.ndarray
    state_intercept: np.ndarray
    n_fit: int
    n_events: int
    version: str = RQ02_VERSION

    def to_dict(self):
        return {
            "horizon": self.horizon, "h": self.h, "columns": self.columns,
            "blocks": self.blocks, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "onset_coef": self.onset_coef.tolist(), "onset_intercept": float(self.onset_intercept),
            "state_coef": self.state_coef.tolist(), "state_intercept": self.state_intercept.tolist(),
            "n_fit": int(self.n_fit), "n_events": int(self.n_events), "version": self.version,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(horizon=d["horizon"], h=d["h"], columns=d["columns"], blocks=d["blocks"],
                   mean=np.asarray(d["mean"], float), scale=np.asarray(d["scale"], float),
                   onset_coef=np.asarray(d["onset_coef"], float),
                   onset_intercept=float(d["onset_intercept"]),
                   state_coef=np.asarray(d["state_coef"], float),
                   state_intercept=np.asarray(d["state_intercept"], float),
                   n_fit=d["n_fit"], n_events=d["n_events"], version=d.get("version", RQ02_VERSION))


def _feature_columns(df: pd.DataFrame):
    base = []
    for s in ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P"):
        base += [f"z_{s}", f"d3_{s}", f"d12_{s}", f"dd_{s}",
                 f"long_{s}", f"short_{s}"]
    base += ["intensity", "n_sources_available"]
    base += [f"c_{c}" for c in EA7]
    ind = [f"L_{s}" for s in ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")] + \
          [f"S_{s}" for s in ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")] + \
          ["n_long_sources", "n_short_sources"]
    joint = [f"short_pair_{a}_{b}" for a, b in PAIRS21] + \
            ["long_pair_pass", "n_long_pairs", "n_short_pairs"]
    country = []
    for c in EA7:
        for s in ("CISS", "CLIFS", "DFC", "VIX", "CG", "DSR-P"):
            name = f"cv_{c}_{s}"
            df[name] = df[f"c_{c}"] * df[f"z_{s}"]
            country.append(name)
    return {"BASE": base, "IND": ind, "JOINT": joint, "COUNTRY": country}


def _fit_binary(X, y, C=1.0):
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=8000, random_state=363)
    m.fit(X, y)
    return np.asarray(m.coef_, float).ravel(), float(m.intercept_[0])


def _fit_state(X, y, C=1.0):
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=8000, random_state=363)
    m.fit(X, y)
    return np.asarray(m.coef_, float), np.asarray(m.intercept_, float).ravel()


def _prob_onset(X, coef, intercept):
    z = X @ coef + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _prob_state(X, coef, intercept):
    z = X @ coef.T + intercept[None, :]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_horizon(df, onset, fstate, eligible, cols, horizon, h, *, blocks=("BASE", "IND", "JOINT", "COUNTRY")):
    sub = df[eligible].reset_index(drop=True)
    y_on = np.asarray(onset[eligible], int)
    # REV373 mapped a non-class future state to class 0 (= N); eligible rows now carry a matured class only
    y_st = np.asarray([CLASSES.index(x) for x in fstate[eligible]], int)
    X = np.column_stack([sub[c].astype(float).values for b in blocks for c in cols[b]])
    names = [c for b in blocks for c in cols[b]]
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    Xs = np.nan_to_num((X - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    on_coef, on_int = _fit_binary(Xs, y_on)
    st_coef, st_int = _fit_state(Xs, y_st)
    return HorizonModel(horizon=horizon, h=h, columns=names, blocks={b: cols[b] for b in blocks},
                        mean=mean, scale=sd, onset_coef=on_coef, onset_intercept=on_int,
                        state_coef=st_coef, state_intercept=st_int,
                        n_fit=len(sub), n_events=int(y_on.sum()))


def predict_horizon(model: HorizonModel, df: pd.DataFrame, cols: dict):
    X = np.column_stack([df[c].astype(float).values for c in model.columns])
    Xs = np.nan_to_num((X - model.mean) / model.scale, nan=0.0, posinf=0.0, neginf=0.0)
    return _prob_onset(Xs, model.onset_coef, model.onset_intercept), \
           _prob_state(Xs, model.state_coef, model.state_intercept)


# ---------------------------------------------------------------------------
# S5 proxy, EXIT, Bottom
# ---------------------------------------------------------------------------
def s5_tape(df: pd.DataFrame, score=None, threshold=None) -> pd.DataFrame:
    """Research-only latest Short confirmation S5.

    When the frozen H6 onset score is supplied, S5 is the latest month whose
    H6 score crossed the frozen cut.  Otherwise the source-count rule is used.
    This is a research proxy, not an authentic issuer stamp.
    """
    out = []
    for c in EA7:
        sub = df[df["country"] == c].sort_values("month")
        last = None
        prev = -1.0
        for idx, r in sub.iterrows():
            if score is not None and threshold is not None:
                cur = float(score[idx])
                confirmed = bool(cur >= threshold and prev < threshold)
                prev = cur
            else:
                cur = float(r["n_short_sources"] >= 4 and r["n_short_pairs"] >= 5)
                confirmed = bool(cur >= 1 and prev < 1)
                prev = cur
            if confirmed:
                last = r["month"]
            age = (None if last is None else
                   (int(r["month"][:4]) * 12 + int(r["month"][5:7]) -
                    int(last[:4]) * 12 - int(last[5:7])))
            out.append({"country": c, "issue": r["month"], "label": r["label"],
                        "s5_last_short": last,
                        "s5_confirmed": pd.NA,
                        "s5_status": "NOT_SPEC_S5__H6_SCORE_TRIGGER_PROXY_ONLY",
                        "s5_proxy_trigger_18m": int(last is not None and age is not None and age <= 18),
                        "s5_age_months": age,
                        "s5_h6_score": (None if score is None else float(score[idx])),
                        "s5_h6_threshold": threshold,
                        "n_short_sources": (None if pd.isna(r["n_short_sources"])
                                            else int(r["n_short_sources"])),
                        "n_short_pairs": (None if pd.isna(r["n_short_pairs"])
                                          else int(r["n_short_pairs"]))})
    return pd.DataFrame(out)


def exit_labels(df: pd.DataFrame, episodes, h: int, *, allow_oos: bool, end: str):
    tte = L.time_to_exit_labels(df, episodes, allow_oos=allow_oos)
    event, eligible = [], []
    for i, r in df.iterrows():
        if r["label"] == "N":
            event.append(None)
            eligible.append(False)
            continue
        t = tte.iloc[i]
        mature = add_months(r["month"], h) <= end
        eligible.append(bool(mature))
        if t is None or not mature:
            event.append(None)
        else:
            event.append(int(t <= h))
    return pd.Series(event, index=df.index), pd.Series(eligible, index=df.index)


def fit_exit(df, event, eligible, cols, h, *, use_rq01: bool, horizon_tag: str):
    sub = df[eligible].reset_index(drop=True)
    y = np.asarray(event[eligible], int)
    use_cols = []
    for c in cols["BASE"] + cols["IND"] + cols["JOINT"] + cols["COUNTRY"]:
        use_cols.append(c)
    if use_rq01:
        use_cols += ["p_rq01_stress", "p_rq01_G", "p_rq01_BOTH", "pool_count", "n_up"]
    X = np.column_stack([sub[c].astype(float).values for c in use_cols])
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    Xs = np.nan_to_num((X - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=8000, random_state=363)
    m.fit(Xs, y)
    p = m.predict_proba(Xs)[:, 1]
    return {"horizon": h, "tag": horizon_tag, "n": len(sub), "n_events": int(y.sum()),
            "n_censored": int(len(sub) - y.sum()),
            "columns": use_cols, "mean": mean.tolist(), "scale": sd.tolist(),
            "coef": np.asarray(m.coef_, float).ravel().tolist(),
            "intercept": float(m.intercept_[0]), "p_fit": p.tolist(),
            "y_fit": y.tolist(), "keys": sub[["country", "month"]].astype(str).agg("|".join, axis=1).tolist()}


def bottom_analysis(dataset_root, out_dir, panel=None):
    """Three-family Bottom zones on the canonical DEV weekly tape (spec RQ02 Bottom 3F).

    F1 = issued EWI cooling; F2 = peer recovery (6 peers, own country excluded); F3 = ASSET-SPECIFIC
    return-volatility cooling + 4-week price rebound, from the newest week <= the issue.  The zone
    starts strictly after the issue; the ex-post trough is the lowest INDEX LEVEL in a strictly future
    window and is used ONLY to score HIT/MISS.  All logic lives in research_adapters.rq02_bottom_*;
    this wrapper only loads the DEV tape and writes the CSVs.  Issue selection is still ex-post
    (episode start + 3 months) and every row says so: this is a zone-quality harness, not an issued tape.
    """
    from research_adapters import rq02_bottom_dev_harness as BH
    from .data import _exact
    root = find_dataset_root(dataset_root)
    tape = pd.read_csv(_exact(root, "V8_RQ03_SIX_ASSET_DEV_WEEKLY.csv"))
    tape["week_end"] = pd.to_datetime(tape["week_end"])
    weekly = tape.set_index("week_end").sort_index()
    episodes = [e for e in L.load_episodes(root)
                if e["country"] in EA7 and e["dev_eligibility"] == "DEV_CHRONOLOGY"]
    if panel is None:
        raise ValueError("BOTTOM_REQUIRES_RQ01_PANEL_FOR_F1_F2")
    zd = BH.bottom_dev_zones(panel, episodes, weekly)
    troughs = BH.realised_troughs(episodes, weekly)
    zd.to_csv(Path(out_dir) / "RQ02_DEV_BOTTOM_ZONES.csv", index=False)
    troughs.to_csv(Path(out_dir) / "RQ02_DEV_BOTTOM_TROUGHS.csv", index=False)
    summary = BH.summarize_bottom(zd)
    summary.update({"n_episodes": len(episodes), "n_zone_records": len(zd),
                    "mean_interval_score": summary.get("mean_interval_score_weeks")})
    return summary, troughs, zd


# ---------------------------------------------------------------------------
# required Long/Short matrices
# ---------------------------------------------------------------------------
def spec_s5_tape(dataset_root, df, episodes, out_dir, *, raw=None, short_vote_mode="IJP"):
    """Run the spec S5 engine beside the legacy H6-trigger proxy.

    The primary Short vote now comes from the IJP kernel (7 Individual Short +
    21 paired Joint Short + COMPLEMENT_AND_JOINT Pool) on released native rows.
    The source-count rule is retained only as a labelled fallback. The
    risk-peak/RoundBack/last-Short chain is the canonical S5 engine. This is
    retrospective and never an H6 feature.
    """
    from .ijp_s5 import native_rows_from_raw, s5_short_votes
    from .spec_bridges import build_spec_s5

    root = find_dataset_root(dataset_root)
    tape = pd.read_csv(_exact(root, "V8_RQ03_SIX_ASSET_DEV_WEEKLY.csv"))
    tape["week_end"] = pd.to_datetime(tape["week_end"])
    weekly = tape.set_index("week_end")[["E1", "E2", "E3"]].astype(float).sort_index()
    if weekly.isna().any().any() or (weekly <= -1).any().any():
        raise ValueError("S5_WEEKLY_RETURNS_FINITE_AND_ABOVE_MINUS_ONE_REQUIRED")
    prices = (1.0 + weekly).cumprod()
    rows = []
    for e in episodes:
        country = e.get("country")
        if country not in EA7 or e.get("dev_eligibility") != "DEV_CHRONOLOGY":
            continue
        onset = pd.Timestamp(e["start"])
        h6 = onset - pd.DateOffset(months=18)
        prior = prices.index[prices.index <= h6]
        base = {"country": country, "onset": onset.date().isoformat(),
                "h6": None if len(prior) == 0 else prior[-1].date().isoformat(),
                "role": "RETROSPECTIVE_NOT_A_H6_FEATURE",
                "short_vote_source": "INTEGRATED_SOURCE_COUNTS_NOT_OWNER_SIGNED"}
        if len(prior) == 0:
            rows.append({**base, "status": "NOT_EVALUATED",
                         "reason": "H6_BEFORE_FIRST_NATIVE_WEEK"})
            continue
        h6_week = prior[-1]
        base = {"country": country, "onset": onset.date().isoformat(),
                "h6": h6_week.date().isoformat(),
                "role": "RETROSPECTIVE_NOT_A_H6_FEATURE"}
        votes = None
        source_label = "INTEGRATED_SOURCE_COUNTS_NOT_OWNER_SIGNED"
        ijp_error = None
        if short_vote_mode == "IJP" and raw is not None:
            try:
                native_rows = native_rows_from_raw(raw, country)
                months = df.loc[df["country"] == country, "month"].tolist()
                ijp_votes = s5_short_votes(native_rows, months)
                if len(ijp_votes):
                    votes = ijp_votes[["issue", "vote"]]
                    source_label = "IJP_7_IND_21_JOINT_COMPLEMENT_RESEARCH_PROFILE"
                else:
                    ijp_error = "IJP_NO_EVALUATED_ISSUES"
            except Exception as exc:  # keep the labelled fallback, never silence the failure
                ijp_error = f"IJP_FAILED:{type(exc).__name__}:{exc}"
        if votes is None:
            sub = df[df["country"] == country].copy()
            sub["issue"] = pd.to_datetime(sub["month"]) + pd.offsets.MonthEnd(0)
            votes = sub[["issue", "n_short_sources", "n_short_pairs"]].dropna().copy()
            votes["vote"] = ((votes["n_short_sources"] >= 4) &
                             (votes["n_short_pairs"] >= 5)).astype(int)
            votes = votes[["issue", "vote"]]
        base["short_vote_source"] = source_label
        if ijp_error:
            base["ijp_status"] = ijp_error
        try:
            rec = build_spec_s5(prices, h6=h6_week, onset=onset,
                                short_votes=votes[["issue", "vote"]])
            rows.append({**base, **{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                                    for k, v in rec.items() if k != "portfolio_weights"}})
        except ValueError as exc:
            rows.append({**base, "status": "NOT_EVALUATED", "reason": str(exc)})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out_dir) / "RQ02_DEV_S5_SPEC.csv", index=False)
    n_confirmed = int((frame.get("status") == "CONFIRMED").sum()) if len(frame) else 0
    n_ijp = int(frame.get("short_vote_source", pd.Series(dtype=str)).astype(str)
                .str.startswith("IJP_").sum()) if len(frame) else 0
    return {"n_episodes": int(len(frame)), "n_confirmed": n_confirmed,
            "n_not_evaluated": int((frame.get("status") == "NOT_EVALUATED").sum()) if len(frame) else 0,
            "n_ijp_short_vote_units": n_ijp,
            "short_vote_mode": short_vote_mode,
            "evidence": "RQ02_DEV_S5_SPEC.csv",
            "spec_engine": "research_adapters.rq02_s5_buyhold_var_es.s5_endpoints",
            "proxy_preserved": "RQ02_DEV_S5.csv"}


def source_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SOURCES7:
        rows.append({
            "source": s,
            "n_long": int((df[f"L_{s}"] == 1).sum()),
            "n_short": int((df[f"S_{s}"] == 1).sum()),
            "mean_long_stress": float(df.loc[df["stress"] == 1, f"L_{s}"].mean()),
            "mean_long_N": float(df.loc[df["stress"] == 0, f"L_{s}"].mean()),
            "mean_short_stress": float(df.loc[df["stress"] == 1, f"S_{s}"].mean()),
            "mean_short_N": float(df.loc[df["stress"] == 0, f"S_{s}"].mean()),
            "mean_long_delta_stress": float(df.loc[df["stress"] == 1, f"long_{s}"].mean()),
            "mean_short_delta_stress": float(df.loc[df["stress"] == 1, f"short_{s}"].mean()),
        })
    return pd.DataFrame(rows)


def pair_motif_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """21 Short pairs x 9 motifs with a DEV selected/rejected policy."""
    rows = []
    for a, b in PAIRS21:
        col = f"p21_{a}_{b}_state"
        for motif in R2_MOTIFS:
            m = (df[col] == motif)
            rows.append({"pair": f"{a}-{b}", "type": "SHORT", "motif": motif,
                         "n": int(m.sum()),
                         "n_stress": int((m & (df["stress"] == 1)).sum()),
                         "n_N": int((m & (df["stress"] == 0)).sum()),
                         "pass_rate_stress": float(df.loc[m & (df["stress"] == 1), col].shape[0] /
                                                   max(int((df["stress"] == 1).sum()), 1)),
                         "pass_rate_N": float(df.loc[m & (df["stress"] == 0), col].shape[0] /
                                              max(int((df["stress"] == 0).sum()), 1))})
    out = pd.DataFrame(rows)
    # selected: top 10 pairs by UU stress-minus-N lift, ties by pair id
    uu = out[out["motif"] == "UU"].copy()
    uu["lift"] = uu["pass_rate_stress"] - uu["pass_rate_N"]
    chosen = set(uu.sort_values(["lift", "pair"], ascending=[False, True]).head(10)["pair"])
    out["selected"] = out["pair"].isin(chosen)
    out["decision"] = np.where(out["selected"], "SELECTED", "REJECTED")
    return out


R2_MOTIFS = tuple(a + b for a in "UDS" for b in "UDS")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_dev(dataset_root, out_dir, *, rq01_freeze=None,
            phase_source="LEGACY", rq01_canonical_phase=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # load frozen RQ01 M11 for the stacked state feature
    freeze_path = Path(rq01_freeze) if rq01_freeze else None
    if freeze_path is None:
        from .config import FREEZE_FILE
        freeze_path = FREEZE_FILE
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    models = MOD.deserialize_models(freeze["models"])
    from .features import FeatureSpec
    rq01_spec = FeatureSpec.from_dict(freeze["feature_spec"])
    # RQ02 candidate calendar starts 2005-01, six months before the RQ01 fit start.
    prep = R1.prepare_panel(dataset_root, allow_oos=False, row_start="2005-01",
                            row_end=DEV_END, data_end=DEV_END, spec=rq01_spec)
    spec = prep["spec"]
    episodes = prep["episodes"]
    df = build_rq02_frame(prep, spec)
    df = attach_rq01_probability(df, prep, models)
    cols = _feature_columns(df)

    # horizon models
    horizon_out = {}
    for hname in HORIZONS:
        h = HORIZON_MONTHS[hname]
        onset, fstate, eligible = horizon_labels(df, episodes, h, allow_oos=False, end=DEV_END)
        # Four-M hierarchy: M00 BASE, M01 +IND, M10 +JOINT, M11 all + country
        block_sets = {"M00": ("BASE",), "M01": ("BASE", "IND"),
                      "M10": ("BASE", "JOINT"),
                      "M11": ("BASE", "IND", "JOINT", "COUNTRY")}
        mout = {}
        for mname, blocks in block_sets.items():
            model = fit_horizon(df, onset, fstate, eligible, cols, hname, h, blocks=blocks)
            p_on, p_st = predict_horizon(model, df, cols)
            sub = df[eligible].reset_index(drop=True)
            y_on = np.asarray(onset[eligible], int)
            y_st = np.asarray([CLASSES.index(x) for x in fstate[eligible]], int)
            s_on = p_on[eligible]
            p_state_elig = p_st[eligible]
            tr = THR.exact_threshold_search(s_on, y_on)
            pred = (s_on >= tr.threshold).astype(int)
            cm = MET.confusion_binary(y_on, pred)
            ll = MET.multiclass_logloss(y_st, p_state_elig)
            br = MET.multiclass_brier(y_st, p_state_elig)
            auc = MET.auc_score(y_on, s_on)
            mout[mname] = {
                "model": model.to_dict(),
                "n_eligible": int(eligible.sum()), "n_events": int(y_on.sum()),
                "event_anchors": len(set(sub.loc[y_on == 1, "country"] + "|" +
                                        sub.loc[y_on == 1, "month"])),
                "onset": {"logloss": float(-np.mean(np.log(np.clip(
                    np.where(y_on == 1, s_on, 1 - s_on), 1e-15, 1))),),
                    "brier": float(np.mean((s_on - y_on) ** 2)),
                    "auc": auc, "threshold": tr.__dict__,
                    "confusion_threshold": cm},
                "state": {"logloss": ll, "brier": br,
                          "confusion4": MET.confusion4(y_st, p_state_elig.argmax(axis=1)),
                          "class_counts": {c: int((y_st == i).sum()) for i, c in enumerate(CLASSES)}},
                "score": s_on.tolist(), "state_p": p_state_elig.tolist(),
                "eligible_index": sub.index.tolist(),
            }
        horizon_out[hname] = {"h": h, "n_eligible": int(eligible.sum()),
                              "n_events": int(onset[eligible].sum()),
                              "models": mout,
                              "eligible_keys": df.loc[eligible, ["country", "month"]].astype(str)
                              .agg("|".join, axis=1).tolist()}

    # S5 research proxy: use the frozen H6 M11 onset score when available
    h6_model = HorizonModel.from_dict(horizon_out["H6"]["models"]["M11"]["model"])
    h6_score_all, _ = predict_horizon(h6_model, df, cols)
    h6_thr = horizon_out["H6"]["models"]["M11"]["onset"]["threshold"]["threshold"]
    s5 = s5_tape(df, h6_score_all, h6_thr)
    s5.to_csv(out / "RQ02_DEV_S5.csv", index=False)
    # This is a rolling H6-score trigger, NOT the spec S5 (Buy-and-Hold VaR/ES peak -> RoundBack -> last Short
    # of the seven EWI + 21 Joint).  The real S5 lives in research_adapters/rq02_s5_buyhold_var_es.py and its
    # end-to-end run is NOT_RUN, so nothing here may be called "confirmed".
    s5_summary = {"n_rows": len(s5), "n_confirmed": None,
                  "n_proxy_trigger": int(s5["s5_proxy_trigger_18m"].sum()),
                  "n_no_trigger": int((s5["s5_proxy_trigger_18m"] == 0).sum()),
                  "official_issued": False, "proxy": True,
                  "status": "RESEARCH_PROXY_NOT_SPEC_S5_NOT_OWNER_CERTIFIED"}

    # EXIT-A and EXIT-B
    exit_out = {}
    for h in EXIT_HORIZONS:
        event, eligible = exit_labels(df, episodes, h, allow_oos=False, end=DEV_END)
        a = fit_exit(df, event, eligible, cols, h, use_rq01=False, horizon_tag=f"EXIT_A_H{h}")
        b = fit_exit(df, event, eligible, cols, h, use_rq01=True, horizon_tag=f"EXIT_B_LEGACY_STACKED_H{h}")
        a["definition"] = "EXIT_A_POOL_SPEC"
        b["definition"] = "A_PLUS_RQ01_STACKED_LEGACY_NOT_SPEC_B"
        for tag, rec in (("A", a), ("B", b)):
            y = np.asarray(rec["y_fit"], int)
            p = np.asarray(rec["p_fit"], float)
            rec["auc"] = MET.auc_score(y, p)
            rec["logloss"] = float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-15, 1))))
            rec["confusion_0_5"] = MET.confusion_binary(y, (p >= 0.5).astype(int))
            rec["threshold"] = THR.exact_threshold_search(p, y).__dict__
        exit_out[f"EXIT_{h}"] = {"A": a, "B": b,
                                 "delta_auc_B_minus_A": (b["auc"] - a["auc"]) if
                                 (a["auc"] is not None and b["auc"] is not None) else None,
                                 "delta_logloss_B_minus_A": (b["logloss"] - a["logloss"]) if
                                 (a["logloss"] is not None and b["logloss"] is not None) else None}

    # longitudinal override ledger
    rq01_tape = pd.read_csv(Path(freeze_path).parent.parent / "RESULTS" / "RQ01" / "RQ01_DEV_ISSUE_TAPE.csv") \
        if (Path(freeze_path).parent.parent / "RESULTS" / "RQ01" / "RQ01_DEV_ISSUE_TAPE.csv").exists() else None
    if rq01_tape is None:
        # build from the frozen RQ01 model directly
        p = R1.model_probabilities(prep["design"], models["M11"])
        rq01_tape = pd.DataFrame({"country": df["country"], "issue": df["month"],
                                  "score": 1 - p[:, 0]})
    thr = float(freeze["selected_threshold"]["threshold"])
    rq01_tape["pred_stress"] = (rq01_tape["score"] >= thr).astype(int)
    ledger = df[["country", "month", "label"]].merge(
        rq01_tape[["country", "issue", "score", "pred_stress"]],
        left_on=["country", "month"], right_on=["country", "issue"], how="left")
    ledger["original_state"] = ledger["label"]
    ledger["effective_state"] = np.where(ledger["pred_stress"] == 1, "Stress", "N")
    ledger["override"] = (ledger["original_state"] == "N") & (ledger["effective_state"] == "Stress")
    ledger["reverse_override"] = (ledger["original_state"] != "N") & (ledger["effective_state"] == "N")
    ledger.to_csv(out / "RQ02_DEV_OVERRIDE_LEDGER.csv", index=False)
    override_summary = {"n_original_N": int((ledger["original_state"] == "N").sum()),
                        "n_N_to_stress": int(ledger["override"].sum()),
                        "n_stress_to_N": int(ledger["reverse_override"].sum()),
                        "n_total": len(ledger)}

    # Optional canonical RQ01 phase source for Bottom F1/F2; per-row legacy fallback.
    phase_source_summary = None
    if phase_source == "CANONICAL":
        if not rq01_canonical_phase:
            raise ValueError("CANONICAL_PHASE_FILE_REQUIRED")
        from .phase_source import apply_canonical_phase, load_canonical_phase
        canonical = load_canonical_phase(rq01_canonical_phase)
        df, phase_source_summary = apply_canonical_phase(df, canonical)
    elif phase_source != "LEGACY":
        raise ValueError("PHASE_SOURCE_MUST_BE_LEGACY_OR_CANONICAL")

    # bottom
    bottom_summary, troughs, zones = bottom_analysis(dataset_root, out, panel=df)
    s5_spec_summary = spec_s5_tape(dataset_root, df, episodes, out, raw=prep["raw"])

    # F1-F5 table across horizons and four M
    f_rows = []
    for hname, hrec in horizon_out.items():
        base_ll = hrec["models"]["M00"]["onset"]["logloss"]
        base_br = hrec["models"]["M00"]["onset"]["brier"]
        for mname, mv in hrec["models"].items():
            n = mv["n_eligible"]
            k = len(mv["model"]["columns"])
            nmask = mv["onset"]["threshold"]["n_unique_masks"]
            f1 = mv["onset"]["logloss"]
            f2 = mv["onset"]["brier"]
            f4 = f1 + (k / (2.0 * max(n, 1))) * math.log(max(n, 2)) + math.log(max(nmask, 1)) / max(n, 1)
            f_rows.append({"horizon": hname, "model": mname, "n": n,
                           "events": mv["n_events"], "F1_logloss": f1,
                           "F2_brier": f2, "F3_auc": mv["onset"]["auc"],
                           "F4_selection_aware": f4,
                           "F5_delta_logloss_vs_M00": f1 - base_ll,
                           "F5_delta_brier_vs_M00": f2 - base_br,
                           "state_logloss": mv["state"]["logloss"],
                           "state_brier": mv["state"]["brier"],
                           "n_masks": nmask, "selected": mname == "M11"})
    # full Long/Short source matrix and 21-pair x 9-motif matrix
    srcm = source_matrix(df)
    pairm = pair_motif_matrix(df)
    srcm.to_csv(out / "RQ02_DEV_SOURCE_MATRIX.csv", index=False)
    pairm.to_csv(out / "RQ02_DEV_PAIR_MOTIFS.csv", index=False)

    status = [
        {"id": "R1", "output": "Long CG/DSR-P and Short seven-source native cumulative matrices",
         "status": "COMPLETE", "unresolved": "QUARTERLY_CARRY_NOT_A_NEW_NATIVE_OBSERVATION",
         "evidence": "RQ02_DEV_SOURCE_MATRIX.csv"},
        {"id": "R2", "output": "Joint 1 Long + 21 Short pairs, chosen/rejected",
         "status": "COMPLETE", "unresolved": "FIXED_UU_RESEARCH_MOTIF_NOT_OWNER_SELECTED",
         "evidence": "RQ02_DEV_PAIR_MOTIFS.csv"},
        {"id": "R3", "output": "Four-M binary first onset and four-class future state at five horizons",
         "status": "COMPLETE", "unresolved": "DEV_APPARENT_NOT_OOS",
         "evidence": "RQ02_DEV_ONSET_METRICS.csv"},
        {"id": "R4", "output": "Last Short S5 with issued confirmation",
         "status": "COMPLETE_RESEARCH_PROXY", "unresolved": "NO_OFFICIAL_ISSUER_STAMP",
         "evidence": "RQ02_DEV_S5.csv"},
        {"id": "R5", "output": "EXIT-A first-N H3/H6/H12/H18 and censored risk set",
         "status": "COMPLETE", "unresolved": "CENSORING_AND_DEPENDENT_MONTHS",
         "evidence": "RQ02_DEV_EXIT_AB.csv"},
        {"id": "R6", "output": "EXIT-B legacy stacked diagnostic (spec state-anchor B bridged but not evaluated without a canonical issued state tape)",
         "status": "COMPLETE_RESEARCH_PROXY",
         "unresolved": "EXIT_B_LEGACY_A_PLUS_RQ01_NOT_SPEC__CANONICAL_STATE_TAPE_MISSING",
         "evidence": "RQ02_DEV_EXIT_AB.csv"},
        {"id": "R7", "output": "Original-N to confirmed Stress longitudinal override ledger",
         "status": "COMPLETE", "unresolved": "RESEARCH_ASOF_NOT_OFFICIAL_ISSUED",
         "evidence": "RQ02_DEV_OVERRIDE_LEDGER.csv"},
        {"id": "R8", "output": "Bottom three families F1/F2/F3 plus issue-time 5-asset zones",
         "status": "COMPLETE_RESEARCH_PROXY", "unresolved": "F1/F2_ARE_CAUSAL_RESEARCH_STATES_NOT_OWNER_GOLD",
         "evidence": "RQ02_DEV_BOTTOM_ZONES.csv"},
        {"id": "R9", "output": "Bottom HIT/MISS/NO_ZONE/IMMATURE, interval score, width days",
         "status": "COMPLETE_RESEARCH_PROXY", "unresolved": "SIX_DEV_EPISODES_LIMIT_SUPPORT",
         "evidence": "RQ02_DEV_BOTTOM_ZONES.csv"},
        {"id": "R10", "output": "F1-F5 penalty and conditional-subtype support",
         "status": "COMPLETE", "unresolved": "SMALL_EPISODE_SUPPORT",
         "evidence": "RQ02_DEV_F1_F5.csv"},
        {"id": "R11", "output": "OOS first-onset/EXIT/Bottom final evaluated issue tape",
         "status": "PENDING_OOS_ONE_SHOT", "unresolved": "OOS_NOT_OPENED_IN_DEV_RUN",
         "evidence": "RQ02_OOS_RESULTS.json"},
    ]

    results = {
        "run_id": f"REV363_RQ02_DEV_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "version": RQ02_VERSION, "claim": "DEV_APPARENT_RESEARCH_NOT_OFFICIAL_ISSUED",
        "population": {"n_panel": len(df), "n_countries": len(EA7),
                       "horizons": {k: {"n_eligible": v["n_eligible"], "n_events": v["n_events"]}
                                    for k, v in horizon_out.items()}},
        "horizons": {k: {"h": v["h"], "n_eligible": v["n_eligible"], "n_events": v["n_events"],
                         "models": {m: {kk: vv for kk, vv in mv.items()
                                        if kk not in ("score", "state_p", "eligible_index", "model")}
                                    for m, mv in v["models"].items()}}
                     for k, v in horizon_out.items()},
        "s5": s5_summary,
        "s5_spec": s5_spec_summary,
        "phase_source": {"mode": phase_source, "summary": phase_source_summary},
        "spec_exit_b": {
            "status": "NOT_EVALUATED",
            "reason": "CANONICAL_RQ01_STATE_TAPE_NOT_AVAILABLE",
            "bridge": "rev363.spec_bridges.build_spec_exit_b",
            "legacy_integrated_branch": "EXIT_B_LEGACY_STACKED_A_PLUS_RQ01_NOT_SPEC",
            "role": "SPEC_STATE_ANCHOR_B_IS_NOT_REPLACED_BY_THE_LEGACY_STACK"},
        "exit": {k: {kk: vv for kk, vv in v.items()} for k, v in exit_out.items()},
        "override": override_summary,
        "bottom": bottom_summary,
        "source_matrix": srcm.to_dict(orient="records"),
        "pair_motif_matrix_rows": len(pairm),
        "pair_motif_selected": int(pairm["selected"].sum()),
        "F1_F5": f_rows,
        "requirements": status,
        "n_requirements_complete": sum(1 for s in status if s["status"].startswith("COMPLETE")),
        "unresolved": [s["id"] for s in status if not s["status"].startswith("COMPLETE")],
        "scientific_boundary": "No OOS tuning; S5/Bottom are research proxies where issuer/owner proof is absent.",
    }

    # CSV outputs
    pd.DataFrame(f_rows).to_csv(out / "RQ02_DEV_F1_F5.csv", index=False)
    # onset metrics
    rows = []
    for hname, hrec in horizon_out.items():
        for m, mv in hrec["models"].items():
            rows.append({"horizon": hname, "model": m, "n": mv["n_eligible"],
                         "events": mv["n_events"], "auc": mv["onset"]["auc"],
                         "logloss": mv["onset"]["logloss"], "brier": mv["onset"]["brier"],
                         **{k: mv["onset"]["confusion_threshold"][k] for k in
                            ("TP", "FN", "FP", "TN", "FNR", "FPR")},
                         "threshold": mv["onset"]["threshold"]["threshold"],
                         "gate_feasible": mv["onset"]["threshold"]["gate_feasible"]})
    pd.DataFrame(rows).to_csv(out / "RQ02_DEV_ONSET_METRICS.csv", index=False)
    # future state metrics
    rows = []
    for hname, hrec in horizon_out.items():
        for m, mv in hrec["models"].items():
            rows.append({"horizon": hname, "model": m, "n": mv["n_eligible"],
                         "logloss_state": mv["state"]["logloss"],
                         "brier_state": mv["state"]["brier"],
                         **{f"class_{c}": mv["state"]["class_counts"][c] for c in CLASSES}})
    pd.DataFrame(rows).to_csv(out / "RQ02_DEV_FUTURE_STATE_METRICS.csv", index=False)
    # long/short matrices
    mat = []
    for a, b in PAIRS21:
        mat.append({"pair": f"{a}-{b}", "type": "SHORT", "n_uu": int(df[f"short_pair_{a}_{b}"].sum()),
                    "mean_when_stress": float(df.loc[df["stress"] == 1, f"short_pair_{a}_{b}"].mean()),
                    "mean_when_N": float(df.loc[df["stress"] == 0, f"short_pair_{a}_{b}"].mean())})
    mat.append({"pair": "CG-DSR-P", "type": "LONG", "n_uu": int(df["long_pair_pass"].sum()),
                "mean_when_stress": float(df.loc[df["stress"] == 1, "long_pair_pass"].mean()),
                "mean_when_N": float(df.loc[df["stress"] == 0, "long_pair_pass"].mean())})
    pd.DataFrame(mat).to_csv(out / "RQ02_DEV_LONG_SHORT_MATRICES.csv", index=False)
    # EXIT table
    ex = []
    for k, v in exit_out.items():
        for tag in ("A", "B"):
            r = v[tag]
            ex.append({"exit": k, "model": tag, "n": r["n"], "events": r["n_events"],
                       "n_censored": r.get("n_censored"), "auc": r["auc"], "logloss": r["logloss"],
                       "definition": r.get("definition"),
                       **{kk: r["confusion_0_5"][kk] for kk in ("TP", "FN", "FP", "TN", "FNR", "FPR")}})
    pd.DataFrame(ex).to_csv(out / "RQ02_DEV_EXIT_AB.csv", index=False)
    # freeze RQ02 models
    freeze02 = {
        "freeze_version": "REV363_RQ02_DEV_FREEZE_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rq01_freeze_sha256": freeze.get("freeze_sha256"),
        "horizons": {k: {m: mv["model"] for m, mv in v["models"].items()}
                     for k, v in horizon_out.items()},
        "thresholds": {k: {m: mv["onset"]["threshold"]["threshold"]
                           for m, mv in v["models"].items()}
                       for k, v in horizon_out.items()},
        "s5": s5_summary, "s5_spec": s5_spec_summary, "bottom": bottom_summary,
        "phase_source": {"mode": phase_source, "summary": phase_source_summary},
        "long_short_calibration": df.attrs.get("long_short_calibration", final_calibration(df)),
        "long_short_calibration_mode": df.attrs.get("long_short_calibration_mode"),
        "long_short_calibration_rule": "EXPANDING_PRIOR_ABS_Q75_NO_LABELS",
        "exit": {k: {"A": {kk: vv for kk, vv in v["A"].items() if kk not in ("p_fit", "y_fit")},
                     "B": {kk: vv for kk, vv in v["B"].items() if kk not in ("p_fit", "y_fit")}}
                 for k, v in exit_out.items()},
        "requirements_complete": results["n_requirements_complete"],
    }
    freeze02["freeze_sha256"] = sha256_json(freeze02)
    (out / "RQ02_DEV_FREEZE.json").write_text(json.dumps(freeze02, indent=2, default=str) + "\n", encoding="utf-8")
    results["freeze_sha256"] = freeze02["freeze_sha256"]
    (out / "RQ02_DEV_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "RQ02_DEV_REQUIREMENTS_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    evidence = {
        "run_id": results["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m rev363.rq02 --dataset-root <locked> --out <RESULTS/RQ02>",
        "dataset_root": str(find_dataset_root(dataset_root)),
        "rq01_freeze": str(freeze_path), "rq01_freeze_sha256": freeze.get("freeze_sha256"),
        "n_panel": len(df), "horizons": results["population"]["horizons"],
        "oos_read": False, "oos_status": "SEALED",
        "outputs_sha256": {p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.glob("RQ02_DEV_*")) if p.is_file()},
    }
    (out / "EXECUTION_EVIDENCE.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rq01-freeze")
    p.add_argument("--phase-source", choices=("LEGACY", "CANONICAL"), default="LEGACY")
    p.add_argument("--rq01-canonical-phase")
    a = p.parse_args(argv)
    r = run_dev(a.dataset_root, a.out, rq01_freeze=a.rq01_freeze,
                phase_source=a.phase_source, rq01_canonical_phase=a.rq01_canonical_phase)
    print(json.dumps({"run_id": r["run_id"], "horizons": r["population"]["horizons"],
                      "n_requirements_complete": r["n_requirements_complete"],
                      "unresolved": r["unresolved"]}, indent=2, default=str))


if __name__ == "__main__":
    main()

