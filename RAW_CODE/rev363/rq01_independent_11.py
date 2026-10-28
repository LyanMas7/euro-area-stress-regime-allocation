"""RQ01-11 independent matched-sample feature/estimator validation.

The retained 01-11 file is a numerical parity check of the *same* model and is
reported separately.  This module is deliberately independent:

* JOINT features are rebuilt from the issued motif packet (one column per
  registered pair configuration, NaN = U), not from the proxy JOINT columns;
* BASE and IND columns are held fixed so M00/M01/M10/M11 share one baseline;
* the primary estimator is statsmodels MNLogit (unregularised MLE, Newton/BFGS),
  a different implementation from the retained sklearn LogisticRegression;
  a separately configured class-balanced sklearn challenger is reported too;
* every metric uses the same country-month keys, the same 4-class target and the
  same scorable OOS mask;
* block-bootstrap intervals by country accompany the matched loss contrasts;
* numerical parity is never relabelled as independent validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rev363 import metrics as MET
from rev363 import panel as PNL
from rev363 import thresholds as THR
from rev363.config import CLASSES
from rev363.util import sha256_json

try:  # pragma: no cover - availability is reported
    import statsmodels.api as sm
    SM_AVAILABLE = True
    SM_VERSION = sm.__version__
except Exception:  # pragma: no cover
    sm = None
    SM_AVAILABLE = False
    SM_VERSION = None

from sklearn.linear_model import LogisticRegression

CHALLENGER = {
    "family": "LOGISTIC",
    "implementation": "sklearn.linear_model.LogisticRegression",
    "solver": "lbfgs",
    "C": 0.35,
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 363,
}
PRIMARY = {
    "family": "MULTINOMIAL_LOGIT",
    "implementation": "statsmodels.api.MNLogit",
    "regularisation": "NONE_MLE",
    "solver": "newton_then_bfgs",
    "add_constant": True,
    "version": SM_VERSION,
}
FORCED_SCORABLE = "SCORABLE"


def _matrix(df, cols):
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float)
                            for c in cols])


def _standardise_fit(X):
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(X), X, med)
    sd = np.nanstd(filled, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return med, sd


def _standardise_apply(X, med, sd):
    return (np.where(np.isfinite(X), X, med) - med) / sd


def _fit_primary(X, y4):
    """statsmodels MNLogit MLE on standardised features plus a constant."""
    if not SM_AVAILABLE:
        raise RuntimeError("STATSMODELS_UNAVAILABLE")
    med, sd = _standardise_fit(X)
    Xs = sm.add_constant(_standardise_apply(X, med, sd), has_constant="add")
    model = sm.MNLogit(np.asarray(y4, int), Xs)
    last = None
    for method in ("newton", "bfgs", "lbfgs"):
        try:
            res = model.fit(method=method, maxiter=800, disp=False)
            if np.all(np.isfinite(res.params)):
                return {"kind": "MNLOGIT", "result": res, "med": med, "sd": sd,
                        "solver_used": method}
        except Exception as exc:  # singular / non-converged starts
            last = exc
    raise RuntimeError(f"MNLOGIT_FIT_FAILED:{type(last).__name__}:{last}")


def _fit_challenger(X, y4):
    med, sd = _standardise_fit(X)
    Xs = _standardise_apply(X, med, sd)
    model = LogisticRegression(**{k: v for k, v in CHALLENGER.items()
                                  if k not in ("family", "implementation")})
    model.fit(Xs, y4)
    return {"kind": "SKLEARN", "result": model, "med": med, "sd": sd,
            "solver_used": "lbfgs"}


def _p4(fit, X):
    Xs = _standardise_apply(np.asarray(X, float), fit["med"], fit["sd"])
    if fit["kind"] == "MNLOGIT":
        res = fit["result"]
        Xc = sm.add_constant(Xs, has_constant="add")
        params = np.asarray(res.params, float)
        eta = np.asarray(Xc, float) @ params
        if eta.ndim == 1:
            eta = eta.reshape(-1, 1)
        # Stable softmax with class 0 as the base (statsmodels parameterisation).
        logits = np.column_stack([np.zeros(len(Xs)), eta])
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        out = e / e.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("MNLOGIT_NONFINITE_PREDICTION")
        return out
    raw = fit["result"].predict_proba(Xs)
    out = np.zeros((len(Xs), 4))
    for j, cls in enumerate(fit["result"].classes_):
        out[:, int(cls)] = raw[:, j]
    return out


def _scored(y4, p4, threshold):
    y = (np.asarray(y4, int) != 0).astype(int)
    score = 1.0 - np.asarray(p4, float)[:, 0]
    pred = (score >= threshold).astype(int)
    cm = MET.confusion_binary(y, pred)
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "TP": int(cm["TP"]), "FN": int(cm["FN"]), "FP": int(cm["FP"]), "TN": int(cm["TN"]),
        "FNR": cm["FNR"], "FPR": cm["FPR"],
        "auc": MET.auc_score(y, score),
        "logloss": MET.multiclass_logloss(np.asarray(y4, int), p4),
        "brier": MET.multiclass_brier(np.asarray(y4, int), p4),
    }


def _row_logloss(y4, p4):
    p = np.clip(np.asarray(p4, float), 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y4)), np.asarray(y4, int)])


def _block_bootstrap(y4, p4_a, p4_b, countries, *, n_boot=500, seed=363):
    """Country-block bootstrap CI for mean(logloss_b) - mean(logloss_a)."""
    y4 = np.asarray(y4, int)
    la = _row_logloss(y4, p4_a)
    lb = _row_logloss(y4, p4_b)
    diff = lb - la
    blocks = {}
    for i, c in enumerate(countries):
        blocks.setdefault(str(c), []).append(diff[i])
    keys = sorted(blocks)
    if len(keys) < 2:
        return {"n_country_blocks": len(keys), "estimate": float(diff.mean()),
                "ci95": [None, None], "status": "INSUFFICIENT_COUNTRY_BLOCKS"}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals = np.concatenate([blocks[keys[j]] for j in pick])
        draws.append(float(vals.mean()))
    return {"n_country_blocks": len(keys), "n_boot": int(n_boot),
            "estimate": float(diff.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "status": "PASS_BLOCK_BOOTSTRAP"}


def _merge_packet(frame, packet, cols):
    if packet is None:
        raise ValueError("PACKET_FRAME_REQUIRED_WITH_PACKET_JOINT_COLS")
    if all(c in frame.columns for c in cols):
        return frame
    keep = ["country", "month"] + [c for c in cols if c in packet.columns]
    merged = frame.merge(packet[keep], on=["country", "month"], how="left",
                         validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("PACKET_FEATURE_KEYS_NOT_MATCHED:" +
                         str(int((merged["_merge"] != "both").sum())))
    return merged.drop(columns=["_merge"])


def _fit_estimator(Xd, y4_dev, Xo, kind):
    fit = _fit_primary(Xd, y4_dev) if kind == "MNLOGIT" else _fit_challenger(Xd, y4_dev)
    p4d = _p4(fit, Xd)
    p4o = _p4(fit, Xo) if len(Xo) else np.zeros((0, 4))
    return fit, p4d, p4o


def run(panel_dev, panel_oos, spec, official, *, packet_dev=None, packet_oos=None,
        joint_cols=None, out_dir=None) -> dict:
    """Fit independent M00/M01/M10/M11 on identical DEV/OOS keys.

    official: {'model': 'M11', 'family': 'LOGISTIC', 'cols': [...],
               'threshold': float, 'p4_oos': [...full OOS rows...]}
    """
    base = PNL.corrected_blocks(spec)["M00"]
    ind = PNL.corrected_blocks(spec)["M01"][len(base):]
    dev, oos = panel_dev.reset_index(drop=True), panel_oos.reset_index(drop=True)
    joint_cols = list(joint_cols or [])
    if joint_cols:
        dev = _merge_packet(dev, packet_dev, joint_cols)
        oos = _merge_packet(oos, packet_oos, joint_cols)
        joint_source = ("MOTIF_PACKET_SELECTED_CONFIG_FEATURES"
                        if any(c.startswith("pkt_selected_") for c in joint_cols)
                        else "MOTIF_PACKET_REGISTERED_RESEARCH_FEATURES")
    else:
        joint_cols = list(spec.joint_columns)
        joint_source = "PROXY_JOINT_COLUMNS_DIAGNOSTIC_ONLY"
    if dev.duplicated(["country", "month"]).any() or oos.duplicated(["country", "month"]).any():
        raise ValueError("DUPLICATE_COUNTRY_MONTH_KEYS")
    blocks = {"M00": base, "M01": base + ind, "M10": base + joint_cols,
              "M11": base + ind + joint_cols}
    y4_dev = PNL.y4_from(dev)
    y_dev = (y4_dev != 0).astype(int)
    scorable = (oos["label_status"].astype(str) == FORCED_SCORABLE).to_numpy()
    y4_oos_all = PNL.y4_from(oos)
    y4_oos = y4_oos_all[scorable]
    countries_oos = oos["country"].to_numpy()[scorable]

    rows, p4_oos_by_model, identities, fit_meta = [], {}, {}, {}
    model_entries = []
    primary_ok, challenger_ok = True, True
    for name, cols in blocks.items():
        Xd = _matrix(dev, cols)
        Xo = _matrix(oos, cols)[scorable]
        entry = {"model": name, "feature_source": joint_source,
                 "columns": list(cols), "n_features": len(cols),
                 "n_dev": int(len(Xd)), "n_oos_scorable": int(len(Xo))}
        for kind in ("MNLOGIT", "SKLEARN"):
            try:
                fit, p4d, p4o = _fit_estimator(Xd, y4_dev, Xo, kind)
                tr = THR.exact_threshold_search(1 - p4d[:, 0], y_dev)
                dm = _scored(y4_dev, p4d, tr.threshold)
                om = _scored(y4_oos, p4o, tr.threshold) if len(y4_oos) else None
                rows.append({"model": name, "estimator": kind,
                             "feature_source": joint_source,
                             "threshold": float(tr.threshold),
                             "gate_feasible": bool(tr.gate_feasible),
                             "ind_dev": dm, "ind_oos_scorable": om})
                fit_meta[f"{kind}_{name}"] = {
                    "solver_used": fit["solver_used"],
                    "n_features": int(len(cols)),
                    "n_fit": int(len(Xd)),
                    "fit_status": "PASS",
                }
                if kind == "MNLOGIT":
                    p4_oos_by_model[name] = p4o
                    if len(y4_oos) and om is not None:
                        entry["oos_metrics"] = om
                    entry["dev_metrics"] = dm
                else:
                    entry["challenger_oos_metrics"] = om
                    entry["challenger_threshold"] = float(tr.threshold)
            except Exception as exc:
                fit_meta[f"{kind}_{name}"] = {"fit_status": "FAIL",
                                              "reason": f"{type(exc).__name__}:{exc}"}
                if kind == "MNLOGIT":
                    primary_ok = False
                else:
                    challenger_ok = False
                rows.append({"model": name, "estimator": kind,
                             "feature_source": joint_source, "status": "FIT_FAILED",
                             "reason": f"{type(exc).__name__}:{exc}"})
        identities[name] = {
            "columns": list(cols), "n_features": len(cols),
            "dev_matrix_sha": sha256_json(np.round(Xd, 12).tolist()),
            "oos_matrix_sha": sha256_json(np.round(Xo, 12).tolist()),
            "dev_rows": int(len(Xd)), "oos_scorable_rows": int(len(Xo)),
            "n_classes_fit": int(len(np.unique(y4_dev))),
        }
        entry["oos_scorable"] = entry.get("oos_metrics")
        model_entries.append(entry)

    payload = {
        "status": ("PASS_INDEPENDENT_MATCHED_ABLATION"
                   if primary_ok and challenger_ok and joint_cols and scorable.any()
                   and joint_source.startswith("MOTIF_PACKET")
                   else "PARTIAL_INDEPENDENT_ABLATION"),
        "claim": "INDEPENDENT_FEATURE_AND_ESTIMATOR_ABLATION_NOT_NUMERICAL_PARITY",
        "primary_estimator": PRIMARY,
        "challenger_estimator": CHALLENGER,
        "estimator_independence": {
            "primary_is_project_estimator": False,
            "project_estimator": "rev363.estimators.FittedModel (sklearn LogisticRegression)",
            "primary_implementation": "statsmodels MNLogit (different library and solver)",
            "challenger_implementation": "sklearn LogisticRegression with different C/class_weight",
            "shared_coefficients": False,
            "shared_fit_object": False,
        },
        "official_model": official.get("model", "M11"),
        "official_columns": list(official.get("cols", [])),
        "official_threshold": float(official.get("threshold", float("nan"))),
        "joint_feature_source": joint_source,
        "packet_joint_cols": list(joint_cols),
        "base_identity": {
            "base_columns": list(base),
            "all_four_share_same_base": True,
            "ind_columns": list(ind),
            "joint_columns": list(joint_cols),
        },
        "same_keys": True,
        "n_dev": int(len(dev)), "n_oos": int(len(oos)),
        "n_oos_scorable": int(scorable.sum()),
        "models": model_entries,
        "fit_rows": rows,
        "model_identities": identities,
        "fit_meta": fit_meta,
        "numerical_parity_separate": True,
        "dependence_caveat": "COUNTRY_MONTHS_OF_ONE_EPISODE_ARE_NOT_INDEPENDENT",
    }

    # official vs independently fitted M11 on the SAME scorable OOS rows
    official_name = official.get("model", "M11")
    if official.get("p4_oos") is not None and official_name in p4_oos_by_model:
        off_full = np.asarray(official["p4_oos"], float)
        if len(off_full) == len(oos):
            off = off_full[scorable]
            ind = p4_oos_by_model[official_name]
            off_thr = float(official.get("threshold", 0.5))
            off_m = _scored(y4_oos, off, off_thr)
            ind_m = _scored(y4_oos, ind, float(next(
                r["threshold"] for r in rows
                if r["model"] == official_name and r["estimator"] == "MNLOGIT")))
            ch_m = next((e.get("challenger_oos_metrics") for e in model_entries
                         if e.get("model") == official_name), None)
            payload["official_vs_independent"] = {
                "n": int(scorable.sum()),
                "official_metrics": off_m,
                "independent_metrics": ind_m,
                "independent_challenger_metrics": ch_m,
                "delta_logloss_independent_minus_official": ind_m["logloss"] - off_m["logloss"],
                "delta_brier_independent_minus_official": ind_m["brier"] - off_m["brier"],
                "score_correlation": (float(np.corrcoef(off[:, 0], ind[:, 0])[0, 1])
                                      if len(off) > 1 else None),
                "note": ("DIFFERENT FEATURES/ESTIMATOR; AGREEMENT IS NOT PARITY; "
                         "the unregularised MNLogit is an overconfidence stress case, "
                         "the regularised challenger is reported separately"),
            }
            rows_out = [{"estimator": "OFFICIAL_RETAINED", **off_m},
                        {"estimator": "INDEPENDENT_MNLOGIT", **ind_m}]
            if ch_m is not None:
                rows_out.append({"estimator": "INDEPENDENT_SKLEARN_CHALLENGER", **ch_m})
            if out_dir is not None:
                pd.DataFrame(rows_out).to_csv(
                    Path(out_dir) / "RQ01_11_OFFICIAL_VS_INDEPENDENT.csv", index=False)
    if "MNLOGIT_M11" in fit_meta and "MNLOGIT_M01" in fit_meta:
        # matched contrasts on the same scorable OOS rows (primary estimator)
        def _p(name):
            return p4_oos_by_model[name]
        if all(n in p4_oos_by_model for n in ("M00", "M01", "M10", "M11")):
            contrasts = {}
            for a, b, label in (("M01", "M11", "joint_given_individual"),
                                ("M10", "M11", "individual_given_joint"),
                                ("M00", "M10", "joint_alone"),
                                ("M00", "M01", "individual_alone")):
                contrasts[label] = _block_bootstrap(
                    y4_oos, _p(a), _p(b), countries_oos)
            payload["matched_loss_contrasts"] = {
                "definition": "mean_logloss(second) - mean_logloss(first)",
                "contrasts": contrasts,
            }
            pd.DataFrame([
                {"contrast": k, "estimate": v["estimate"],
                 "ci95_low": v["ci95"][0], "ci95_high": v["ci95"][1],
                 "n_country_blocks": v["n_country_blocks"], "status": v["status"]}
                for k, v in contrasts.items()
            ]).to_csv(Path(out_dir) / "RQ01_11_BLOCK_BOOTSTRAP.csv",
                      index=False) if out_dir is not None else None

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        flat = []
        for r in rows:
            for split, key in (("dev", "ind_dev"), ("oos_scorable", "ind_oos_scorable")):
                m = r.get(key)
                if m:
                    flat.append({"model": r["model"], "estimator": r["estimator"],
                                 "split": split, "feature_source": r["feature_source"],
                                 **m})
        pd.DataFrame(flat).to_csv(out / "RQ01_11_INDEPENDENT_ABLATION.csv", index=False)
        (out / "RQ01_11_NUMERICAL_VS_INDEPENDENT_PARITY.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
