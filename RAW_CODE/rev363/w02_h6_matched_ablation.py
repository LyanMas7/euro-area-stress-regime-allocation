"""W02-006: fixed H6 full-DEV RETROSPECTIVE matched four-model ablation.

Standalone isolated W02 module. Depends only on prior executed 005 native research
CSV and 001 DEV-only onset labels, both derived from ASSET03_LOCKED. It NEVER
consumes/creates OOS targets or splits DEV, and never promotes this as issued PIT.
The 22 Joint rows are Gate2 CANDIDATES (Gate1 not owner certified).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEV_START, DEV_END = "2005-01", "2019-12"
RUN_ID = "W02-MATCHED-H6-DEV-RETROSPECTIVE-20261024-006"
C = 0.20
SEED = 406
ETA = 0.5
COUNTRIES = ("BE", "DE", "ES", "FI", "FR", "IT", "NL")
LONG = ("CG", "DSR-P")
SHORT = ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")
PAIRS = {"LONG": tuple([(LONG[0], LONG[1])]),
         "SHORT": tuple((SHORT[a], SHORT[b]) for a in range(7) for b in range(a+1, 7))}
SRC_ORDER = tuple((branch, source, sign)
                  for branch, arr in (("LONG", LONG), ("SHORT", SHORT))
                  for source in arr for sign in (-1, 1))
PAIR_ORDER = tuple((branch, a, b, sign)
                   for branch, arr in (("LONG", PAIRS["LONG"]), ("SHORT", PAIRS["SHORT"]))
                   for a, b in arr for sign in (-1, 1))


def _sha_bytes(blob: bytes) -> str:
    return sha256(blob).hexdigest()


def csv_member(zpath, member):
    with zipfile.ZipFile(zpath) as zz:
        blob = zz.read(member)
    return pd.read_csv(BytesIO(blob), low_memory=False), _sha_bytes(blob)


def fetch_inputs(evidence_005_zip, handoff_001_zip):
    prefix = "EVIDENCE/"
    names = {"ind": "W02_005_INDIVIDUAL_DEV.csv",
             "joint": "W02_005_JOINT_DEV.csv",
             "four": "W02_005_IL_IS_JL_JS_DEV.csv"}
    with zipfile.ZipFile(evidence_005_zip) as zz:
        register = json.loads(zz.read(prefix+"W02_005_EVIDENCE.json"))
    if register["source_zip_sha256"] != "74b7d5b25108a18bf9514a57f3ed3a05ff26eb9a1d0a019af24f33ebd11c2d0e":
        raise ValueError("UNKNOWN_SOURCE_ASSET03")
    csvs, hashes = {}, {}
    for k, filename in names.items():
        csvs[k], hashes[k] = csv_member(evidence_005_zip, prefix+filename)
        if hashes[k] != register["artifacts"][{"ind": "individual", "joint": "joint", "four": "four"}[k]]["sha256"]:
            raise ValueError("PREVIOUS_EVIDENCE_HASH_MISMATCH:"+k)
    targets, hashes["targets"] = csv_member(handoff_001_zip, "W02_HANDOFF/RQ02_DEV_ONSET_TARGET_KEYS.csv")
    return csvs["ind"], csvs["joint"], csvs["four"], targets, hashes


def select_h6_keys(targets):
    req = {"country", "issue", "horizon", "horizon_months", "target_end",
           "research_label_mature_at", "first_onset"}
    if not req.issubset(targets):
        raise ValueError("TARGET_SCHEMA_MISSING")
    t = targets.loc[(targets.horizon == "H6") & (targets.issue.between(DEV_START, DEV_END))].copy()
    if t.empty or t.duplicated(["country", "issue"]).any():
        raise ValueError("EMPTY_OR_DUPLICATE_H6_KEYS")
    if not t.country.isin(COUNTRIES).all() or not t.first_onset.isin([0, 1]).all():
        raise ValueError("UNKNOWN_COUNTRY_OR_BINARY_LABEL")
    if not ((t.target_end <= DEV_END) & (t.research_label_mature_at <= DEV_END)
            & (t.horizon_months == 18)).all():
        raise ValueError("DEV_MATURITY_LEAKAGE")
    t = t.sort_values(["country", "issue"]).reset_index(drop=True)
    return t


def matched_design(ind, joint, targets):
    """Build true same-key matrices and report explicit U; no label in features.

    J_perp is linear residual of all 22 candidate Gate2 votes/U flags after
    projection on [country, Individual], not an owner-approved PureJoint.
    """
    t = select_h6_keys(targets)
    keys = list(zip(t.country, t.issue))
    keyset = set(keys)
    i = ind.loc[ind.apply(lambda r: (r["country"], r["issue"]) in keyset, axis=1)].copy()
    j = joint.loc[joint.apply(lambda r: (r["country"], r["issue"]) in keyset, axis=1)].copy()
    if i.duplicated(["country", "issue", "branch", "source", "sign"]).any():
        raise ValueError("DUPLICATE_INDIVIDUAL_RECORD")
    if j.duplicated(["country", "issue", "branch", "pair_id"]).any():
        raise ValueError("DUPLICATE_JOINT_RECORD")
    n = len(t)
    if len(i) != n*len(SRC_ORDER) or len(j) != n*(len(PAIRS["LONG"])+len(PAIRS["SHORT"])):
        raise ValueError("MISSING_MATCHED_NATIVE_RECORDS")
    idx_i = {(r.country, r.issue, r.branch, r.source, int(r.sign)): r.candidate_vote
             for r in i.itertuples(index=False)}
    idx_j = {(r.country, r.issue, r.branch, r.pair_id): r
             for r in j.itertuples(index=False)}
    base = np.zeros((n, len(COUNTRIES)), dtype=float)
    # GLOBAL NO-U: one binary 1[PASS] feature per signed source/pair; the REV373 `vote_is_U` dummy columns
    # (unknown-vote regressors) are retired.  Non-evaluated votes are counted as coverage, never a feature.
    individual = np.zeros((n, len(SRC_ORDER)), dtype=float)
    joint_raw = np.zeros((n, len(PAIR_ORDER)), dtype=float)
    plus_joint = 0
    pure_semantic = 0
    joint_ne = 0
    ind_ne = 0
    for row, (country, issue) in enumerate(keys):
        base[row, COUNTRIES.index(country)] = 1
        for col, (branch, src, sign) in enumerate(SRC_ORDER):
            vote = idx_i[(country, issue, branch, src, sign)]
            if not (vote in ("PASS", "FAIL") or _not_evaluated(vote)):
                raise ValueError("INVALID_I_VOTE")
            individual[row, col] = float(vote == "PASS")
            ind_ne += not (vote in ("PASS", "FAIL"))
        for col, (branch, a, b, sign) in enumerate(PAIR_ORDER):
            r = idx_j[(country, issue, branch, a+"|"+b)]
            vote = r.vote_plus if sign == 1 else r.vote_minus
            if not (vote in ("PASS", "FAIL") or _not_evaluated(vote)):
                raise ValueError("INVALID_J_VOTE")
            joint_raw[row, col] = float(vote == "PASS")
            joint_ne += not (vote in ("PASS", "FAIL"))
            if vote == "PASS":
                plus_joint += 1
                ia = idx_i[(country, issue, branch, a, sign)]
                ib = idx_i[(country, issue, branch, b, sign)]
                if ia != "PASS" or ib != "PASS":
                    pure_semantic += 1
    # Full-DEV linear feature-only nuisance fit, no y; valid retrospective
    # projection, NOT an issue-time projection for live forecasts.
    z = np.column_stack((base, individual))
    beta, _, _, _ = np.linalg.lstsq(z, joint_raw, rcond=1e-10)
    j_perp = joint_raw - z @ beta
    orth = float(np.max(np.abs(z.T @ j_perp)))
    if orth > 1e-6:
        raise AssertionError("JOINT_RESIDUAL_NON_ORTHOGONAL")
    jrank = int(np.linalg.matrix_rank(j_perp, tol=1e-8))
    cols_b = ["country_"+c for c in COUNTRIES]
    cols_i = [f"I_{b}_{s}_{sgn}_{field}" for b, s, sgn in SRC_ORDER
              for field in ("pass",)]
    cols_j = [f"J_perp_{b}_{a}_{bb}_{sgn}_{field}" for b, a, bb, sgn in PAIR_ORDER
              for field in ("pass",)]
    frames = {"M00": base, "M01": z, "M10": np.column_stack((base, j_perp)),
              "M11": np.column_stack((z, j_perp))}
    columns = {"M00": cols_b, "M01": cols_b+cols_i,
               "M10": cols_b+cols_j, "M11": cols_b+cols_i+cols_j}
    diagnostic = {"n_matched_keys": n, "n_countries": len(COUNTRIES),
                  "n_positive": int(t.first_onset.sum()),
                  "n_negative": int(n-t.first_onset.sum()),
                  "n_individual_votes": int(individual.shape[1]),
                  "n_candidate_pairs": 22,
                  "n_joint_signed_votes": int(joint_raw.shape[1]),
                  "n_joint_pass": int(plus_joint), "n_joint_pass_without_both_I_pass": int(pure_semantic),
                  "n_I_not_evaluated": int(ind_ne), "n_J_not_evaluated": int(joint_ne),
                  "j_perp_rank": jrank, "j_perp_absmax": float(np.abs(j_perp).max()),
                  "z_t_j_perp_absmax": orth,
                  "retrospective_proxy": True,
                  "gate1_approved": False}
    return t, frames, columns, diagnostic, (base, individual, joint_raw, j_perp)


def _not_evaluated(vote) -> bool:
    """An empty vote cell (None/NaN) is a non-executed record; any other non-binary token is invalid."""
    return vote is None or (isinstance(vote, float) and vote != vote)


def fit_fixed_logistic(X, y, names, model_name, *, return_artifact=False):
    """One fixed in-sample exploratory logistic fit; not future performance."""
    std = np.std(X, axis=0)
    std = np.where(std > 1e-10, std, 1.0)
    xs = X/std
    model = LogisticRegression(C=C, random_state=SEED, solver="lbfgs",
                               max_iter=1500, tol=1e-7, fit_intercept=False)
    model.fit(xs, y)
    p = model.predict_proba(xs)[:, list(model.classes_).index(1)]
    p = np.clip(p, 1e-12, 1-1e-12)
    pred = (p >= ETA).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    llrow = -(y*np.log(p) + (1-y)*np.log(1-p))
    artifact = {"name": model_name, "features": names, "C": C,
                "scale": std.tolist(), "coefficients": model.coef_[0].tolist(),
                "rows": len(y), "seed": SEED, "eta": ETA,
                "estimator": "sklearn.LogisticRegression", "solver": "lbfgs",
                "fit_intercept": False, "class_order": [0, 1],
                "claim": "DEV_RETROSPECTIVE_NOT_ISSUED"}
    payload = json.dumps({k: artifact[k] for k in ("name", "features", "C", "scale",
                     "coefficients", "rows", "seed")}, sort_keys=True).encode()
    metrics = {"model": model_name, "n": int(len(y)), "P": int(y.sum()),
               "N": int(len(y)-y.sum()), "n_features": len(names),
               "logloss_insample": float(llrow.mean()),
               "brier_insample": float(np.mean((p-y)**2)),
               "TP": tp, "FN": fn, "FP": fp, "TN": tn,
               "FNR_insample": fn/(tp+fn) if (tp+fn) else None,
               "FPR_insample": fp/(fp+tn) if (fp+tn) else None,
               "prob_min": float(p.min()), "prob_max": float(p.max()),
               "model_sha256": _sha_bytes(payload), "converged": bool(model.n_iter_[0] < 1500),
               "fit_class": "EXECUTED_DEV_RETROSPECTIVE_NOT_FORECAST_VALIDATION"}
    artifact["model_sha256"] = metrics["model_sha256"]
    if return_artifact:
        reconstructed = np.exp(-np.logaddexp(0.0, -(xs @ model.coef_[0])))
        if not np.allclose(reconstructed, p, atol=1e-10, rtol=0):
            raise AssertionError("SERIALIZED_LOGISTIC_REPLAY_MISMATCH")
        return metrics, p, llrow, artifact
    return metrics, p, llrow


def run(evidence_zip, handoff_zip, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ind, joint, four, targets, input_hashes = fetch_inputs(evidence_zip, handoff_zip)
    t, models, cols, geometry, pieces = matched_design(ind, joint, targets)
    y = t.first_onset.to_numpy(dtype=int)
    base_raw, ind_raw, joint_raw, joint_perp = pieces
    z_raw = np.column_stack((base_raw, ind_raw))
    proj_beta, _, _, _ = np.linalg.lstsq(z_raw, joint_raw, rcond=1e-10)
    if not np.allclose(joint_raw-z_raw@proj_beta, joint_perp, atol=1e-10, rtol=0):
        raise AssertionError("FROZEN_PROJECTION_REPLAY_MISMATCH")
    projection = {"run_id": RUN_ID,
                  "projection_kind": "FULL_DEV_RETROSPECTIVE_FIT_ON_X_ONLY",
                  "rcond": 1e-10, "z_features": cols["M01"],
                  "j_raw_features": [f"J_raw_{b}_{a}_{bb}_{sgn}_{field}"
                                     for b,a,bb,sgn in PAIR_ORDER for field in ("pass",)],
                  "beta_rows": int(proj_beta.shape[0]),
                  "beta_cols": int(proj_beta.shape[1]),
                  "beta": proj_beta.tolist(),
                  "issuer_time_valid": False}
    (out/"W02_006_H6_JOINT_PROJECTION.json").write_text(json.dumps(projection, indent=2))
    metrics = []
    predictions = []
    individual_loss = {}
    artifacts = {}
    for name in ("M00", "M01", "M10", "M11"):
        stats, p, ll, artifact = fit_fixed_logistic(models[name], y, cols[name], name, return_artifact=True)
        artifacts[name] = artifact
        metrics.append(stats)
        individual_loss[name] = ll
        predictions.append(pd.DataFrame({"country": t.country, "issue": t.issue, "horizon": "H6",
                                         "target_end": t.target_end, "model": name,
                                         "y_onset": y, "p_onset_insample": p,
                                         "pred_eta_0p5": (p >= ETA).astype(int),
                                         "row_logloss": ll,
                                         "claim": "RETROSPECTIVE_IN_SAMPLE_ONLY_NOT_ISSUED"}))
    met = pd.DataFrame(metrics)
    pred = pd.concat(predictions, ignore_index=True)
    country = (pred.groupby(["model", "country"], sort=True)
               .agg(n=("y_onset", "size"), positive=("y_onset", "sum"),
                    mean_logloss=("row_logloss", "mean")).reset_index())
    differences = []
    for a, b in (("M01", "M00"), ("M10", "M00"), ("M11", "M01"), ("M11", "M10")):
        d = individual_loss[a]-individual_loss[b]
        differences.append({"contrast": a+"-"+b, "n_paired": len(d),
                            "mean_delta_logloss": float(np.mean(d)),
                            "fraction_rows_delta_negative": float(np.mean(d < 0)),
                            "interpretation": "DEV_IN_SAMPLE_OPTIMISTIC"})
    contrasts = pd.DataFrame(differences)
    results = {"run_id": RUN_ID, "evidence_input_sha256": input_hashes,
               "population": "EA7_H6_MATURED_FIRST_ONSET_DEV_ONLY", "source_clock":
               "DATASET_DIRECT_OBSERVATION_PROXY_NOT_ISSUER_TIME",
               "horizon_target": "FIRST_ONSET_WITHIN_18_MONTHS_NOT_FUTURE_STATE",
               "no_oos_access": True, "dev_time_folds": 0,
               "xgboost": "DORMANT_GATE_A_NOT_FITTED", "expanding": "BLOCKED_BY_CONTRACT",
               "threshold_search": "NOT_RUN_FIXED_ETA_0.5", "owner_gate": "NOT_CERTIFIED",
               "joint_gate1": "UNVERIFIED_ALL_22_PAIRS_ARE_GATE2_CANDIDATES",
               "purity": "DEV_FULL_SAMPLE_LINEAR_RESIDUALIZATION_NOT_CAUSAL",
               "geometry": geometry,
               "model_sha256": {x["model"]: x["model_sha256"] for x in metrics}}
    (out/"W02_006_H6_FITTED_PARAMETERS.json").write_text(json.dumps(artifacts, indent=2))
    for name, obj in (("W02_006_H6_MATCHED_PREDICTIONS.csv", pred),
                      ("W02_006_H6_MODEL_LOSS.csv", met),
                      ("W02_006_H6_COUNTRY_SUPPORT.csv", country),
                      ("W02_006_H6_PAIRED_CONTRASTS.csv", contrasts)):
        obj.to_csv(out/name, index=False)
    for name in ("W02_006_H6_MATCHED_PREDICTIONS.csv", "W02_006_H6_MODEL_LOSS.csv",
                 "W02_006_H6_COUNTRY_SUPPORT.csv", "W02_006_H6_PAIRED_CONTRASTS.csv"):
        results[name] = {"sha256": _sha_bytes((out/name).read_bytes()),
                         "rows": int(len(pd.read_csv(out/name)))}
    results["W02_006_H6_FITTED_PARAMETERS.json"] = {"sha256": _sha_bytes((out/"W02_006_H6_FITTED_PARAMETERS.json").read_bytes()), "models": 4}
    results["W02_006_H6_JOINT_PROJECTION.json"] = {"sha256": _sha_bytes((out/"W02_006_H6_JOINT_PROJECTION.json").read_bytes()), "shape": list(proj_beta.shape)}
    (out/"W02_006_EVIDENCE.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--evidence-005-zip", required=True)
    cli.add_argument("--handoff-001-zip", required=True)
    cli.add_argument("--out", required=True)
    args = cli.parse_args()
    result = run(args.evidence_005_zip, args.handoff_001_zip, args.out)
    print(json.dumps({"run_id": result["run_id"], "geometry": result["geometry"]}, indent=2))


if __name__ == "__main__":
    main()
