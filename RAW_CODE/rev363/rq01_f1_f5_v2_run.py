"""RQ01-09 FIVE_FITTING_RESEARCH_GATES_V2 runner (additive, DEV-only).

Rebuilds the eight retained RQ01 four-M models (LOGISTIC/XGBOOST × M00/M01/M10/M11)
on the exact DEV design of run `REV_NEXT_RQ01_20261025T080433Z`, verifies the frozen
DEV confusion matrices reproduce, then computes the five typed fitting gates:

    F1/F2  same-key class-prior null (M00) / same-family M00 (M01/M10/M11)
    F3     LOGISTIC penalized-solver certificate; XGB NOT_APPLICABLE
    F4     B=19 real within-country label-placebo refits, p=(1+k)/(B+1)
    F5     LOGISTIC Fisher EDF in the contrast subspace; XGB NOT_APPLICABLE

Writes three ledgers (§2.5 of the dossier) + the run manifest.  No OOS is read.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_adapters import fitting_gates_v2 as G
from rev363 import estimators as EST
from rev363 import panel as PNL

DEV_LAST = "2019-12"
DEV_FIT_START = "2005-07"
B_PLACEBOS = 19
RUN_VERSION = "RQ01_F1F5_V2_RUN_V1"
R10_REFERENCE = {
    ("LOGISTIC", "M00"): (0.5558170896677577, 0.38801908365792603),
    ("LOGISTIC", "M01"): (0.5291399652, 0.3680622724),
    ("LOGISTIC", "M10"): (0.5056323030, 0.3497146203),
    ("LOGISTIC", "M11"): (0.4951959006, 0.3410068108),
    ("XGBOOST", "M00"): (0.5607153777269425, 0.3877846140369293),
    ("XGBOOST", "M01"): (0.4651269404, 0.3081211337),
    ("XGBOOST", "M10"): (0.4165915806, 0.2702650435),
    ("XGBOOST", "M11"): (0.3868227780, 0.2441781989),
}


def _sha_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def _digest_model(mdl) -> dict:
    sig = mdl.signature
    return {"digest": mdl.digest, "family": mdl.family, "name": mdl.name,
            "n_class": int(mdl.n_class), "n_features": len(mdl.columns),
            "coef_sha256": (None if mdl.coef is None else _sha_bytes(np.round(mdl.coef, 10).tobytes())),
            "booster_sha256": (None if mdl.booster_b64 is None else
                               _sha_bytes(mdl.booster_b64.encode())),
            "columns_sha256": _sha_bytes("|".join(mdl.columns).encode())}


def build_design(dataset_root, rq01_dir):
    prep = PNL.build_panel(dataset_root, row_start="2004-01", row_end=DEV_LAST, data_end=DEV_LAST)
    panel = prep["panel"]
    spec = prep["spec"]
    dev = panel[(panel["month"] >= DEV_FIT_START) & (panel["month"] <= DEV_LAST)].reset_index(drop=True)
    rq01_dir = Path(rq01_dir)
    pf = pd.read_csv(rq01_dir / "RQ01_03_PACKET_FEATURES_DEV.csv")
    joint_cols = [c for c in pf.columns if c.startswith("pkt_selected_") and not c.endswith("_pass_count")]
    keep = ["country", "month"] + joint_cols + ["pkt_joint_not_evaluated_count", "pkt_pool_issued"]
    dev = dev.merge(pf[keep], on=["country", "month"], how="left", validate="one_to_one")
    it = pd.read_csv(rq01_dir / "RQ01_02_INDIVIDUAL_NATIVE_TAPE.csv")
    ev = it[it["status"] == "EVALUATED"][["country", "issue", "source", "vote"]].copy()
    piv = ev.pivot_table(index=["country", "issue"], columns="source", values="vote", aggfunc="first")
    piv.columns = [f"pkt_ind_{c}" for c in piv.columns]
    ind_cols = sorted(piv.columns)
    feat = piv.reset_index().rename(columns={"issue": "month"})
    dev = dev.merge(feat, on=["country", "month"], how="left", validate="one_to_one")
    base = list(PNL.corrected_blocks(spec)["M00"])
    blocks = {"M00": base, "M01": base + ind_cols, "M10": base + joint_cols,
              "M11": base + ind_cols + joint_cols}
    return dev, spec, blocks, joint_cols, ind_cols


def main(dataset_root, rq01_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rq01_dir = Path(rq01_dir)
    dev, spec, blocks, joint_cols, ind_cols = build_design(dataset_root, rq01_dir)
    y4 = PNL.y4_from(dev)
    y = (y4 != 0).astype(int)
    keys = list(zip(dev["country"], dev["month"]))
    groups = dev["country"].to_numpy()
    frozen = pd.read_csv(rq01_dir / "RQ01_05_FOUR_M_DEV.csv").set_index(["family", "model"])

    models, scores, fits = {}, {}, {}
    repro = []
    for fam in ("LOGISTIC", "XGBOOST"):
        for mname in ("M00", "M01", "M10", "M11"):
            cols = blocks[mname]
            X = PNL.matrix_for(dev, cols)
            mdl = EST.fit_two_family(X, cols, y4, family=fam, name=mname)
            P = mdl.p4(X)
            score = 1.0 - P[:, 0]
            thr = float(frozen.loc[(fam, mname), "threshold"])
            pred = (score >= thr).astype(int)
            row = frozen.loc[(fam, mname)]
            tp = int(((pred == 1) & (y == 1)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            ok = (tp, fn, fp, tn) == (int(row["TP"]), int(row["FN"]), int(row["FP"]), int(row["TN"]))
            repro.append({"family": fam, "model": mname, "status": "PASS" if ok else "FAIL",
                          "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                          "frozen_tp": int(row["TP"]), "frozen_fn": int(row["FN"]),
                          "frozen_fp": int(row["FP"]), "frozen_tn": int(row["TN"]),
                          "threshold": thr})
            if not ok:
                raise AssertionError(f"DEV_REPRODUCTION_FAILED:{fam}:{mname}")
            models[(fam, mname)] = mdl
            scores[(fam, mname)] = P
            fits[(fam, mname)] = {"X": X, "cols": cols,
                                  "mdl": mdl, "digest": _digest_model(mdl)}
            print(f"[fit] {fam} {mname}: n={len(y)} logloss={G.logloss(y4, P):.6f}", flush=True)

    import sklearn.linear_model as SK
    rows, f4_detail = [], {}
    for fam in ("LOGISTIC", "XGBOOST"):
        m00P = scores[(fam, "M00")]
        for mname in ("M00", "M01", "M10", "M11"):
            mdl = models[(fam, mname)]
            P = scores[(fam, mname)]
            X = fits[(fam, mname)]["X"]
            cols = fits[(fam, mname)]["cols"]
            ref_kind = "CLASS_PRIOR_NULL" if mname == "M00" else "SAME_FAMILY_M00"
            ref = G.build_matched_reference(keys, y4, m00P if ref_kind == "SAME_FAMILY_M00" else None,
                                            list(range(4)), "country-month", ref_kind)
            f12 = G.score_f1_f2(P, ref["reference_probs"], y4)
            base_row = {"task_id": "RQ01-09", "family": fam, "model": mname,
                        "n_matched": f12["n_matched"],
                        "reference_kind": ref_kind,
                        "model_digest": fits[(fam, mname)]["digest"]["digest"],
                        "columns_sha256": fits[(fam, mname)]["digest"]["columns_sha256"],
                        "score_sha256": f12["candidate_sha256"]}
            rows.append({**base_row, "gate": "F1", "metric": "MULTICLASS_LOGLOSS_NATS_PER_KEY",
                         "observed_value": f12["F1_candidate"], "reference_value": f12["F1_reference"],
                         "comparator": "<", "margin": f12["F1_margin"],
                         "pass": f12["F1_pass"], "status": "PASS" if f12["F1_pass"] else "FAIL",
                         "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                         "notes": "same-key reference; M00 uses class-prior null"})
            rows.append({**base_row, "gate": "F2", "metric": "BRIER_SUM_K_MEAN",
                         "observed_value": f12["F2_candidate"], "reference_value": f12["F2_reference"],
                         "comparator": "<", "margin": f12["F2_margin"],
                         "pass": f12["F2_pass"], "status": "PASS" if f12["F2_pass"] else "FAIL",
                         "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                         "notes": "four-class summed Brier per key; never compare to binary Brier"})
            # ---- F3
            if fam == "LOGISTIC":
                xs = mdl.matrix(X)
                # tight refit (tol=1e-8) of the identical estimator = the mathematical optimum;
                # the retained max_iter/1e-4 fit is compared against it and must agree at the
                # decision level (documented delta), otherwise the certificate is not representative.
                direct = SK.LogisticRegression(C=float(EST.LOGISTIC_PARAMS["C"]),
                                               solver=EST.LOGISTIC_PARAMS["solver"],
                                               max_iter=EST.LOGISTIC_PARAMS["max_iter"],
                                               random_state=EST.LOGISTIC_PARAMS["random_state"],
                                               tol=1e-8)
                direct.fit(xs, y4)
                coef_delta = float(np.max(np.abs(np.asarray(direct.coef_, float) - np.asarray(mdl.coef, float))))
                p_tight = direct.predict_proba(xs)
                p_retained = mdl.p4(PNL.matrix_for(dev, cols))
                p_delta = float(np.max(np.abs(p_tight - p_retained)))
                C = float(EST.LOGISTIC_PARAMS["C"])
                J_ref = C * len(y4) * np.log(4.0)
                f3 = G.certify_f3_optimization(
                    {"family": "LOGISTIC", "X_fit": xs, "y_fit": y4,
                     "coef": np.asarray(direct.coef_, float), "intercept": np.asarray(direct.intercept_, float),
                     "C": C, "convergence_flag": "CONVERGED" if int(direct.n_iter_[0]) < EST.LOGISTIC_PARAMS["max_iter"] else "MAX_ITER_REACHED",
                     "n_iter": int(direct.n_iter_[0])},
                    {"objective": "0.5||W||^2 + C*sum loss"},
                    {"objective": J_ref, "kind": "ZERO_COEFFICIENT_FEASIBLE_POINT"})
                f3["tight_refit_coef_delta_vs_retained"] = coef_delta
                f3["tight_refit_prob_delta_vs_retained"] = p_delta
                f3["certificate_point"] = "TIGHT_REFIT_TOL_1E8"
                if p_delta > 1e-4:
                    f3["status"] = "FAIL"
                    f3["reason"] = "TIGHT_REFIT_PROBABILITY_DIVERGES_FROM_RETAINED_MODEL"
                rows.append({**base_row, "gate": "F3", "metric": "PENALIZED_SOLVER_KKT",
                             "observed_value": f3.get("kkt_scaled"),
                             "reference_value": f3.get("eps_solver"),
                             "comparator": "<=", "pass": f3["status"] == "PASS",
                             "status": f3["status"], "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                             "notes": json.dumps({k: f3.get(k) for k in
                                                  ("objective_hat", "objective_ref", "delta", "kkt_max_abs",
                                                   "kkt_scaled", "scale", "n_iter", "convergence_flag",
                                                   "tight_refit_coef_delta_vs_retained",
                                                   "tight_refit_prob_delta_vs_retained",
                                                   "certificate_point")})})
            else:
                rows.append({**base_row, "gate": "F3", "metric": "PENALIZED_SOLVER_KKT",
                             "observed_value": None, "reference_value": None,
                             "comparator": None, "pass": None, "status": "NOT_APPLICABLE",
                             "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                             "notes": "XGB tree boosting is discrete/non-convex; no Ridge/KKT certificate may be claimed"})
            # ---- F4 (real 19 placebo refits)
            L_real = G.logloss(y4, P)
            prior = G.class_prior_probs(y4)
            L_null_real = G.logloss(y4, np.tile(prior, (len(y4), 1)))
            skill_real = 1.0 - L_real / L_null_real
            placebos = G.devise_f4_placebos(y4, groups, B=B_PLACEBOS)

            def refit_and_loss(y_perm, fam=fam, cols=cols):
                mdl_b = EST.fit_two_family(PNL.matrix_for(dev, cols), cols, y_perm,
                                           family=fam, name=f"PLACEBO_{fam}")
                Pb = mdl_b.p4(PNL.matrix_for(dev, cols))
                return G.logloss(y_perm, Pb)

            seeded = []
            for pl in placebos:
                yb = pl["labels"]
                Lb = refit_and_loss(yb)
                prior_b = G.class_prior_probs(yb)
                L_null_b = G.logloss(yb, np.tile(prior_b, (len(yb), 1)))
                seeded.append({"placebo_id": pl["placebo_id"], "seed": pl["seed"],
                               "labels": yb, "null_loss": L_null_b, "_L": Lb})
            # refit over the seeded placebos (loss recomputed once, stored)
            def loss_lookup(y_perm):
                arr = np.asarray(y_perm)
                for s in seeded:
                    if np.array_equal(arr, s["labels"]):
                        return s["_L"]
                return refit_and_loss(y_perm)

            f4 = G.execute_f4_placebos_with_real_skill(L_real, skill_real, seeded, loss_lookup)
            f4_detail[f"{fam}:{mname}"] = {"real_loss": L_real, "null_loss_real": L_null_real,
                                           "skill_real": skill_real,
                                           "B": f4.get("B"), "k": f4.get("k"),
                                           "p_value": f4.get("p_value"),
                                           "max_placebo_skill": f4.get("max_placebo_skill"),
                                           "real_beats_max_placebo": f4.get("real_beats_max_placebo")}
            rows.append({**base_row, "gate": "F4", "metric": "LABEL_PLACEBO_SKILL",
                         "observed_value": f4.get("p_value"), "reference_value": G.F4_ALPHA,
                         "comparator": "<=", "pass": f4["status"] == "PASS",
                         "status": f4["status"], "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                         "notes": json.dumps({k: f4.get(k) for k in
                                              ("B", "k", "skill_real", "max_placebo_skill",
                                               "real_beats_max_placebo")})})
            # ---- F5
            if fam == "LOGISTIC":
                xs = mdl.matrix(X)
                pd4 = mdl.p4(X)
                f5 = G.compute_f5_fisher_edf({"family": "LOGISTIC", "C": float(EST.LOGISTIC_PARAMS["C"])},
                                             xs, pd4, {"lambda_": 1.0 / float(EST.LOGISTIC_PARAMS["C"])})
                rows.append({**base_row, "gate": "F5", "metric": "FISHER_EDF_RATIO",
                             "observed_value": f5.get("ratio"), "reference_value": G.F5_RATIO_CAP,
                             "comparator": "<=", "pass": f5["status"] == "PASS",
                             "status": f5["status"], "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                             "notes": json.dumps({k: f5.get(k) for k in
                                                  ("df_eff", "d_ident", "lambda_", "structural_pass",
                                                   "cap_pass", "min_eigenvalue", "contrast_basis")})})
            else:
                rows.append({**base_row, "gate": "F5", "metric": "FISHER_EDF_RATIO",
                             "observed_value": None, "reference_value": None,
                             "comparator": None, "pass": None, "status": "NOT_APPLICABLE",
                             "evidence_grade": "NEW_OWNER_RESEARCH_DEFAULT",
                             "notes": "XGB has no Fisher EDF; a separate typed complexity proxy would be required"})

    board = G.evaluate_five_gates_v2(rows, actual_run_manifest={"version": RUN_VERSION,
                                                                "n_models": 8, "B": B_PLACEBOS})
    # ---------------- ledgers
    v2 = pd.DataFrame(rows)
    v2.to_csv(out / "F1_F5_RESEARCH_OWNER_V2_ACCEPTANCE.csv", index=False)

    hist = pd.read_csv(rq01_dir / "RQ01_09_F1_F5.csv")
    hist = hist.assign(namespace="F1_F5_HISTORICAL_REV363_DIAGNOSTICS",
                       owner_pass="NOT_EVALUATED",
                       note="REV363 AUC/heuristic/Δlogloss; NOT the five typed fitting gates")
    hist.to_csv(out / "F1_F5_HISTORICAL_REV363_DIAGNOSTICS.csv", index=False)

    src_rows = []
    for (fam, m), (f1, f2) in R10_REFERENCE.items():
        src_rows += [
            {"gate": "F1", "family": fam, "model": m, "reference_value": f1,
             "unit": "nats/eligible DEV key", "n_context": 1260,
             "source": "R10_FIVE_FITTING_GATE_READINESS.csv (dossier §2.2/§2.2a)",
             "evidence_grade": "DOCUMENT_REPORTED_NOT_LOCALLY_VERIFIED",
             "owner_pass": "NOT_EVALUATED"},
            {"gate": "F2", "family": fam, "model": m, "reference_value": f2,
             "unit": "summed 4-class squared probability per key", "n_context": 1260,
             "source": "R10_FIVE_FITTING_GATE_READINESS.csv (dossier §2.2/§2.2a)",
             "evidence_grade": "DOCUMENT_REPORTED_NOT_LOCALLY_VERIFIED",
             "owner_pass": "NOT_EVALUATED"}]
    for gate, reason in (("F3", "AWAIT_ACTUAL_FIT_TRACE"),
                         ("F4", "AWAIT_PREREGISTERED_PLACEBO_REFITS"),
                         ("F5", "AWAIT_FISHER_EDF_AND_OWNER_CAPACITY")):
        src_rows.append({"gate": gate, "family": "ALL", "model": "ALL", "reference_value": None,
                         "unit": None, "n_context": 1260,
                         "source": "R10 status column", "evidence_grade": "SOURCE_RECOVERED",
                         "status": reason, "owner_pass": "NOT_EVALUATED"})
    pd.DataFrame(src_rows).to_csv(out / "F1_F5_SOURCE_RECOVERED_R10_REV317_EVIDENCE.csv", index=False)

    decisions = {
        "version": "RESEARCH_OWNER_DECISIONS_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "user research-owner mandate (dossier §1.1); NOT an owner signature",
        "decisions": [
            {"decision_id": "RD-01", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "preserve latest 34-card additive release; archived legacy preserved"},
            {"decision_id": "RD-02", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "T-C: retain incumbent thresholds; DEV band infeasible certificate"},
            {"decision_id": "RD-03", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "FIVE_FITTING_RESEARCH_GATES_V2 research namespace (no historical backfill)"},
            {"decision_id": "RD-04", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "frozen Owner C read-only; active votes remain binary"},
            {"decision_id": "RD-05", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "EW five-risky-sleeve geometry; ±0.05 absolute weight deviations"},
            {"decision_id": "RD-06", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "PC-021 active research Main; PC-017 historical comparator; two W0"},
            {"decision_id": "RD-07", "status": "USER_AUTHORIZED_RESEARCH",
             "scope": "fail-closed for EW/ES conflicts, FI phases, missing issuer data"},
            {"decision_id": "RD-08", "status": "PENDING_EXACT_SCOPE",
             "scope": "protected PC-012/PC-018 edits only after exact-scope dossier; none requested here"},
        ],
        "owner_signed": False,
    }
    (out / "RESEARCH_OWNER_DECISIONS_V2.json").write_text(
        json.dumps(decisions, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_version": RUN_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root), "rq01_dir": str(rq01_dir),
        "n_dev_keys": int(len(dev)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "keys_sha256": _sha_bytes(("|".join(f"{c}:{m}" for c, m in keys)).encode()),
        "frozen_dev_reproduction": repro,
        "model_digests": {f"{f}:{m}": fits[(f, m)]["digest"] for (f, m) in fits},
        "board": board,
        "f4_detail": f4_detail,
        "gate_constants": {"eps_solver": G.EPS_SOLVER, "delta_obj": G.DELTA_OBJ,
                           "F4_B": B_PLACEBOS, "F4_alpha": G.F4_ALPHA, "F5_ratio_cap": G.F5_RATIO_CAP},
        "outputs": {},
        "owner_certified": False,
    }
    for name in ("F1_F5_RESEARCH_OWNER_V2_ACCEPTANCE.csv",
                 "F1_F5_HISTORICAL_REV363_DIAGNOSTICS.csv",
                 "F1_F5_SOURCE_RECOVERED_R10_REV317_EVIDENCE.csv",
                 "RESEARCH_OWNER_DECISIONS_V2.json"):
        manifest["outputs"][name] = _sha_bytes((out / name).read_bytes())
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                                           encoding="utf-8")
    print("RUN_DONE", json.dumps({"n_rows": len(rows), "models": board["n_models"],
                                  "dev_gates_pass": board["FIVE_FITTING_DEV_DIAGNOSTICS_PASS"],
                                  "n_models_pass": board["n_models_dev_diagnostics_pass"]}))
    return manifest


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
