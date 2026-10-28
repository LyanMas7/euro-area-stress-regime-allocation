"""RQ02 end-to-end runner: RQ02-01 ... RQ02-11 on DEV (2005-2019) and OOS (2020-2024), DEC-015/DEC-017.

    python -m rev363.rq02_e2e --dataset-root <Dataset> --rq01-dir <RQ01_NEXT_FINAL> --out <dir> \
        --period FULL --oos-authorization USER_AUTHORIZED_OOS_20261025_DEC015 \
        --expanding-authorization USER_AUTHORIZED_RQ02_EXPANDING_20261025_DEC017

`--period DEV` is the legal DEV-only path (no OOS byte is read; expanding and OOS views are NOT_RUN).
Every selection (IJP profile, pairs, models' thresholds, EXIT-A (profile, K), EXIT-B config, Bottom candidate) is
made on DEV rows only and frozen before any OOS row is scored.  Pre-registration:
AI_HANDOFF/APPENDIX/2026-10-25_RQ02_E2E_PREREGISTRATION.md.  Run with OMP/OPENBLAS/MKL_NUM_THREADS=1 so the RQ01
refit reproduces the issued tape bit-for-bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EA7
from .data import find_dataset_root
from .labels import load_episodes
from .rq02_e2e_contract import (DEV_FIRST, DEV_FIT_FIRST, DEV_LAST, ENTRY_MONTHS, EXIT_HORIZONS, OOS_FIRST,
                                OOS_LAST, POST_EXPOSURE, RQ02_E2E_VERSION, mi, ml, oos_view, require_expanding,
                                require_oos)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _months(a: str, b: str) -> list:
    return [ml(i) for i in range(mi(a), mi(b) + 1)]


def _w(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _j(obj, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Run:
    def __init__(self, dataset_root, rq01_dir, out, *, period, oos_authorization, expanding_authorization):
        self.root = find_dataset_root(dataset_root)
        self.rq01_dir = Path(rq01_dir)
        self.out = Path(out)
        self.period = period
        self.oos_token = oos_authorization
        self.exp_token = expanding_authorization
        self.run_id = f"RQ02_E2E_{period}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.files: dict = {}
        self.summary: dict = {"run_id": self.run_id, "version": RQ02_E2E_VERSION, "period": period,
                              "created_utc": datetime.now(timezone.utc).isoformat(),
                              "threads": {k: os.environ.get(k) for k in
                                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                              "oos_label": POST_EXPOSURE if period == "FULL" else "OOS_NOT_READ", "timing_s": {}}
        self.last_month = OOS_LAST if period == "FULL" else DEV_LAST
        self.asof_final = OOS_LAST if period == "FULL" else DEV_LAST
        if period == "FULL":
            self.summary["oos_gate"] = require_oos(oos_authorization)

    def save(self, rel: str, obj) -> None:
        p = self.out / rel
        self.files[rel] = _w(obj, p) if isinstance(obj, pd.DataFrame) else _j(obj, p)

    def tick(self, name, t0):
        self.summary["timing_s"][name] = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ data / RQ01 / gold
    def stage_data(self):
        from . import rq02_e2e_native as N
        from .rq02_e2e_ijp import IJPEngine
        from .rq02_e2e_labels import GoldBook
        from .rq02_e2e_rq01 import rq01_issued_view
        t0 = time.time()
        f_dev = N.native_frame7(self.root, data_end="2019-12-31")
        self.tape_dev = N.NativeTape(N.reduce_to_month_snapshots(f_dev), label="DEV")
        self.ref = N.ReferenceBook(self.tape_dev)
        ev = {"dev_native_rows": int(len(f_dev)), "dev_tape_rows": self.tape_dev.n_rows,
              "dev_tape_sha256": self.tape_dev.sha256}
        if self.period == "FULL":
            f_full = N.native_frame7(self.root, data_end="2024-12-31")
            dev_part = f_full[f_full["available_at"] <= "2019-12-31"].reset_index(drop=True)
            cols = ["country", "source", "native_period", "period_end", "available_at", "value"]
            same = dev_part[cols].reset_index(drop=True).equals(f_dev[cols].reset_index(drop=True))
            ev.update({"full_native_rows": int(len(f_full)), "dev_rows_identical_in_full_tape": bool(same)})
            if not same:
                raise ValueError("NATIVE_REVISION_CHANGES_DEV_ROWS_LOOKAHEAD_RISK")
            self.tape = N.NativeTape(N.reduce_to_month_snapshots(f_full), label="FULL")
            ev["full_tape_sha256"] = self.tape.sha256
            self.save("RQ02_01/RQ02_01_NATIVE_COVERAGE_OOS.csv",
                      N.coverage_table(f_full, period_first=OOS_FIRST, period_last=OOS_LAST))
        else:
            self.tape = self.tape_dev
        self.save("RQ02_01/RQ02_01_NATIVE_COVERAGE_DEV.csv",
                  N.coverage_table(f_dev, period_first=DEV_FIRST, period_last=DEV_LAST))
        self.save("RQ02_01/RQ02_01_THRESHOLD_REFERENCE_SUPPORT.csv", pd.DataFrame(self.ref.support()))
        self.engine_dev = IJPEngine(self.tape_dev, self.ref, owner="RQ02_E2E_ENTRY")
        self.engine = IJPEngine(self.tape, self.ref, owner="RQ02_E2E_ENTRY")
        self.engine_exit_dev = IJPEngine(self.tape_dev, self.ref, owner="RQ02_E2E_EXIT_A")
        self.engine_exit = IJPEngine(self.tape, self.ref, owner="RQ02_E2E_EXIT_A")
        view, panel, meta = rq01_issued_view(self.root, self.rq01_dir, period=self.period,
                                             oos_authorization=self.oos_token)
        self.view = view[view["month"] >= DEV_FIRST].reset_index(drop=True)
        self.panel = panel
        self.rq01_meta = meta
        self.gold = GoldBook(load_episodes(self.root))
        self.summary["data"] = ev
        self.summary["rq01_interface"] = meta
        self.tick("data", t0)

    # ------------------------------------------------------------------ ENTRY IJP (RQ02-01 / 02)
    def stage_entry(self):
        from . import rq02_e2e_entry as E
        from .rq02_e2e_ijp import profile_grid
        t0 = time.time()
        dev_months = _months(DEV_FIRST, DEV_LAST)
        table, rank, chosen, frames = E.profile_search(self.engine_dev, profile_grid(), self.gold, EA7, dev_months)
        self.profile = next(p for p in profile_grid() if p.pid == chosen)
        self.save("RQ02_01/RQ02_01_PROFILE_SEARCH.csv", table)
        self.save("RQ02_01/RQ02_01_PROFILE_RANK.csv", rank)
        dev_frame = frames[chosen]
        pairs = E.pair_selection(dev_frame)
        self.selected_pairs = pairs.loc[(pairs["branch"] == "SHORT") & (pairs["decision"] == "SELECTED"), "pair"].tolist()
        self.save("RQ02_02/RQ02_02_PAIR_SELECTION.csv", pairs)
        months = _months(DEV_FIRST, self.last_month)
        comp = E.component_frame(self.engine, self.profile, EA7, months)
        # FULL-tape DEV issues must equal the DEV-tape DEV issues (as-of invariance of the native tape)
        dev_cols = [c for c in comp.columns if c.startswith(("IS__", "JS__", "IL__", "JL__", "POOL__", "CNT__"))]
        a = comp[comp["month"] <= DEV_LAST].reset_index(drop=True)[dev_cols]
        b = dev_frame.reset_index(drop=True)[dev_cols]
        self.summary["entry_asof_invariance_dev_rows_equal"] = bool(a.fillna(-9).equals(b.fillna(-9)))
        comp = E.attach_entry_targets(comp, self.gold, self.asof_final)
        dev_t = E.attach_entry_targets(comp[["country", "month"]], self.gold, DEV_LAST)
        for h in ENTRY_MONTHS:
            comp[f"y_on_dev__{h}"] = dev_t[f"y_on__{h}"].to_numpy()
            comp[f"y_st_dev__{h}"] = dev_t[f"y_st__{h}"].to_numpy()
        self.comp = comp
        self.save("RQ02_01/RQ02_01_COMPONENT_TAPE.csv", comp)
        self.summary["entry_profile"] = {**self.profile.as_dict(), "rank": rank.iloc[0].to_dict(),
                                         "n_profiles": int(len(rank)), "n_selected_pairs": len(self.selected_pairs),
                                         "selected_pairs": self.selected_pairs}
        self.tick("entry_profile_search", t0)
        t0 = time.time()
        ind, jnt, pools = E.official_tape(self.engine, self.profile, EA7, months)
        self.save("RQ02_01/RQ02_01_INDIVIDUAL_TAPE.csv", ind)
        self.save("RQ02_02/RQ02_02_JOINT_TAPE.csv", jnt)
        self.save("RQ02_02/RQ02_02_POOL_TAPE.csv", pools)
        self.save("RQ02_01/RQ02_01_INDIVIDUAL_SUMMARY.csv", self._individual_summary(ind))
        # fast vs official parity on every issue
        mism = 0
        for h in E.H_IJP:
            st = E.ENTRY_STAGE_OF[h]
            o = pools[(pools["horizon"] == h) & (pools["role"] == "ComplementConfirmed")].set_index(["country", "issue"])
            f = comp.set_index(["country", "month"])[f"POOL__ComplementConfirmed__{h}"]
            ov = pd.to_numeric(o["vote"], errors="coerce").reindex(f.index)
            mism += int(((ov.fillna(-9)) != (f.fillna(-9))).sum())
        self.summary["entry_fast_vs_official_pool_mismatch"] = mism
        self.tick("entry_official_tape", t0)
        t0 = time.time()
        proofs = E.motif_proofs(self.engine, self.profile, EA7, months)
        self.save("RQ02_02/RQ02_02_MOTIF_PROOFS_S4_9MOTIFS.csv", proofs)
        self.save("RQ02_02/RQ02_02_MOTIF_SUMMARY.csv", E.motif_summary(proofs, comp))
        self.tick("entry_motif_proofs", t0)

    def _individual_summary(self, ind: pd.DataFrame) -> pd.DataFrame:
        y = self.comp[["country", "month"] + [f"y_on_dev__{h}" for h in ENTRY_MONTHS] +
                      [f"y_on__{h}" for h in ENTRY_MONTHS]].rename(columns={"month": "issue"})
        d = ind.merge(y, on=["country", "issue"], how="left")
        d["period"] = np.where(d["issue"] <= DEV_LAST, "DEV", "OOS")
        rows = []
        for (src, br, h, per), g in d.groupby(["source", "branch", "horizon", "period"]):
            v = pd.to_numeric(g["vote"], errors="coerce")
            ycol = f"y_on_dev__{h}" if per == "DEV" else f"y_on__{h}"
            gg = g[(g["issue"] >= DEV_FIT_FIRST)]
            vv = pd.to_numeric(gg["vote"], errors="coerce")
            yy = gg[ycol]
            ok = vv.notna() & yy.notna()
            tp = int(((vv == 1) & (yy == 1) & ok).sum()); fn = int(((vv == 0) & (yy == 1) & ok).sum())
            fp = int(((vv == 1) & (yy == 0) & ok).sum()); tn = int(((vv == 0) & (yy == 0) & ok).sum())
            rows.append({"source": src, "branch": br, "horizon": h, "period": per, "n_issues": int(len(g)),
                         "n_executed": int((g["status"] == "EXECUTED").sum()),
                         "n_not_evaluated": int((g["status"] == "NOT_EVALUATED").sum()),
                         "vote_rate": float(v.mean()) if v.notna().any() else None,
                         "mean_n_levels": float(pd.to_numeric(g["n_levels"], errors="coerce").mean()),
                         "mean_count_pairs_M": float(pd.to_numeric(g["count_pairs_M"], errors="coerce").mean()),
                         "mean_terminal_B": float(pd.to_numeric(g["terminal_bases_B"], errors="coerce").mean()),
                         "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                         "TPR": tp / (tp + fn) if tp + fn else None, "FPR": fp / (fp + tn) if fp + tn else None,
                         "target": f"onset_within_{ENTRY_MONTHS[h]}m", "scored": per == "DEV" or (tp + fn + fp + tn) > 0})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ models (RQ02-03 / 10 / 11)
    def stage_models(self):
        from . import rq02_e2e_models as MD
        from .rq02_e2e_entry import BASE_COLUMNS
        t0 = time.time()
        base = [c for c in BASE_COLUMNS if c in self.panel.columns]
        if len(base) != len(BASE_COLUMNS):
            raise ValueError("RQ02_BASE_COLUMNS_MISSING_IN_PANEL")
        data = self.comp.merge(self.panel[["country", "month"] + base], on=["country", "month"], how="left")
        self.data = data
        fz = MD.fit_frozen(data, self.selected_pairs, run_id=self.run_id)
        self.fz = fz
        self.save("RQ02_03/RQ02_03_ENTRY_METRICS.csv", pd.DataFrame(fz["metrics"]))
        self.save("RQ02_03/RQ02_03_STATE_METRICS.csv", pd.DataFrame(fz["state_metrics"]))
        tape = fz["tape"].copy()
        tape["oos_view"] = [oos_view(m) for m in tape["month"]]
        self.save("RQ02_03/RQ02_03_ISSUED_P_TAPE.csv", tape)
        self.save("RQ02_03/XGB_ESTIMATOR_LEDGER.csv", pd.DataFrame(fz["xgb_ledger"]))
        thr = [{"horizon": k[0], "model": k[1], "family": k[2], **{kk: vv for kk, vv in v.items()
                                                                   if not isinstance(vv, (list, dict))}}
               for k, v in fz["thresholds"].items()]
        self.save("RQ02_03/RQ02_03_THRESHOLD_REGISTER.csv", pd.DataFrame(thr))
        self.save("RQ02_03/RQ02_03_PUREJOINT_ORTHOGONALITY.csv",
                  pd.DataFrame([{"horizon": h, "max_abs_ZT_Jperp": d.orthogonality_max_abs,
                                 "n_joint_columns": len(d.joint_raw), "n_ind_columns": len(d.ind.columns)}
                                for h, d in fz["designs"].items()]))
        self.save("RQ02_10/RQ02_10_F1_F5.csv", MD.f1_f5_table(fz))
        acc = MD.accumulation(tape, self.comp, self.gold, asof=self.asof_final)
        self.save("RQ02_10/RQ02_10_ACCUMULATION_TAPE.csv", acc)
        self.save("RQ02_10/RQ02_10_ACCUMULATION_SUMMARY.csv", self._acc_summary(acc))
        self.tick("models_frozen", t0)
        if self.period == "FULL":
            t0 = time.time()
            require_expanding(self.exp_token, self.oos_token)
            ex = MD.expanding(data, fz["designs"], self.gold, expanding_authorization=self.exp_token,
                              run_id=self.run_id)
            self.save("RQ02_03/EXPANDING_FIT_LEDGER.csv", ex["ledger"])
            self.save("RQ02_03/EXPANDING_PREDICTIONS.csv", ex["predictions"])
            self.expanding_preds = ex["predictions"]
            from .rq02_e2e_labels import GoldBook
            base_eps = load_episodes(self.root)

            def factory(perturb):
                eps = [dict(e) for e in base_eps]
                if perturb:
                    import datetime as _dt
                    for e in eps:
                        if e["start"] > _dt.date(2020, 9, 30):
                            e["start"] = e["start"] - _dt.timedelta(days=31)
                    eps.append({"country": "DE", "start": _dt.date(2020, 11, 30), "normal": _dt.date(2021, 2, 28),
                                "origin": "G", "systemic": 1, "dev_eligibility": "OOS_ONLY_SEALED",
                                "ea7_scope": "EA7", "role": "PERTURBATION"})
                return GoldBook(eps)
            probes = [MD.expanding_invariance_probe(data, fz["designs"], factory, h=h, u=u)
                      for h, u in (("H6", "2020-09"), ("H12", "2021-01"), ("H15", "2020-06"))]
            self.save("RQ02_03/EXPANDING_INVARIANCE.json", probes)
            self.save("RQ02_03/RQ02_03_E0_E3_COMPARISON.csv", self._e_comparison(tape, ex["predictions"]))
            self.tick("models_expanding", t0)

    def _acc_summary(self, acc: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if acc.empty:
            return pd.DataFrame(rows)
        acc = acc[acc["gold_at_T"].notna() & acc["gold_released"]]
        for grp, g in (("TARGET_STRESS", acc[acc["gold_at_T"] != "N"]), ("TARGET_N", acc[acc["gold_at_T"] == "N"]),
                       ("TARGET_ONSET_MONTH", acc[acc["is_onset_month"]])):
            rec = {"group": grp, "n_targets": int(len(g)),
                   "period_of_target": "DEV+OOS_SCORABLE"}
            for h in ENTRY_MONTHS:
                rec[f"mean_P_stress_{h}"] = float(g[f"P_stress_{h}"].mean()) if len(g) else None
            for h in ("H12", "H10", "H8", "H6"):
                if f"long_confirmations_{h}" in g:
                    rec[f"mean_long_conf_{h}"] = float(g[f"long_confirmations_{h}"].mean()) if len(g) else None
                    rec[f"mean_short_src_{h}"] = float(g[f"short_sources_{h}"].mean()) if len(g) else None
            rows.append(rec)
        return pd.DataFrame(rows)

    def _e_comparison(self, tape: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
        from . import rq02_e2e_models as MD
        rows = []
        lab = self.comp.set_index(["country", "month"])
        for h in ENTRY_MONTHS:
            for mname in MD.M_NAMES:
                for fam, e_froz, e_exp in (("LOGISTIC", "E0", "E1"), ("XGBOOST", "E2", "E3")):
                    pe = preds[(preds["horizon"] == h) & (preds["model"] == mname) & (preds["family"] == fam)]
                    if pe.empty:
                        continue
                    pe = pe.set_index(["country", "month"])
                    fr = tape.set_index(["country", "month"])[f"p_on__{h}__{mname}__{fam}"].reindex(pe.index)
                    frd = tape.set_index(["country", "month"])[f"dec_on__{h}__{mname}__{fam}"].reindex(pe.index)
                    y = lab[f"y_on__{h}"].reindex(pe.index)
                    sc = y.notna()
                    for e_id, s, d in ((e_froz, fr, frd), (e_exp, pe["p_on"], pe["dec_on"])):
                        rec = {"horizon": h, "model": mname, "estimator": fam, "experiment": e_id,
                               "update": "FROZEN_DEV" if e_id in ("E0", "E2") else "OOS_EXPANDING_ASOF",
                               "n_oos_issued": int(len(pe)), "n_oos_scorable": int(sc.sum()),
                               "mean_p_oos": float(s.mean()), "alarm_rate_oos": float(d.mean()),
                               "status": "EXECUTED", "label": POST_EXPOSURE}
                        if sc.any():
                            bm = MD.binary_metrics(y[sc].astype(int).to_numpy(), s[sc].to_numpy(), 0.5)
                            rec.update({k: bm.get(k) for k in ("n", "events", "auc", "logloss", "brier")})
                            yy, dd = y[sc].astype(int).to_numpy(), d[sc].astype(int).to_numpy()
                            rec.update({"TP_frozen_cut": int(((dd == 1) & (yy == 1)).sum()),
                                        "FN_frozen_cut": int(((dd == 0) & (yy == 1)).sum()),
                                        "FP_frozen_cut": int(((dd == 1) & (yy == 0)).sum()),
                                        "TN_frozen_cut": int(((dd == 0) & (yy == 0)).sum())})
                        rows.append(rec)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ S5 (RQ02-04)
    def stage_s5(self):
        from .rq02_e2e_assets import weekly_levels
        from .rq02_e2e_s5 import run_s5
        t0 = time.time()
        self.levels, lev_ev = weekly_levels(self.root, period=self.period, oos_authorization=self.oos_token)
        self.summary["asset_levels"] = lev_ev
        tape_path = self.out / "RQ02_03/RQ02_03_ISSUED_P_TAPE.csv"
        before = hashlib.sha256(tape_path.read_bytes()).hexdigest()
        endpoints, votes = run_s5(self.engine, self.levels, self.gold, self.profile, last_month=self.last_month)
        after = hashlib.sha256(tape_path.read_bytes()).hexdigest()
        endpoints["sample"] = np.where(endpoints["onset"] <= "2019-12-31", "DEV_EPISODE", "OOS_EPISODE")
        self.save("RQ02_04/RQ02_04_S5_ENDPOINTS.csv", endpoints)
        self.save("RQ02_04/RQ02_04_S5_SHORT_VOTES.csv", votes)
        self.save("RQ02_04/RQ02_04_S5_H6_INVARIANCE.json",
                  {"issued_entry_tape_sha_before": before, "issued_entry_tape_sha_after": after,
                   "status": "PASS_S5_NEVER_EDITS_H6" if before == after else "FAIL"})
        self.s5 = endpoints
        self.tick("s5", t0)

    # ------------------------------------------------------------------ EXIT (RQ02-05 / 06 / 07)
    def stage_exit(self):
        from . import rq02_e2e_exit as X
        from .rq02_e2e_ijp import profile_grid
        t0 = time.time()
        months = _months(DEV_FIT_FIRST, self.last_month)
        retro = [(c, m) for c in EA7 for m in months if self.gold.gold(c, m) not in (None, "N")]
        issued_set = {(c, m) for c, m, s in zip(self.view["country"], self.view["month"], self.view["rq01_stress"])
                      if s and DEV_FIT_FIRST <= m <= self.last_month}
        dev_retro = [k for k in retro if k[1] <= DEV_LAST]
        fa = X.exit_a_frame(self.engine_exit_dev, profile_grid(), dev_retro)
        fa["y_exact_dev"] = [self.gold.exit_target(c, u, h, DEV_LAST).get("exact", np.nan)
                             for c, u, h in zip(fa["country"], fa["month"], fa["h"])]
        tbl_a, chosen_a = X.select_exit_a(fa, "y_exact_dev")
        self.save("RQ02_05/RQ02_05_EXIT_A_SEARCH.csv", tbl_a)
        self.summary["exit_a_selected"] = chosen_a
        keys = sorted(set(retro) | issued_set)
        prof = {p.pid: p for p in profile_grid()}
        rows = []
        for c, u in keys:
            for h in EXIT_HORIZONS:
                ch = chosen_a[h]
                comp = self.engine_exit.exit_fast(c, u, h, prof[ch["profile_id"]])
                fam = self.engine_exit.family_votes(comp)
                v = self.engine_exit.exit_a_vote(fam, ch["K"], X.K_OWNER if ch["K"] in (2, 3) else None)
                # two-layer row: A_vote is PASS/FAIL only when the K-of-4 decision was executed (no 'U')
                rows.append({"country": c, "issue": u, "horizon": h, "profile_id": ch["profile_id"], "K": ch["K"],
                             **{f"fam_{k}": val for k, val in fam.items()},
                             **{f"fam_{k}_status": ("EXECUTED" if val in (0, 1) else "NOT_EVALUATED")
                                for k, val in fam.items()},
                             "A_vote": {1: "PASS", 0: "FAIL"}.get(v),
                             "A_status": "EXECUTED" if v in (0, 1) else "NOT_EVALUATED",
                             "A_reason": None if v in (0, 1) else "EXIT_A_K_OF_4_NOT_DETERMINED"})
        ta = pd.DataFrame(rows)
        stape = X.StateTape(self.view, self.tape)
        diags = {(c, u): X.b_diagnostics(stape, c, u) for c, u in keys}
        dev_b = [(c, u, h, self.gold.exit_target(c, u, h, DEV_LAST).get("exact"))
                 for c, u in keys if u <= DEV_LAST and (c, u) in issued_set and self.gold.gold(c, u) not in (None, "N")
                 for h in EXIT_HORIZONS]
        dev_b = [k for k in dev_b if k[3] is not None]
        tbl_b, chosen_b = X.select_exit_b(diags, dev_b, self.ref)
        self.save("RQ02_06/RQ02_06_EXIT_B_SEARCH.csv", tbl_b)
        self.summary["exit_b_selected"] = chosen_b
        bt = []
        for c, u in keys:
            thr = X.b_source_thresholds(self.ref, c)
            d = diags[(c, u)]
            for h in EXIT_HORIZONS:
                r = X.b_vote(d, chosen_b[h], thr)
                bt.append({"country": c, "issue": u, "horizon": h, "config_id": chosen_b[h]["config_id"],
                           "B_vote": r.get("vote"), "B_status": r["execution_status"], "B_reason": r.get("reason"),
                           "diag_status": d.get("status"),
                           **{f"gate_{k}": v for k, v in (r.get("gates") or {}).items()},
                           "gates_not_evaluated": ";".join(sorted((r.get("gates_not_evaluated") or {}))),
                           **{f"diag_{k}": d.get(k) for k in ("age", "gap", "span", "n_states", "max_rank",
                                                               "n_native", "anchor_state")}})
        tb = pd.DataFrame(bt)
        dev_keys_b = [(c, u) for c, u in keys if u <= DEV_LAST and diags[(c, u)].get("status") == "OK"]
        parity = X.b_engine_parity(stape, self.ref, dev_keys_b, [chosen_b[h] for h in EXIT_HORIZONS])
        self.save("RQ02_06/RQ02_06_EXIT_B_ENGINE_PARITY.json", parity)
        self.summary["exit_b_parity"] = {k: parity[k] for k in ("n_compared", "n_mismatch", "status")}
        ex = ta.merge(tb, on=["country", "issue", "horizon"], how="outer")
        lab = [self.gold.exit_target(c, u, int(h), self.asof_final) for c, u, h in
               zip(ex["country"], ex["issue"], ex["horizon"])]
        ex["gold_risk_set"] = [self.gold.gold(c, u) not in (None, "N") for c, u in zip(ex["country"], ex["issue"])]
        ex["issued_risk_set"] = [(c, u) in issued_set for c, u in zip(ex["country"], ex["issue"])]
        ex["label_status"] = [d["status"] for d in lab]
        ex["y_exact"] = [d.get("exact") for d in lab]
        ex["y_by"] = [d.get("by") for d in lab]
        ex["first_n"] = [d.get("first_n") for d in lab]
        ex["episode_id"] = [d.get("episode_id") for d in lab]
        ex["period"] = np.where(ex["issue"] <= DEV_LAST, "DEV", "OOS")
        ex["oos_view"] = [oos_view(m) for m in ex["issue"]]
        ex["fusion_AND"] = [X.fuse(a, b, "AND") for a, b in zip(ex["A_vote"], ex["B_vote"])]
        ex["fusion_OR"] = [X.fuse(a, b, "OR") for a, b in zip(ex["A_vote"], ex["B_vote"])]
        for col in ("fusion_AND", "fusion_OR"):
            ex[col + "_status"] = np.where(ex[col].isin(["PASS", "FAIL"]), "EXECUTED", "NOT_EVALUATED")
        self.save("RQ02_05/RQ02_05_EXIT_AB_ISSUED_TAPE.csv", ex)
        self.save("RQ02_06/RQ02_06_EXIT_MATCHED.csv", X.matched_table(ex))
        self.save("RQ02_05/RQ02_05_EXIT_A_METRICS.csv", X.branch_metrics(ex, "A_vote"))
        self.save("RQ02_06/RQ02_06_EXIT_B_METRICS.csv", X.branch_metrics(ex, "B_vote"))
        self.save("RQ02_05/RQ02_05_EXIT_FITTED_METRICS.csv", X.fitted_exit(ex, self.data))
        iss = []
        for br, col in (("EXIT_A", "A_vote"), ("EXIT_B", "B_vote")):
            g = ex[ex["issued_risk_set"] & ex[col].isin(["PASS", "FAIL"])]
            iss.append(pd.DataFrame({"country": g["country"], "issue": g["issue"], "horizon": g["horizon"],
                                     "branch": br, "vote": g[col]}))
        led = X.override_ledger(pd.concat(iss, ignore_index=True), self.view)
        led["period"] = np.where(led["issued_at"] <= DEV_LAST, "DEV", "OOS")
        self.save("RQ02_07/RQ02_07_OVERRIDE_LEDGER.csv", led)
        summ = led.groupby(["period", "branch", "override_type"]).size().reset_index(name="n")
        self.save("RQ02_07/RQ02_07_OVERRIDE_SUMMARY.csv", summ)
        self.exit_tape = ex
        self.tick("exit", t0)

    # ------------------------------------------------------------------ Bottom (RQ02-08 / 09)
    def stage_bottom(self):
        from . import rq02_e2e_bottom as BT
        t0 = time.time()
        view = self.view[self.view["month"] <= self.last_month]
        eps = BT.episodes_from_view(view, self.last_month)
        book = BT.BottomBook(view, self.levels)
        grid = BT.candidate_grid()
        cand_rows = []
        dev_eps = [e for e in eps if e["window_end"] <= DEV_LAST]
        for cand in grid:
            cand_rows.append({**cand, **BT.summarize(BT.build_tape(book, dev_eps, cand))})
        sel = BT.select_candidate(cand_rows)
        cands = pd.DataFrame(cand_rows)
        cands["selected"] = cands["candidate_id"] == sel.get("candidate_id")
        self.save("RQ02_08/RQ02_08_BOTTOM_CANDIDATES_DEV.csv", cands)
        base = next(c for c in grid if c["is_baseline"])
        chosen = next((c for c in grid if c["candidate_id"] == sel.get("candidate_id")), base)
        self.summary["bottom_selected"] = {**chosen, "selection_status": sel.get("status"),
                                           "dev_hit_rate": sel.get("hit_rate"),
                                           "dev_forecast_coverage": sel.get("forecast_coverage"),
                                           "dev_mean_interval_score_weeks": sel.get("mean_interval_score_weeks"),
                                           "n_dev_episodes": len(dev_eps), "n_episodes": len(eps)}
        full = BT.build_tape(book, eps, chosen)
        full["period"] = np.where(full["arming_month"] <= DEV_LAST, "DEV", "OOS")
        self.save("RQ02_08/RQ02_08_BOTTOM_ISSUED_TAPE.csv", full)
        self.save("RQ02_08/RQ02_08_BOTTOM_BASELINE_TAPE.csv", BT.build_tape(book, eps, base))
        rows = []
        for per in ("DEV", "OOS"):
            g = full[full["period"] == per]
            rows.append({"period": per, "asset": "ALL5", **BT.summarize(g)})
            for a in ("E1", "E2", "E3", "B1", "B2"):
                rows.append({"period": per, "asset": a, **BT.summarize(g[g["asset"] == a])})
        self.save("RQ02_09/RQ02_09_BOTTOM_SCORES.csv", pd.DataFrame(rows))
        abl = BT.ablation(book, eps, chosen)
        self.save("RQ02_08/RQ02_08_BOTTOM_ABLATION.json", {k: v for k, v in abl.items()
                                                            if k not in ("ea5", "loss_table")})
        self.save("RQ02_08/RQ02_08_BOTTOM_ABLATION_LOSS_TABLE.csv", abl["loss_table"])
        ea5 = abl["ea5"]
        comp = []
        for per in ("DEV", "OOS"):
            ids = {e["episode_id"] for e in eps if (e["arming"] <= DEV_LAST) == (per == "DEV")}
            a = full[full["episode_id"].isin(ids)]
            comp.append({"period": per, "variant": "F3_ASSET_SPECIFIC", **BT.summarize(a)})
            if len(ea5):
                comp.append({"period": per, "variant": "F3_EA5_SHARED",
                             **BT.summarize(ea5[ea5["episode_id"].isin(ids)])})
        self.save("RQ02_08/RQ02_08_F3_ASSET_VS_EA5.csv", pd.DataFrame(comp))
        self.bottom = full
        self.bottom_eps = eps
        self.tick("bottom", t0)

    # ------------------------------------------------------------------ RQ02-11 packet / acceptance
    def stage_packet(self):
        from research_adapters.rq03 import main_lock as ML
        t0 = time.time()
        shorts, bottoms, exits, ends = [], [], [], []

        def stamp(m):
            return (pd.Timestamp(m) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%dT23:59:00Z")

        regime = {(c, m): s for c, m, s in zip(self.view["country"], self.view["month"], self.view["rq01_stress"])}
        bot = self.bottom[self.bottom["issue_month"].notna()]
        for c in EA7:
            g = self.comp[self.comp["country"] == c].sort_values("month")
            bmon = {}
            for r in bot[bot["country"] == c].itertuples(index=False):
                bmon.setdefault(r.issue_month, []).append(r)
            prev, open_ep, bottomed = 0, None, False
            for m, v in zip(g["month"], g["POOL__ComplementConfirmed__H6"]):
                cur = int(v == 1)
                if cur and not prev:                      # rising edge = issued Short confirmation
                    if open_ep is None:
                        open_ep, bottomed = f"{c}_RQ02EP_{m}", False
                    shorts.append({"short_id": f"{open_ep}_S_{m}", "episode_id": open_ep, "confirmed": True,
                                   "issued_at": stamp(m), "confirmed_at": stamp(m), "available_at": stamp(m),
                                   "issuer_seq": 0, "status": "ISSUED", "country": c,
                                   "source": "RQ02_H6_S4_COMPLEMENT_CONFIRMED_POOL"})
                prev = cur
                for r in bmon.get(m, []):
                    ep = open_ep if open_ep is not None else f"{c}_UNLINKED_BOTTOM_{m}"
                    bottoms.append({"record_id": f"{ep}_{r.asset}_B_{m}", "episode_id": ep, "issued_at": stamp(m),
                                    "confirmed_at": stamp(m), "available_at": stamp(m), "issuer_seq": 1,
                                    "status": "ISSUED", "country": c, "asset": r.asset,
                                    "linked_to_open_short_episode": open_ep is not None,
                                    "zone_lower": r.zone_lower, "zone_upper": r.zone_upper,
                                    "bottom_episode": r.episode_id})
                    if open_ep is not None:
                        bottomed = True
                if open_ep is not None and bottomed and regime.get((c, m)) is False and m not in bmon:
                    exits.append({"record_id": f"{open_ep}_X_{m}", "episode_id": open_ep, "issued_at": stamp(m),
                                  "confirmed_at": stamp(m), "available_at": stamp(m), "issuer_seq": 2,
                                  "status": "ISSUED", "country": c, "source": "RQ01_ISSUED_REGIME_N_AFTER_BOTTOM"})
                    nxt = ml(mi(m) + 1)
                    ends.append({"record_id": f"{open_ep}_E_{nxt}", "episode_id": open_ep, "issued_at": stamp(nxt),
                                 "confirmed_at": stamp(nxt), "available_at": stamp(nxt), "issuer_seq": 3,
                                 "status": "ISSUED", "country": c})
                    open_ep, bottomed = None, False
        packet = {"producer": "RQ02_E2E", "run_id": self.run_id, "schema": "research_adapters.rq03.main_lock packet",
                  "scope": "PER_COUNTRY_ISSUED_EVENTS__EA_AGGREGATION_IS_AN_RQ03_OWNER_DECISION",
                  "episode_rule": "OPEN_AT_FIRST_ISSUED_SHORT; BOTTOM_ATTACHES_TO_OPEN_EPISODE; EXIT_AT_NEXT_ISSUED_RQ01_N",
                  "short_confirmations": shorts, "bottom_confirmations": bottoms, "exit_confirmations": exits,
                  "episode_end_records": ends}
        events = ML.lock_events_from_packet(packet)
        after_last = pd.Timestamp(self.last_month) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        vis = ML.visible_events(events, after_last.strftime("%Y-%m-%dT00:00:00Z"))
        self.save("RQ02_11/RQ02_11_RQ03_ISSUED_PACKET.json", packet)
        self.summary["rq03_packet"] = {"n_short": len(shorts), "n_bottom": len(bottoms), "n_exit": len(exits),
                                       "n_episode_end": len(ends), "n_events": len(events),
                                       "n_bottom_linked": sum(b["linked_to_open_short_episode"] for b in bottoms),
                                       "visible_check": vis.get("status"), "visible_reason": vis.get("reason")}
        self.tick("packet", t0)

    def stage_acceptance(self):
        from .rq02_e2e_acceptance import acceptance_table
        tbl = acceptance_table(self)
        self.save("RQ02_11/RQ02_ACCEPTANCE_STATUS.csv", tbl)
        self.summary["acceptance"] = tbl[["id", "status"]].to_dict(orient="records")
        self.summary["files_sha256"] = dict(self.files)
        _j(self.summary, self.out / "RQ02_E2E_SUMMARY.json")

    def stage_final_report(self):
        from .rq02_e2e_final_report import build_all
        self.summary["final_report"] = build_all(self.out)
        _j(self.summary, self.out / "RQ02_E2E_SUMMARY.json")

    def run_all(self):
        for st in ("stage_data", "stage_entry", "stage_models", "stage_s5", "stage_exit", "stage_bottom",
                   "stage_packet", "stage_acceptance", "stage_final_report"):
            getattr(self, st)()
            _j(self.summary, self.out / "RQ02_E2E_SUMMARY.json")
        return self.summary


def refresh_acceptance(out_dir) -> dict:
    """Recompute RQ02_ACCEPTANCE_STATUS.csv from the saved outputs of a finished run (no stage is re-run)."""
    import types
    from .rq02_e2e_acceptance import acceptance_table
    out = Path(out_dir)
    summary = json.loads((out / "RQ02_E2E_SUMMARY.json").read_text(encoding="utf-8"))
    run = types.SimpleNamespace(out=out, summary=summary, period=summary["period"], run_id=summary["run_id"])
    tbl = acceptance_table(run)
    sha = _w(tbl, out / "RQ02_11/RQ02_ACCEPTANCE_STATUS.csv")
    summary["acceptance"] = tbl[["id", "status"]].to_dict(orient="records")
    summary.setdefault("files_sha256", {})["RQ02_11/RQ02_ACCEPTANCE_STATUS.csv"] = sha
    summary["acceptance_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
    _j(summary, out / "RQ02_E2E_SUMMARY.json")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--refresh-acceptance", help="finished run folder: recompute only the acceptance table")
    p.add_argument("--dataset-root")
    p.add_argument("--rq01-dir")
    p.add_argument("--out")
    p.add_argument("--period", choices=("DEV", "FULL"), default="DEV")
    p.add_argument("--oos-authorization")
    p.add_argument("--expanding-authorization")
    a = p.parse_args(argv)
    if a.refresh_acceptance:
        s = refresh_acceptance(a.refresh_acceptance)
        print(json.dumps({"run_id": s["run_id"], "acceptance": s["acceptance"]}, indent=2))
        return
    if not (a.dataset_root and a.rq01_dir and a.out):
        p.error("--dataset-root, --rq01-dir and --out are required for a run")
    r = Run(a.dataset_root, a.rq01_dir, a.out, period=a.period, oos_authorization=a.oos_authorization,
            expanding_authorization=a.expanding_authorization)
    s = r.run_all()
    print(json.dumps({"run_id": s["run_id"], "timing_s": s["timing_s"], "acceptance": s["acceptance"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
