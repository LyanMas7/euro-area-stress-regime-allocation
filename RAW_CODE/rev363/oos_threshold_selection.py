"""Master-prompt §5/§6: OOS exploratory threshold selection with full exposure disclosure.

For every mechanism with per-row issued OOS scores AND genuine gold labels, enumerate
the exact score-threshold geometry on the OOS ESRB-scorable view, select the best
observed cut by a pre-registered rule, and record every trial.  Selected versions are
`OOS_SELECTED_EXPLORATORY` + `POST_OOS_EXPOSURE_RESEARCH`; they are NOT independent
validation, NOT frozen-before-OOS, and never relabeled as such.  Where denominators
are undefined or gold is missing, the record says UNDEFINED / NOT_EVALUATED.

Selection rule (pre-registered here, before reading OOS outcomes):
    primary   minimize total OOS misclassifications FN + FP
    tie-break minimize |FNR - 0.125| + |FPR - 0.20| (DEV rectangle center)
"""
import csv
import hashlib
import json
from pathlib import Path

RULE = "MIN_OOS_TOTAL_ERROR_FN_PLUS_FP_THEN_DEV_RECTANGLE_CENTER_DISTANCE"
FNR_LO, FNR_HI, FPR_LO, FPR_HI = 0.10, 0.15, 0.15, 0.25


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_csv(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _frozen_row(handoff: Path, mechanism_id: str):
    cal = handoff / "DEV_RUN_20261025_FNR_FPR_BAND_CALIBRATION" / "CALIBRATION" / "CALIBRATION_SUMMARY.csv"
    for r in _load_csv(cal) or []:
        if r["mechanism_id"] == mechanism_id:
            return {k: r.get(k) for k in ("selected_threshold", "selection_status",
                                          "oos_window", "oos_n", "oos_n_pos", "oos_n_neg",
                                          "oos_TP", "oos_FN", "oos_FP", "oos_TN", "oos_FNR", "oos_FPR",
                                          "generalization_gap_FNR", "generalization_gap_FPR", "oos_status")}
    return None


def _write_once(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return "UNCHANGED_IDENTICAL" if _sha(text) == hashlib.sha256(path.read_bytes()).hexdigest() else "UNCHANGED_DIFFERS"
    path.write_text(text, encoding="utf-8")
    return "WRITTEN"


def _enumerate_and_select(scores, labels, mechanism_id, frozen):
    if not scores:
        return None
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    trials = []
    best = None
    for cut in sorted(set(scores)):
        pred = [1 if x >= cut else 0 for x in scores]
        tp = sum(1 for a, b in zip(pred, labels) if a == 1 and b == 1)
        fn = sum(1 for a, b in zip(pred, labels) if a == 0 and b == 1)
        fp = sum(1 for a, b in zip(pred, labels) if a == 1 and b == 0)
        tn = sum(1 for a, b in zip(pred, labels) if a == 0 and b == 0)
        fnr = fn / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        dist = None
        if fnr is not None and fpr is not None:
            dist = abs(fnr - 0.125) + abs(fpr - 0.20)
        row = {"cut": cut, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
               "FNR": fnr, "FPR": fpr, "total_error": fn + fp, "center_dist": dist,
               "dev_band_pass": bool(FNR_LO <= fnr <= FNR_HI and FPR_LO <= fpr <= FPR_HI)
               if (fnr is not None and fpr is not None) else None}
        trials.append(row)
        if best is None or row["total_error"] < best["total_error"] or (
                row["total_error"] == best["total_error"]
                and row["center_dist"] is not None
                and (best.get("center_dist") is None or row["center_dist"] < best["center_dist"])):
            best = row
    return {"mechanism_id": mechanism_id, "rule": RULE, "n_oos": len(scores),
            "n_pos": n_pos, "n_neg": n_neg,
            "n_trials": len(trials), "best": best, "trials": trials,
            "frozen_comparison": frozen,
            "exposure": "POST_OOS_EXPOSURE_RESEARCH",
            "selected_status": "OOS_SELECTED_EXPLORATORY",
            "claim_limits": "NOT an independent OOS validation; NOT frozen before OOS; NOT proof of live generalization"}


def run(handoff_dir, threshold_root=None) -> dict:
    H = Path(handoff_dir)
    T = Path(threshold_root) if threshold_root else (H.parent / "Threshold")
    expl = T / "EXPLORATORY_OOS_SELECTED" / "THR_OOS_EXPLORATORY_20261025_R01"
    records = []

    # ---- RQ01 four-M on the 112-row ESRB-scorable OOS view
    tape = _load_csv(H / "DEV_RUN_20261025_FNR_FPR_BAND_CALIBRATION" / "RQ01_DEV_OOS_RERUN_F3B" /
                     "RQ01_10_ESRB_SCORABLE_EVALUATION.csv") or []
    sc = [r for r in tape if (r.get("scorable") or "1").strip().lower() not in ("false", "0", "")]
    for fam in ("LOGISTIC", "XGBOOST"):
        for model in ("M00", "M01", "M10", "M11"):
            mid = f"RQ01_{model}_{fam}"
            labels, scores = [], []
            for r in sc:
                g = (r.get("gold_label") or "").strip()
                try:
                    pn = float(r[f"p_{fam}_{model}_N"])
                except (KeyError, ValueError):
                    continue
                if g == "N":
                    labels.append(0); scores.append(1.0 - pn)
                elif g in ("G", "D", "BOTH"):
                    labels.append(1); scores.append(1.0 - pn)
                else:
                    continue  # unknown gold -> excluded, never assumed Normal
            rec = _enumerate_and_select(scores, labels, mid, _frozen_row(H, mid))
            if rec is None:
                records.append({"mechanism_id": mid, "selected_status": "NOT_EVALUATED",
                                "reason": "no evaluable score/gold pairs"})
                continue
            _write_once(expl / "RQ01" / f"{mid}_TRIALS.csv",
                        "cut,TP,FN,FP,TN,FNR,FPR,total_error,center_dist,dev_band_pass\n" + "\n".join(
                            f"{t['cut']},{t['TP']},{t['FN']},{t['FP']},{t['TN']},{t['FNR']},{t['FPR']},"
                            f"{t['total_error']},{t['center_dist']},{t['dev_band_pass']}" for t in rec["trials"]))
            _write_once(expl / "RQ01" / f"{mid}_SELECTED.json", json.dumps(
                {k: v for k, v in rec.items() if k != "trials"}, indent=2))
            records.append({k: v for k, v in rec.items() if k != "trials"})

    # ---- RQ02 ENTRY: per-row gold join attempt (no gold => NOT_EVALUATED, never assumed)
    p_tape = _load_csv(H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_03" / "RQ02_03_ISSUED_P_TAPE.csv")
    state = _load_csv(H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_03" / "RQ02_03_STATE_METRICS.csv")
    rq02_done = 0
    if p_tape is not None and state is not None:
        label_col = next((c for c in state[0] if c.lower() in ("label", "gold", "gold_label", "episode_on")), None)
        month_col = next((c for c in state[0] if c.lower() == "month"), None)
        if label_col and month_col:
            gold_map = {(r.get("country"), r.get(month_col)): r.get(label_col) for r in state}
            rq02_rows = [r for r in p_tape if (r.get("oos_view") or "").startswith("OOS_ESRB")]
            for horizon in ("H15", "H12", "H6", "H3", "H18"):
                for model in ("M00", "M01", "M10", "M11"):
                    for fam in ("LOGISTIC", "XGBOOST"):
                        mid = f"RQ02_ENTRY_{horizon}_{model}_{fam}"
                        labels, scores = [], []
                        for r in rq02_rows:
                            g = gold_map.get((r.get("country"), r.get("month")))
                            key = f"p_on__{horizon}__{model}__{fam}"
                            if g in (None, "") or key not in r:
                                continue
                            try:
                                p = float(r[key])
                            except ValueError:
                                continue
                            if str(g).strip().upper() in ("N", "0", "FALSE", "NO"):
                                labels.append(0)
                            else:
                                labels.append(1)
                            scores.append(p)
                        rec = _enumerate_and_select(scores, labels, mid, _frozen_row(H, mid))
                        if rec is not None:
                            rq02_done += 1
                            _write_once(expl / "RQ02" / f"{mid}_TRIALS.csv",
                                        "cut,TP,FN,FP,TN,FNR,FPR,total_error,center_dist,dev_band_pass\n" + "\n".join(
                                            f"{t['cut']},{t['TP']},{t['FN']},{t['FP']},{t['TN']},{t['FNR']},{t['FPR']},"
                                            f"{t['total_error']},{t['center_dist']},{t['dev_band_pass']}" for t in rec["trials"]))
                            _write_once(expl / "RQ02" / f"{mid}_SELECTED.json", json.dumps(
                                {k: v for k, v in rec.items() if k != "trials"}, indent=2))
                            records.append({k: v for k, v in rec.items() if k != "trials"})
    if not rq02_done:
        records.append({"mechanism_id": "RQ02_ENTRY_ALL", "selected_status": "NOT_EVALUATED",
                        "reason": "no per-row OOS gold join available in this evidence set; "
                                  "FNR/FPR are not computed without labels (master prompt §6)"})

    summary = _write_once(expl / "OOS_SELECTION_SUMMARY.csv",
                          "mechanism_id,selected_status,n_oos,n_pos,n_neg,n_trials,best_cut,best_FNR,best_FPR,"
                          "frozen_oos_FNR,frozen_oos_FPR,reason\n" + "\n".join(
                              (f"{r.get('mechanism_id')},{r.get('selected_status')},{r.get('n_oos','')},"
                               f"{r.get('n_pos','')},{r.get('n_neg','')},{r.get('n_trials','')},"
                               f"{(r.get('best') or {}).get('cut','')},{(r.get('best') or {}).get('FNR','')},"
                               f"{(r.get('best') or {}).get('FPR','')},"
                               f"{(r.get('frozen_comparison') or {}).get('oos_FNR','')},"
                               f"{(r.get('frozen_comparison') or {}).get('oos_FPR','')},{r.get('reason','')}")
                              for r in records))
    meta = {"status": "OOS_SELECTION_EXECUTED_20261025", "rule": RULE,
            "exposure": "POST_OOS_EXPOSURE_RESEARCH",
            "claim_limits": "not independent validation; not frozen before OOS; multiplicity disclosed per trial file",
            "n_rq01_selected": sum(1 for r in records if r.get("selected_status") == "OOS_SELECTED_EXPLORATORY"),
            "n_rq02_selected": rq02_done}
    _write_once(expl / "SELECTION_METADATA.json", json.dumps(meta, indent=2))
    return {"records": records, "meta": meta, "summary_written": summary,
            "threshold_exploratory_dir": str(expl)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--handoff", type=Path, required=True)
    args = p.parse_args()
    result = run(args.handoff)
    print(json.dumps({"meta": result["meta"], "summary_written": result["summary_written"]}, indent=2))


if __name__ == "__main__":
    main()
