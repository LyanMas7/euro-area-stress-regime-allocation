"""Read-only RQ01 issued interface for RQ02 (current regime, P4, canonical six-phase state, BASE panel).

RQ01 code is CALLED, never modified: `panel.build_panel` -> corrected M11 block -> `estimators.fit_two_family`
(LOGISTIC, DEV 2005-07..2019-12), the frozen RQ01 threshold of `RQ01_08_THRESHOLD_REGISTER.json`.  For the
FULL chronology the refit must reproduce the issued OOS tape `RQ01_10_ISSUED_TAPE_OOS.csv` (P4 within 1e-8 and
identical decisions), otherwise the run stops.  DEV RQ01 probabilities are IN-SAMPLE (RQ01 was fitted on DEV).

Canonical phase: `RQ01_CANONICAL_PHASE_FULL.csv` (DEV-frozen snapshots replayed through OOS) or the DEV file;
FI has no canonical phase (ISS-025) -> EXIT-B NOT_EVALUATED for FI, Bottom F1/F2 use the labelled legacy fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import estimators as EST
from . import panel as PNL
from .config import CLASSES
from .rq02_e2e_contract import require_oos

CANON_TO_BOTTOM = {"PEAK": "PEAK", "DE_ESCALATION": "DEESC", "RECOVERY": "RECOVERY", "ENTRY": "ENTRY",
                   "EXIT": "EXIT", "PRE_ONSET": "PRE_ONSET", "NONE": "NONE"}
LEGACY_TO_BOTTOM = {"PEAK": "PEAK", "DEESC": "DEESC", "RECOVERY": "RECOVERY", "PRE": "PRE_ONSET",
                    "ENTRY": "ENTRY", "EXIT": "EXIT"}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rq01_issued_view(dataset_root, rq01_dir, *, period: str, oos_authorization=None) -> tuple:
    rq01_dir = Path(rq01_dir)
    reg = json.loads((rq01_dir / "RQ01_08_THRESHOLD_REGISTER.json").read_text(encoding="utf-8"))
    if reg.get("selected_family") != "LOGISTIC" or reg.get("selected_model") != "M11":
        raise ValueError("RQ01_SELECTED_MODEL_IS_NOT_LOGISTIC_M11")
    theta = float(reg["selected_threshold"])
    if period == "FULL":
        gate = require_oos(oos_authorization)
        data_end = "2024-12"
        phase_file = rq01_dir / "RQ01_CANONICAL_PHASE_FULL.csv"
    elif period == "DEV":
        gate = "DEV_ONLY_NO_OOS_READ"
        data_end = "2019-12"
        phase_file = rq01_dir / "RQ01_DEV_CANONICAL_PHASE.csv"
    else:
        raise ValueError("PERIOD_DEV_OR_FULL")
    prep = PNL.build_panel(dataset_root, row_start="2004-01", row_end=data_end, data_end=data_end)
    panel, spec = prep["panel"], prep["spec"]
    cols = PNL.corrected_blocks(spec)["M11"]
    dev = ((panel["month"] >= "2005-07") & (panel["month"] <= "2019-12")).to_numpy()
    mdl = EST.fit_two_family(PNL.matrix_for(panel[dev], cols), cols, PNL.y4_from(panel[dev]),
                             family="LOGISTIC", name="M11")
    p4 = mdl.p4(PNL.matrix_for(panel, cols))
    view = panel[["country", "month", "phase", "dd_CLIFS", "z_CISS"]].copy()
    for k, c in enumerate(CLASSES):
        view[f"p_rq01_{c}"] = p4[:, k]
    view["p_rq01_stress"] = 1.0 - p4[:, 0]
    sub = np.argmax(p4[:, 1:], axis=1)
    view["rq01_dec"] = np.where(view["p_rq01_stress"] >= theta, np.array(CLASSES[1:])[sub], "N")
    view["rq01_stress"] = view["rq01_dec"] != "N"
    view["rq01_theta"] = theta
    view["rq01_sample"] = np.where(view["month"] <= "2019-12", "DEV_IN_SAMPLE", "OOS_FROZEN_DEV")
    repro = {"status": "NOT_APPLICABLE_DEV_ONLY"}
    if period == "FULL":
        tape_p = rq01_dir / "RQ01_10_ISSUED_TAPE_OOS.csv"
        issued = pd.read_csv(tape_p)
        chk = view.merge(issued[["country", "issue", "dec_LOGISTIC_M11"] +
                                [f"p_LOGISTIC_M11_{c}" for c in CLASSES]],
                         left_on=["country", "month"], right_on=["country", "issue"], how="inner")
        diff = max(float((chk[f"p_rq01_{c}"] - chk[f"p_LOGISTIC_M11_{c}"]).abs().max()) for c in CLASSES)
        dec_eq = int((chk["rq01_dec"] == chk["dec_LOGISTIC_M11"]).sum())
        repro = {"n_compared": int(len(chk)), "max_abs_p4_diff": diff, "n_decisions_equal": dec_eq,
                 "issued_oos_tape_sha256": _sha(tape_p),
                 "status": "PASS" if len(chk) == 420 and diff <= 1e-8 and dec_eq == len(chk) else "FAIL"}
        if repro["status"] != "PASS":
            raise ValueError("RQ01_REFIT_DOES_NOT_REPRODUCE_ISSUED_OOS_TAPE:" + json.dumps(repro))
    can = pd.read_csv(phase_file, usecols=["country", "issue_at", "state_after"])
    can["month"] = can["issue_at"].astype(str).str[:7]
    can = can.drop_duplicates(["country", "month"], keep="last")
    view = view.merge(can[["country", "month", "state_after"]].rename(columns={"state_after": "canonical_state"}),
                      on=["country", "month"], how="left")
    view["phase_source"] = np.where(view["canonical_state"].notna(), "CANONICAL", "LEGACY_FALLBACK")
    view["phase_bottom"] = [CANON_TO_BOTTOM.get(cs) if isinstance(cs, str) else LEGACY_TO_BOTTOM.get(lg, lg)
                            for cs, lg in zip(view["canonical_state"], view["phase"])]
    meta = {"gate": gate, "period": period, "theta": theta, "model": "LOGISTIC_M11_DEV_FROZEN",
            "n_features": len(cols), "n_dev_fit": int(dev.sum()), "reproduction": repro,
            "canonical_phase_file": str(phase_file), "canonical_phase_sha256": _sha(phase_file),
            "threshold_register_sha256": _sha(rq01_dir / "RQ01_08_THRESHOLD_REGISTER.json"),
            "phase_source_counts": view["phase_source"].value_counts().to_dict(),
            "dev_probability_status": "IN_SAMPLE_DEV_FIT_NOT_PIT_ISSUED"}
    return view, panel, meta
