"""REV363 configuration: frozen scope, source roster, versions, incumbent controls.

Everything here is research infrastructure. No value is an owner-approved
production threshold. The REV362 incumbent thresholds are preserved verbatim
as ``INCUMBENT_*`` constants and are never overwritten; all new controls are
versioned ``REV363_*`` challengers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

EA7 = ("BE", "DE", "ES", "FI", "FR", "IT", "NL")
SOURCES5 = ("CISS", "CLIFS", "DFC", "GFC", "VIX")
SOURCES7 = ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")
CLASSES = ("N", "D", "G", "BOTH")
REGIMES = ("N", "D", "G", "BOTH")
MOTIFS = tuple(a + b for a in "UDS" for b in "UDS")
PAIRS10 = tuple((SOURCES5[i], SOURCES5[j]) for i in range(5) for j in range(i + 1, 5))
PAIRS21 = tuple((SOURCES7[i], SOURCES7[j]) for i in range(7) for j in range(i + 1, 7))
CANONICAL_LONG_PAIR = ("CG", "DSR-P")
HORIZONS = ("H15", "H12", "H10", "H8", "H6")
HORIZON_MONTHS = {"H15": 45, "H12": 36, "H10": 30, "H8": 24, "H6": 18}
EXIT_HORIZONS = (3, 6, 12, 18)

DEV_START, DEV_END = "2005-01", "2019-12"
DEV_FIT_START = "2005-07"          # 2005-01..06 is one-sided warm-up only
OOS_START, OOS_END = "2020-01", "2024-12"
WARMUP_START = "2004-01"
PANEL_START = "2004-01"

# Incumbent REV362 values. These are preserved, not replaced.
INCUMBENT_RQ01_M11_THRESHOLD = 0.4162494835709779
INCUMBENT_RQ01_M11_L = 0.41327537939643133
INCUMBENT_RQ02_THRESHOLDS = {"H15": 0.6866232386128425, "H12": 0.5226748117916373,
                             "H10": 0.5226203639149063, "H8": 0.45775602016985256,
                             "H6": 0.4681253349955738}
INCUMBENT_RQ03_FEE = None          # no owner cost schedule was supplied
INCUMBENT_RQ01_STATUS = "EXPLORATORY_CHALLENGER_NOT_CERTIFIED"

# REV363 challenger identities
VERSION = "REV363"
MODEL_VERSION = "REV363_VARYING_COEFFICIENT_TWO_HEAD"
FEATURE_VERSION = "REV363_PIT_ASOF_MONTHLY_V1"
FREEZE_VERSION = "REV363_DEV_FREEZE_V1"
OOS_VERSION = "REV363_OOS_ONE_SHOT_V1"
FEE_RESEARCH = 0.0010              # 10 bp one-way research assumption, applied once per turnover
CASH_ASSET = "C1"
RISKY_ASSETS = ("E1", "E2", "E3", "B1", "B2")
ALL_ASSETS = ("E1", "E2", "E3", "B1", "B2", "C1")
BENCHMARKS = ("EW", "IVOL", "MVP", "ERC", "MaxDiv", "POLICY")

# PIT research release rules. Owner status is explicitly NOT_CERTIFIED.
RELEASE_RULES = {
    "CISS": {"lag_days": 7, "basis": "weekly publication + 7 calendar days"},
    "CLIFS": {"lag_days": 0, "basis": "supplied AvailabilityDate column"},
    "DFC": {"lag_days": 135, "basis": "quarter end + 135 calendar days (REV362 W01 research policy)"},
    "GFC": {"lag_days": 30, "basis": "month end + 30 calendar days (REV362 W01 research policy)"},
    "VIX": {"lag_days": 0, "basis": "same-day close, end-of-day UTC"},
    "CG": {"lag_days": 90, "basis": "quarter end + 90 calendar days (research assumption)"},
    "DSR-P": {"lag_days": 90, "basis": "quarter end + 90 calendar days (research assumption)"},
}
PIT_STATUS = "RESEARCH_ASOF_NOT_OFFICIAL_ISSUER_PIT"

# Model hierarchy: block names per model. M11 is the REV363 challenger with a
# country-varying-coefficient basis and a two-head (stress + subtype) fit.
MODEL_BLOCKS = {
    "M00": ("BASE",),
    "M01": ("BASE", "IND"),
    "M10": ("BASE", "JOINT"),
    "M11": ("BASE", "IND", "JOINT", "COUNTRY"),
}
MODEL_ROLES = {
    "M00": "BASE_ONLY",
    "M01": "BASE_PLUS_INDIVIDUAL",
    "M10": "BASE_PLUS_JOINT",
    "M11": "BASE_PLUS_INDIVIDUAL_PLUS_JOINT_PLUS_COUNTRY_VARYING",
}

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT.parent
FREEZE_DIR = ROOT / "frozen"
FREEZE_FILE = FREEZE_DIR / "REV363_DEV_FREEZE.json"
RESULTS_DIR = RAW_ROOT.parent / "RESULTS"


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
