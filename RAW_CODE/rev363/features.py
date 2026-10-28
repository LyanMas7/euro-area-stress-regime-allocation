"""REV363 feature/functions layer: PIT z-scores, votes, pairs, pools, phases.

The module implements the FINAL OUTPUT -> IDEAS -> FUNCTION/FORMULA chain for
RQ01/RQ02.  It is deliberately explicit about every threshold owner:

* source z-scores use prior-only expanding statistics (never the current or a
  future month);
* vote quantiles and pair motifs are calibrated once on N-labelled DEV rows and
  frozen into ``spec`` for OOS replay;
* pair motifs are ordered 9-state objects with exact contemporaneous maturity;
* the six Pool operators are BINARY with partial-input determination (GLOBAL NO-U contract,
  research_adapters.vote_contract): a decision the evaluable inputs cannot prove is NOT issued
  (`*_status` = NOT_EVALUATED, empty vote cell), never a third vote;
* the phase controller uses only a running maximum of past composite stress,
  never an ex-post peak.

Movement-state codes `vote_{s}` in {U, D, S, M}: U = UP, D = DOWN, S = SIDEWAY (legacy direction letters
shared with the ordered motifs), M = missing native level.  They are direction/readiness codes, not votes;
the U letter is never an unknown (audited LEGACY_UP_DIRECTION_INGEST exception).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from research_adapters.vote_contract import (NonVote, decide_and, decide_or, is_nonvote,
                                             not_evaluated)

from .config import (CLASSES, EA7, MOTIFS, PAIRS10, PAIRS21, SOURCES5, SOURCES7)

POOL_OPS = ("IND_ONLY", "JOINT_ONLY", "IND_AND_JOINT", "IND_OR_JOINT",
            "COMPLEMENT_AND_JOINT", "COMPLEMENT_OR_JOINT")

# The phase controller in `compute_phases` runs on a COMPOSITE stress intensity.  The RQ01 six-phase design lock
# prescribes the native REV352/REV353 five-state law (ENTRY/PEAK/DE_ESCALATION/RECOVERY/EXIT quorum 3/2/2/3/3,
# persistence 2) implemented in research_adapters/rq01/five_state.py, so this engine must never be reported as the
# canonical chain: every output that carries `phase` also carries this tag.
PHASE_ENGINE = "COMPOSITE_LEGACY_NOT_CANONICAL_REV353"
CANONICAL_PHASE_ENGINE = "research_adapters.rq01.five_state.FiveStateEngine"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_q(values, q, default=0.0):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.quantile(a, q)) if len(a) else float(default)


def _state_from_z(z, d6, q_hi, q_lo):
    """Motif state: U=up/high, D=down/low, S=sideways, M=missing.

    M is deliberately outside the nine ordered U/D/S motifs so a missing source
    can never be decoded as an UP vote.
    """
    if z is None or not np.isfinite(z):
        return "M"
    if z >= q_hi:
        return "U"
    if z <= q_lo and (d6 is None or d6 < 0):
        return "D"
    return "S"


# ---------------------------------------------------------------------------
# feature specification (fitted on DEV N rows, frozen for OOS)
# ---------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    vote_q_hi: dict = field(default_factory=dict)
    vote_q_lo: dict = field(default_factory=dict)
    strong_q: dict = field(default_factory=dict)
    change_q: dict = field(default_factory=dict)
    entry_q: float = 0.0
    exit_q: float = 0.0
    selected_motifs: dict = field(default_factory=dict)
    chosen_pairs: list = field(default_factory=list)
    pair_survey: dict = field(default_factory=dict)
    impute: dict = field(default_factory=dict)
    base_columns: list = field(default_factory=list)
    ind_columns: list = field(default_factory=list)
    joint_columns: list = field(default_factory=list)
    country_columns: list = field(default_factory=list)
    median: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    missing_indicator: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "vote_q_hi": self.vote_q_hi, "vote_q_lo": self.vote_q_lo,
            "strong_q": self.strong_q, "change_q": self.change_q,
            "entry_q": self.entry_q, "exit_q": self.exit_q,
            "selected_motifs": {f"{a}-{b}": m for (a, b), m in self.selected_motifs.items()},
            "chosen_pairs": [list(p) for p in self.chosen_pairs],
            "pair_survey": {f"{p[0]}-{p[1]}": v for p, v in self.pair_survey.items()},
            "impute": self.impute, "base_columns": self.base_columns,
            "ind_columns": self.ind_columns, "joint_columns": self.joint_columns,
            "country_columns": self.country_columns,
            "median": self.median, "scale": self.scale,
            "missing_indicator": self.missing_indicator,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for k in ("vote_q_hi", "vote_q_lo", "strong_q", "change_q", "impute",
                  "median", "scale", "missing_indicator"):
            setattr(obj, k, d.get(k, {}))
        obj.entry_q = d.get("entry_q", 0.0)
        obj.exit_q = d.get("exit_q", 0.0)
        obj.selected_motifs = {tuple(k.split("-")): v for k, v in d.get("selected_motifs", {}).items()}
        obj.chosen_pairs = [tuple(p) for p in d.get("chosen_pairs", [])]
        obj.pair_survey = d.get("pair_survey", {})
        for k in ("base_columns", "ind_columns", "joint_columns", "country_columns"):
            setattr(obj, k, d.get(k, []))
        return obj


# ---------------------------------------------------------------------------
# base feature construction
# ---------------------------------------------------------------------------
SOURCE_COL = {"CISS": "ciss", "CLIFS": "clifs", "DFC": "dfc", "GFC": "gfc",
              "VIX": "vix", "CG": "cg", "DSR-P": "dsr"}
ORIENT = {"CISS": 1.0, "CLIFS": 1.0, "DFC": -1.0, "GFC": 1.0, "VIX": 1.0,
          "CG": 1.0, "DSR-P": 1.0}


def _rolling_prior_z(series, window=60, min_obs=24):
    out = []
    for i, v in enumerate(series):
        if v is None or not np.isfinite(v) or i < min_obs:
            out.append(None)
            continue
        hist = [x for x in series[max(0, i - window):i] if x is not None and np.isfinite(x)]
        if len(hist) < min_obs:
            out.append(None)
            continue
        sd = float(np.std(hist, ddof=1))
        out.append(None if sd <= 1e-12 else float((v - np.mean(hist)) / sd))
    return out


def _diff(series, k):
    out = []
    for i, v in enumerate(series):
        if i - k < 0 or v is None or series[i - k] is None:
            out.append(None)
        else:
            out.append(float(v - series[i - k]))
    return out


def _pct(series, k):
    out = []
    for i, v in enumerate(series):
        if i - k < 0 or v is None or series[i - k] in (None, 0):
            out.append(None)
        else:
            out.append(float(v / series[i - k] - 1.0))
    return out


def _roll_max(series, window):
    out = []
    for i, v in enumerate(series):
        hist = [x for x in series[max(0, i - window + 1):i + 1] if x is not None and np.isfinite(x)]
        out.append(max(hist) if hist else None)
    return out


def build_raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add z, changes, drawdowns and composite intensity to a raw panel."""
    df = panel.sort_values(["country", "month"]).reset_index(drop=True).copy()
    for s in SOURCES7:
        col = SOURCE_COL[s]
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        oriented = df[col] * ORIENT[s]
        df[f"z_{s}"] = np.nan
        df[f"d3_{s}"] = np.nan
        df[f"d6_{s}"] = np.nan
        df[f"d12_{s}"] = np.nan
        df[f"pct12_{s}"] = np.nan
        df[f"dd_{s}"] = np.nan
        df[f"max12_{s}"] = np.nan
        for c in EA7:
            idx = df.index[df["country"] == c]
            ser = list(oriented.loc[idx])
            z = _rolling_prior_z(ser)
            d3 = _diff(ser, 3)
            d6 = _diff(ser, 6)
            d12 = _diff(ser, 12)
            p12 = _pct(ser, 12)
            mx = _roll_max(ser, 12)
            dd = [None if (v is None or m in (None, 0)) else float(v / m - 1.0)
                  for v, m in zip(ser, mx)]
            def _f(v):
                return np.array([np.nan if x is None else float(x) for x in v], dtype=float)
            df.loc[idx, f"z_{s}"] = _f(z)
            df.loc[idx, f"d3_{s}"] = _f(d3)
            df.loc[idx, f"d6_{s}"] = _f(d6)
            df.loc[idx, f"d12_{s}"] = _f(d12)
            df.loc[idx, f"pct12_{s}"] = _f(p12)
            df.loc[idx, f"max12_{s}"] = _f(mx)
            df.loc[idx, f"dd_{s}"] = _f(dd)
    zcols = [f"z_{s}" for s in SOURCES5]
    df["intensity"] = df[zcols].mean(axis=1, skipna=True)
    df["n_sources_available"] = df[zcols].notna().sum(axis=1)
    # country dummies
    for c in EA7:
        df[f"c_{c}"] = (df["country"] == c).astype(float)
    return df


# ---------------------------------------------------------------------------
# calibration / votes
# ---------------------------------------------------------------------------
def calibrate_spec(df: pd.DataFrame, labels: pd.Series) -> FeatureSpec:
    """Calibrate source quantiles, pair motifs and Pool choice on DEV N rows."""
    spec = FeatureSpec()
    n_mask = (labels == "N").values
    for s in SOURCES7:
        z = df.loc[n_mask, f"z_{s}"].dropna().values
        d6 = df.loc[n_mask, f"d6_{s}"].dropna().values
        spec.vote_q_hi[s] = _safe_q(z, 0.60, 0.5)
        spec.vote_q_lo[s] = _safe_q(z, 0.40, -0.5)
        spec.strong_q[s] = _safe_q(z, 0.80, 1.0)
        spec.change_q[s] = _safe_q(np.abs(d6), 0.60, 0.0)
    # composite entry/exit thresholds from N rows
    ci = df.loc[n_mask, "intensity"].dropna().values
    spec.entry_q = _safe_q(ci, 0.70, 0.5)
    spec.exit_q = _safe_q(ci, 0.40, 0.0)

    # pair motif survey: rescue Individual FAIL, then smallest new FP
    states = _source_states(df, spec)
    ind_any = _individual_any(states)
    y = (labels != "N").astype(int).values
    survey = {}
    for p in PAIRS10:
        a, b = p
        rows = []
        for motif in MOTIFS:
            mask = (states[a] == motif[0]) & (states[b] == motif[1])
            # exclude MISSING sources ('M').  REV373 excluded 'U' here, i.e. it discarded every UP state as if it
            # were unknown (GLOBAL NO-U falsifier).  Verified numerically inert on DEV (0/10 motifs change, because
            # an UP source already makes Individual PASS so no rescue is possible), but semantically corrected.
            mask &= (states[a] != "M") & (states[b] != "M")
            rescued = int(((ind_any == "FAIL") & mask & (y == 1)).sum())
            newfp = int(((ind_any == "FAIL") & mask & (y == 0)).sum())
            rows.append((rescued, newfp, MOTIFS.index(motif), motif, mask))
        best = sorted(rows, key=lambda r: (-r[0], r[1], r[2]))[0]
        spec.selected_motifs[p] = best[3]
        survey[p] = {"motif": best[3], "rescued_fn": best[0], "new_fp": best[1],
                     "all": {r[3]: {"rescued_fn": r[0], "new_fp": r[1]} for r in rows}}
    # policy pair choice: greedy, max 2 uses per source, FP budget 20
    candidates = []
    for p in PAIRS10:
        m = spec.selected_motifs[p]
        mask = (states[p[0]] == m[0]) & (states[p[1]] == m[1])
        mask &= (states[p[0]] != "M") & (states[p[1]] != "M")     # missing, not UP (see survey above)
        rescued = int(((ind_any == "FAIL") & mask & (y == 1)).sum())
        newfp = int(((ind_any == "FAIL") & mask & (y == 0)).sum())
        candidates.append((rescued, -newfp, p))
    used = {s: 0 for s in SOURCES5}
    chosen = []
    for rescued, negfp, p in sorted(candidates, reverse=True):
        if all(used[s] < 2 for s in p):
            chosen.append(p)
            for s in p:
                used[s] += 1
            if len(chosen) >= 5:
                break
    spec.chosen_pairs = chosen
    spec.pair_survey = survey
    return spec


def _source_states(df: pd.DataFrame, spec: FeatureSpec) -> dict:
    out = {}
    for s in SOURCES7:
        z = df[f"z_{s}"].values
        d6 = df[f"d6_{s}"].values
        out[s] = np.array([_state_from_z(zz, dd, spec.vote_q_hi[s], spec.vote_q_lo[s])
                           for zz, dd in zip(z, d6)], dtype=object)
    return out


def _source_presence(state):
    """UP movement -> 'PASS'; DOWN/SIDEWAY -> 'FAIL'; missing level 'M' -> NonVote (not evaluated)."""
    if state == "M":
        return not_evaluated("SOURCE_LEVEL_MISSING")
    return "PASS" if state == "U" else "FAIL"


def _individual_any(states: dict) -> np.ndarray:
    """Individual-any: PASS if any source is UP (proven); FAIL if all five are evaluated and none is UP;
    otherwise a NonVote (missing sources could still change it).  Never a third vote token."""
    n = len(next(iter(states.values())))
    out = []
    for i in range(n):
        out.append(decide_or(*[_source_presence(states[s][i]) for s in SOURCES5]))
    return np.array(out, dtype=object)


def _kand(a, b):
    return decide_and(a, b)


def _kor(a, b):
    return decide_or(a, b)


def _vote_cell(v):
    """Panel serialisation: binary vote, or None for a non-executed decision (status column says why)."""
    return None if is_nonvote(v) else v


def _status_cell(v):
    return v.status if is_nonvote(v) else "EXECUTED"


def compute_votes(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = df.copy()
    states = _source_states(out, spec)
    # Individual 5 sources: UP/DOWN/SIDEWAY (U/D/S); M is missing.
    for s in SOURCES5:
        out[f"vote_{s}"] = states[s]
        out[f"up_{s}"] = (states[s] == "U").astype(int)
        out[f"down_{s}"] = (states[s] == "D").astype(int)
        out[f"side_{s}"] = (states[s] == "S").astype(int)
        out[f"missing_{s}"] = (states[s] == "M").astype(int)
        # per-source direction summary: PASS = UP, FAIL = DOWN, SIDE = SIDEWAY; a missing level has no value
        out[f"tri_{s}"] = np.where(states[s] == "U", "PASS",
                                   np.where(states[s] == "D", "FAIL",
                                            np.where(states[s] == "M", None, "SIDE")))
        strong = (out[f"z_{s}"].values >= spec.strong_q[s]) & \
                 (np.abs(out[f"d6_{s}"].fillna(0).values) >= spec.change_q[s])
        out[f"strong_{s}"] = strong.astype(int)
    out["n_up"] = out[[f"up_{s}" for s in SOURCES5]].sum(axis=1)
    out["n_down"] = out[[f"down_{s}" for s in SOURCES5]].sum(axis=1)
    out["n_side"] = out[[f"side_{s}" for s in SOURCES5]].sum(axis=1)
    out["n_strong"] = out[[f"strong_{s}" for s in SOURCES5]].sum(axis=1)
    out["n_missing"] = out[[f"missing_{s}" for s in SOURCES5]].sum(axis=1)

    # Individual PASS: at least one strong OR at least two UP.  When missing information could still
    # change a FAIL into a PASS the decision is NOT issued (NOT_EVALUATED, empty vote); never a FAIL.
    ind = []
    for _, r in out.iterrows():
        if r["n_strong"] >= 1 or r["n_up"] >= 2:
            ind.append("PASS")
        elif r["n_missing"] > 0:
            ind.append(not_evaluated("INDIVIDUAL_NOT_DETERMINED_MISSING_SOURCE_LEVEL"))
        else:
            ind.append("FAIL")
    out["individual_vote"] = [_vote_cell(v) for v in ind]
    out["individual_vote_status"] = [_status_cell(v) for v in ind]

    # Pair motifs: ordered state of each source; exact contemporaneous lag 0.
    # `*_not_evaluated` is an ELIGIBILITY indicator (a member level is missing); it is not a vote and is
    # not a model regressor (REV373 `pair_*_unknown` unknown-vote dummies are retired from the design).
    for p in PAIRS10:
        a, b = p
        out[f"pair_{a}_{b}_state"] = [f"{x}{y}" for x, y in zip(states[a], states[b])]
        m = spec.selected_motifs.get(p, "UU")
        mask = (out[f"pair_{a}_{b}_state"] == m) & \
               (states[a] != "M") & (states[b] != "M")
        out[f"pair_{a}_{b}_pass"] = mask.astype(int)
        out[f"pair_{a}_{b}_not_evaluated"] = ((states[a] == "M") | (states[b] == "M")).astype(int)
    # RQ02 roster: 21 pairs over seven sources, fixed UU (UP-UP) research motif.
    for p in PAIRS21:
        a, b = p
        out[f"p21_{a}_{b}_state"] = [f"{x}{y}" for x, y in zip(states[a], states[b])]
        out[f"p21_{a}_{b}_pass"] = ((out[f"p21_{a}_{b}_state"] == "UU") &
                                    (states[a] != "M") & (states[b] != "M")).astype(int)
        out[f"p21_{a}_{b}_not_evaluated"] = ((states[a] == "M") | (states[b] == "M")).astype(int)

    # Pool operators over the policy-chosen RQ01 pairs (binary + partial-input determination).
    chosen = spec.chosen_pairs or []
    pool_vals = {op: [] for op in POOL_OPS}
    for pos, (i, r) in enumerate(out.iterrows()):
        iv = ind[pos]
        pair_vals = []
        for a, b in chosen:
            if int(r[f"pair_{a}_{b}_pass"]):
                pair_vals.append("PASS")
            elif int(r[f"pair_{a}_{b}_not_evaluated"]):
                pair_vals.append(not_evaluated("PAIR_MEMBER_LEVEL_MISSING"))
            else:
                pair_vals.append("FAIL")
        jv = "FAIL"
        for v in pair_vals:
            jv = _kor(jv, v)
        # COMPLEMENT_* are PAIR-indexed: OR_pair[ J_pair (AND|OR) OR_{h not in pair} I_h ].
        # A source that sits in another chosen pair still lies outside THIS pair and may
        # support it; the complement is never one global set of sources outside every pair.
        comp_and, comp_or = "FAIL", "FAIL"
        for (a, b), pv in zip(chosen, pair_vals):
            comp = "FAIL"
            for s in SOURCES5:
                if s in (a, b):
                    continue
                comp = _kor(comp, _source_presence(states[s][pos]))
            comp_and = _kor(comp_and, _kand(comp, pv))
            comp_or = _kor(comp_or, _kor(comp, pv))
        vals = {
            "IND_ONLY": iv, "JOINT_ONLY": jv,
            "IND_AND_JOINT": _kand(iv, jv), "IND_OR_JOINT": _kor(iv, jv),
            "COMPLEMENT_AND_JOINT": comp_and, "COMPLEMENT_OR_JOINT": comp_or,
        }
        for op, v in vals.items():
            pool_vals[op].append(v)
    for op in POOL_OPS:
        out[f"pool_{op}"] = [_vote_cell(v) for v in pool_vals[op]]
        out[f"pool_{op}_status"] = [_status_cell(v) for v in pool_vals[op]]
    out["pool_count"] = out[[f"pool_{op}" for op in POOL_OPS]].apply(
        lambda r: int((r == "PASS").sum()), axis=1)
    return out


# ---------------------------------------------------------------------------
# phase controller (causal, running maximum only)
# ---------------------------------------------------------------------------
def compute_phases(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = df.sort_values(["country", "month"]).copy()
    out["phase"] = "PRE"
    out["phase_start"] = 0
    out["phase_runmax"] = np.nan
    out["phase_drawdown"] = np.nan
    for c in EA7:
        idx = out.index[out["country"] == c]
        runmax = None
        state = "PRE"
        pending = None
        pending_n = 0
        prev = "PRE"
        for i in idx:
            ci = out.at[i, "intensity"]
            if ci is not None and np.isfinite(ci):
                runmax = ci if runmax is None else max(runmax, ci)
            # A signed standardized stress composite can have negative maxima;
            # a percent drawdown is mathematically undefined on a non-positive
            # anchor. Do not interpret negative/zero denominators as phases.
            dd = (None if (ci is None or not np.isfinite(ci) or runmax is None
                            or not np.isfinite(runmax) or runmax <= 0)
                  else float(ci / runmax - 1.0))
            out.at[i, "phase_runmax"] = runmax
            out.at[i, "phase_drawdown"] = dd

            def want():
                if state == "PRE":
                    return "ENTRY" if (ci is not None and ci >= spec.entry_q) else "PRE"
                if state == "ENTRY":
                    if dd is not None and dd <= -0.10:
                        return "PEAK"
                    if ci is not None and ci <= spec.exit_q:
                        return "EXIT"
                    return "ENTRY"
                if state == "PEAK":
                    if dd is not None and dd <= -0.25:
                        return "DEESC"
                    if ci is not None and ci <= spec.exit_q:
                        return "EXIT"
                    return "PEAK"
                if state == "DEESC":
                    if dd is not None and dd <= -0.50:
                        return "RECOVERY"
                    if ci is not None and ci <= spec.exit_q:
                        return "EXIT"
                    return "DEESC"
                if state == "RECOVERY":
                    if ci is not None and ci <= spec.exit_q:
                        return "EXIT"
                    return "RECOVERY"
                if state == "EXIT":
                    return "ENTRY" if (ci is not None and ci >= spec.entry_q) else "EXIT"
                return state

            w = want()
            if w != state:
                if pending == w:
                    pending_n += 1
                else:
                    pending, pending_n = w, 1
                if pending_n >= 2:
                    state, pending, pending_n = w, None, 0
            else:
                pending, pending_n = None, 0
            out.at[i, "phase"] = state
            out.at[i, "phase_start"] = int(state != prev)
            prev = state
    # Machine-readable provenance: this controller is NOT the canonical REV352/REV353 native five-state law.
    out["phase_engine"] = PHASE_ENGINE
    out.attrs["phase_engine"] = PHASE_ENGINE
    return out


# ---------------------------------------------------------------------------
# model design matrices
# ---------------------------------------------------------------------------
def _add_missing_indicators(df: pd.DataFrame, cols: list[str]) -> list[str]:
    extra = []
    for c in cols:
        if df[c].isna().any():
            name = f"miss_{c}"
            df[name] = df[c].isna().astype(float)
            extra.append(name)
    return extra


def build_design(df: pd.DataFrame, spec: FeatureSpec, *, fit: bool = False) -> dict:
    """Return BASE/IND/JOINT/COUNTRY matrices and the (possibly fitted) spec."""
    d = df.copy()
    base = []
    for s in SOURCES7:
        base += [f"z_{s}", f"d3_{s}", f"d12_{s}", f"dd_{s}"]
    base += ["intensity", "n_sources_available"] + [f"c_{c}" for c in EA7]
    # nonlinear / interaction basis (explicit, low-dimensional)
    for s in ("CISS", "CLIFS", "DFC", "VIX"):
        d[f"sq_{s}"] = d[f"z_{s}"] ** 2
        base.append(f"sq_{s}")
    for a, b in (("CISS", "CLIFS"), ("CISS", "DFC"), ("CLIFS", "DFC"),
                 ("CISS", "VIX"), ("CLIFS", "VIX")):
        d[f"ix_{a}_{b}"] = d[f"z_{a}"] * d[f"z_{b}"]
        base.append(f"ix_{a}_{b}")
    # country-varying coefficients on the four core state variables
    country_cols = []
    for c in EA7:
        for s in ("CISS", "CLIFS", "DFC", "VIX", "CG", "DSR-P"):
            name = f"cv_{c}_{s}"
            d[name] = d[f"c_{c}"] * d[f"z_{s}"]
            country_cols.append(name)
    # IND block: movement-direction indicators (legacy letter U = UP, D = DOWN, S = SIDEWAY)
    ind = []
    for s in SOURCES5:
        for state, label in (("U", "UP"), ("D", "DOWN"), ("S", "SIDEWAY")):
            name = f"ind_{s}_{label}"
            d[name] = (d[f"vote_{s}"] == state).astype(float)
            ind.append(name)
        ind.append(f"strong_{s}")
    ind += ["n_up", "n_down", "n_strong", "n_missing"]
    # JOINT block.  GLOBAL NO-U: no unknown-vote dummy regressors and no -1 third-class encoding.  Each Pool
    # enters as the binary indicator of an ISSUED PASS, 1[pool == PASS]; a non-executed pool issued no PASS
    # (its readiness is carried by the data-availability count `n_sources_available` in BASE, not by a vote).
    joint = []
    for p in PAIRS10:
        a, b = p
        joint.append(f"pair_{a}_{b}_pass")
    joint += ["pool_count"]
    for op in POOL_OPS:
        d[f"pool_{op}_issued_pass"] = (d[f"pool_{op}"] == "PASS").astype(float)
    joint += [f"pool_{op}_issued_pass" for op in POOL_OPS]
    # dedupe preserve order
    joint = list(dict.fromkeys(joint))
    all_cols = list(dict.fromkeys(base + ind + joint + country_cols))

    if fit:
        spec.base_columns = base
        spec.ind_columns = ind
        spec.joint_columns = joint
        spec.country_columns = country_cols
        for c in all_cols:
            vals = d[c].replace([np.inf, -np.inf], np.nan)
            med = float(vals.median()) if vals.notna().any() else 0.0
            spec.median[c] = med
            filled = vals.fillna(med)
            sd = float(filled.std(ddof=0))
            spec.scale[c] = sd if sd > 1e-12 else 1.0
            spec.missing_indicator[c] = int(vals.isna().any())
    for c in all_cols:
        if c not in spec.median:
            spec.median[c] = 0.0
            spec.scale[c] = 1.0
    X = {}
    for name, cols in (("BASE", base), ("IND", ind), ("JOINT", joint),
                       ("COUNTRY", country_cols)):
        arr = np.empty((len(d), len(cols)), dtype=float)
        for j, c in enumerate(cols):
            v = d[c].replace([np.inf, -np.inf], np.nan).fillna(spec.median.get(c, 0.0))
            arr[:, j] = (v.values - spec.median.get(c, 0.0)) / spec.scale.get(c, 1.0)
        X[name] = arr
    return {"X": X, "df": d, "spec": spec, "columns": {
        "BASE": base, "IND": ind, "JOINT": joint, "COUNTRY": country_cols}}
