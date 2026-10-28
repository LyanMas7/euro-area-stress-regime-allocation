"""RQ01 W01 source-native movement + exact-lag joint + six pool truth tables.

This is an executable *research* geometry, not a reconstructed Rev55/Rev64 owner
threshold.  GLOBAL NO-U: votes are PASS/FAIL; a non-computable vote is a non-vote record
(`vote` empty, `vote_status` NOT_EVALUATED + `vote_reason`).  Movement codes UP/DOWN/SIDEWAY/
UNAVAILABLE are direction/readiness attributes, not votes.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Optional
import numpy as np
import pandas as pd

from research_adapters.vote_contract import NonVote, ThirdVoteError, decide_and, decide_or, is_nonvote

SOURCES5 = ("CISS", "CLIFS", "DFC", "GFC", "VIX")
PAIRS10 = tuple(combinations(SOURCES5, 2))
POOL6 = ("IND_ONLY", "JOINT_ONLY", "IND_AND_JOINT", "IND_OR_JOINT",
         "COMPLEMENT_AND_JOINT", "COMPLEMENT_OR_JOINT")


@dataclass(frozen=True)
class NativeThreshold:
    source: str
    native_frequency: str
    unit: str
    strong_abs: float
    weak_abs: float
    min_count: int
    window: int
    direction: str = "POSITIVE"

    def __post_init__(self):
        if self.source not in SOURCES5 or self.window <= 0 or self.min_count < 1:
            raise ValueError("BAD_SOURCE_OR_COUNT")
        if self.strong_abs < self.weak_abs or self.weak_abs < 0:
            raise ValueError("BAD_STRONG_WEAK_THRESHOLD")


def kl_and(a, b):
    """Binary AND with partial-input determination (vote_contract); 'U' is rejected."""
    return decide_and(a, b)


def kl_or(a, b):
    """Binary OR with partial-input determination (vote_contract); 'U' is rejected."""
    return decide_or(a, b)


def pools(individual, joint, complement):
    """Six pools over PASS/FAIL votes or NonVote components; outputs are PASS/FAIL or NonVote."""
    for x in (individual, joint, complement):
        if x == "U":
            raise ThirdVoteError("LEGACY_U_POOL_INPUT_REJECTED")
        if not (is_nonvote(x) or x in ("PASS", "FAIL")):
            raise ValueError("BAD_BINARY_VOTE_OR_NONVOTE")
    return dict(zip(POOL6, (individual, joint, kl_and(individual,joint),
                            kl_or(individual,joint), kl_and(complement,joint),
                            kl_or(complement,joint))))


def binary_pool_truth_table() -> list:
    """Six pools over the 2^3 binary (Individual, Joint, Complement) votes."""
    rows=[]
    for i in ("PASS","FAIL"):
        for j in ("PASS","FAIL"):
            for comp in ("PASS","FAIL"):
                rows.append({"Individual":i,"Joint":j,"Complement":comp,**pools(i,j,comp)})
    return rows


def pool_determination_table() -> list:
    """Partial-input proof: inputs PASS/FAIL/NE (not evaluated); outputs vote or the not-determined status."""
    from research_adapters.vote_contract import not_evaluated
    conv={"PASS":"PASS","FAIL":"FAIL","NE":not_evaluated("COMPONENT_NOT_EVALUATED")}
    rows=[]
    for i in conv:
        for j in conv:
            for comp in conv:
                out=pools(conv[i],conv[j],conv[comp])
                rows.append({"Individual":i,"Joint":j,"Complement":comp,
                             **{k:(v.reason if is_nonvote(v) else v) for k,v in out.items()}})
    return rows


def native_motion_table(panel: pd.DataFrame, source: str, *, issue_col="month") -> pd.DataFrame:
    """A carried quarterly level never generates a second native increment.

    Source observed period and release cutoff both must be recorded. Operates
    per-country in issue order; records a new observation only when the
    distinct native timestamp changes. Does not infer newness from value change.
    """
    if source not in SOURCES5: raise ValueError("UNREGISTERED_SOURCE")
    required = {"country", issue_col, source.lower(), f"observed_at_{source}",f"available_at_{source}"}
    if not required.issubset(panel.columns):
        raise ValueError("MISSING_NATIVE_TIMESTAMP_OR_SOURCE:" + ",".join(sorted(required-set(panel.columns))))
    df = panel.sort_values(["country",issue_col]).copy()
    rec = []
    for c, sub in df.groupby("country", sort=True):
        prev_stamp, prev_value = None, None
        for _,r in sub.iterrows():
            stamp = r[f"observed_at_{source}"]
            avail = r[f"available_at_{source}"]
            v = r[source.lower()]
            issue = pd.Period(str(r[issue_col]),freq="M").to_timestamp(how="end").date()
            ready = stamp is not None and pd.notna(stamp) and avail is not None and pd.notna(avail) and pd.notna(v)
            if ready and pd.Timestamp(avail).date() > issue:
                raise ValueError("LOOKAHEAD_RELEASE_AFTER_ISSUE")
            fresh = bool(ready and stamp != prev_stamp)
            delta = None; pct = None; movement = "UNAVAILABLE"
            if fresh:
                if prev_value is not None and pd.notna(prev_value):
                    delta = float(v-prev_value)
                    pct = float(v/prev_value-1) if prev_value != 0 else None
                    movement = "UP" if delta>0 else ("DOWN" if delta<0 else "SIDEWAY")
                prev_stamp, prev_value = stamp, float(v)
            rec.append({"country":c,"issue":r[issue_col],"source":source,
                        "native_observed_at":stamp,"available_at":avail,
                        "native_fresh":fresh,"value":v,"terminal_abs":delta,
                        "terminal_pct":pct,"movement":movement,
                        "source_pit":("RESEARCH_ASOF_ISSUER_UNVERIFIED" if ready else "UNAVAILABLE")})
    return pd.DataFrame(rec)


def individual_threshold_vote(native: pd.DataFrame, rule: NativeThreshold) -> pd.DataFrame:
    """Strong OR (weak AND sufficient count) on *new native moves* only."""
    t = native.sort_values(["country", "issue"]).copy()
    out=[]
    sign = 1 if rule.direction=="POSITIVE" else -1
    for _, sub in t.groupby("country", sort=True):
        history=[]
        for _,r in sub.iterrows():
            if bool(r["native_fresh"]) and pd.notna(r["terminal_abs"]):
                history.append(float(r["terminal_abs"])*sign)
            d = history[-rule.window:]
            nup=sum(x>=rule.weak_abs and x>0 for x in d)
            strong=any(x>=rule.strong_abs and x>0 for x in d)
            if not d:
                vote,status,why=None,"NOT_EVALUATED","NO_NATIVE_MOVE_IN_WINDOW"
            else:
                vote,status,why=("PASS" if strong or (nup>=rule.min_count) else "FAIL"),"EXECUTED",None
            out.append({"country":r["country"],"issue":r["issue"],"source":rule.source,
                        "native_moves_available":len(d),"up_count":nup,
                        "strong":strong,"vote":vote,"vote_status":status,"vote_reason":why,
                        "threshold_source":rule.source,
                        "threshold_unit":rule.unit})
    return pd.DataFrame(out)


def exact_native_joint(left: pd.DataFrame, right: pd.DataFrame, *,
                       left_min:float, right_min:float, max_lag_days:int=0) -> pd.DataFrame:
    """True two-terminal Gate1 & Gate2, never a same-month carried proxy.

    Both terminals must be fresh and available at issue, and underlying
    observation dates must align within approved exact max_lag_days.
    """
    if max_lag_days < 0: raise ValueError("NEGATIVE_LAG")
    keys=["country","issue"]
    a=left.rename(columns={x:"a_"+x for x in left if x not in keys})
    b=right.rename(columns={x:"b_"+x for x in right if x not in keys})
    t=a.merge(b,on=keys,how="inner",validate="one_to_one")
    rows=[]
    for _,r in t.iterrows():
        valid=(bool(r["a_native_fresh"]) and bool(r["b_native_fresh"])
               and pd.notna(r["a_terminal_abs"]) and pd.notna(r["b_terminal_abs"]))
        lag=None
        if valid:
            lag=abs((pd.Timestamp(r["a_native_observed_at"])-pd.Timestamp(r["b_native_observed_at"])).days)
            valid=lag<=max_lag_days
        if not valid:
            vote,status,why=None,"NOT_EVALUATED","NO_FRESH_MATCHED_NATIVE_TERMINALS_WITHIN_LAG"
        else:
            vote=("PASS" if float(r["a_terminal_abs"])>=left_min and float(r["b_terminal_abs"])>=right_min
                  else "FAIL")
            status,why="EXECUTED",None
        rows.append({"country":r["country"],"issue":r["issue"],
                     "pair":str(r["a_source"])+"|"+str(r["b_source"]),
                     "observed_lag_days":lag,"required_max_lag_days":max_lag_days,
                     "gate1": (float(r["a_terminal_abs"])>=left_min if valid else None),
                     "gate2": (float(r["b_terminal_abs"])>=right_min if valid else None),
                     "vote":vote,"vote_status":status,"vote_reason":why})
    return pd.DataFrame(rows)
