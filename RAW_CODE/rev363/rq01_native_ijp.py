"""RQ01 native Individual -> Joint -> Pool research tape.

Implements the universal kernel contracts in
`Individual_Joint_Pool áp dụng RQ01, RQ02.md` (B1-B4, C2-C4, D2-D4) and the
EXACT(0) RQ01 pair roster in `RQ01_xác định pair hợp lý chạy joint.md` on the
locked native source files.

Research status:
* monthly issue grid with one as-of snapshot per source per month;
* thresholds are label-free trailing adjacent-delta quantiles (Q75/Q35),
  frozen per issue before use;
* no owner-approved source cuts exist, so every vote is research-only;
* Sideway Joint Gate1 is not implemented by the shared kernel and is emitted
  as NOT_EVALUATED, never as a fabricated FAIL.
* GLOBAL NO-U: every tape row is two-layer -- `status` EVALUATED with a binary vote/decision, or
  NOT_EVALUATED with an empty vote cell and a non-null reason code.  No third vote class exists.
"""
from __future__ import annotations

import calendar
import json
import math
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research_adapters.ijp_kernel import (EligibilityError, NativeLevel,
                                          individual as kernel_individual,
                                          joint as kernel_joint,
                                          pool as kernel_pool)
from research_adapters.rq01.motif_issue_packet import (LAG_OWNER, build_issue_packet,
                                                       research_pair_thresholds,
                                                       research_state_and_terminal)
from research_adapters.rq01.pairs import LagPolicy
from .config import EA7, PAIRS10, RELEASE_RULES, SOURCES5
from .data import (add_months, construct_dfc_country, find_dataset_root,
                   month_end, read_ciss, read_clifs, read_dfc_inputs, read_gfc,
                   read_vix)

PROFILE_ID = "RQ01_NATIVE_IJP_RESEARCH_V1"
WINDOW_MONTHS = 12
PRIOR_MONTHS = 24
DIRECTIONAL = (("UU", (1, 1)), ("UD", (1, -1)), ("DU", (-1, 1)), ("DD", (-1, -1)))
SIDEWAY = ("US", "SU", "DS", "SD", "SS")
MOTIFS = tuple(a + b for a in "UDS" for b in "UDS")
FREQUENCY = {"CISS": "W", "CLIFS": "M", "DFC": "Q", "GFC": "M", "VIX": "D"}
SIGNS = {"CISS": 1, "CLIFS": 1, "DFC": -1, "GFC": 1, "VIX": 1}
NATIVE_FILE = {"CISS": "ciss_total_weekly.csv", "CLIFS": "CLIFS_EA8_Monthly.csv",
               "DFC": "DFC_Construction.xlsx", "GFC": "GFC_Price_Construction.xlsx",
               "VIX": "VIX_Daily.csv"}


def _iso(d) -> str:
    if isinstance(d, str):
        return d[:10]
    return pd.Timestamp(d).date().isoformat()


def _month_end_iso(month: str) -> str:
    return month_end(int(month[:4]), int(month[5:7])).isoformat()


def _add_days(d, days: int) -> str:
    return (pd.Timestamp(d).date() + timedelta(days=int(days))).isoformat()


def _period(source: str, d) -> str:
    t = pd.Timestamp(d)
    if source == "DFC":
        return f"{t.year}-Q{t.quarter}"
    if source in ("CLIFS", "GFC"):
        return t.strftime("%Y-%m")
    return t.date().isoformat()


def _append(rows, country, source, d, value, available):
    if value is None or not np.isfinite(float(value)):
        return
    rows.append({
        "country": country, "source": source,
        "native_period": _period(source, d),
        "period_end": _iso(d), "available_at": _iso(available),
        "value": float(value), "native_frequency": FREQUENCY[source],
        "source_file": NATIVE_FILE[source],
        "clock_quality": "RESEARCH_RELEASE_CLOCK_NOT_ISSUER_CERTIFIED",
    })


def native_frame(dataset_root, *, data_end: str = "2019-12-31") -> pd.DataFrame:
    """Build the native EA7 frame for the five RQ01 sources through `data_end`."""
    root = find_dataset_root(dataset_root)
    end = _iso(data_end)
    rows: list[dict] = []
    ciss_d, ciss_v = read_ciss(root)
    vix_d, vix_v = read_vix(root)
    gfc_d, gfc_v = read_gfc(root)
    for c in EA7:
        for d, v in zip(ciss_d, ciss_v):
            if _iso(d) <= end:
                _append(rows, c, "CISS", d, v, _add_days(d, RELEASE_RULES["CISS"]["lag_days"]))
        for d, v in zip(vix_d, vix_v):
            if _iso(d) <= end:
                _append(rows, c, "VIX", d, v, _add_days(d, RELEASE_RULES["VIX"]["lag_days"]))
        for d, v in zip(gfc_d, gfc_v):
            if _iso(d) <= end:
                _append(rows, c, "GFC", d, v, _add_days(d, RELEASE_RULES["GFC"]["lag_days"]))
    clifs = read_clifs(root, countries=EA7)
    for c in EA7:
        ds, vs, av = clifs[c]
        for d, v, a in zip(ds, vs, av):
            if _iso(d) <= end:
                _append(rows, c, "CLIFS", d, v, a)
    tc, rpp, cpi = read_dfc_inputs(root)
    last_q = f"{end[:4]}-Q{(int(end[5:7]) - 1) // 3 + 1}"
    for c in EA7:
        vals = construct_dfc_country(tc, rpp, cpi, c, last_quarter=last_q)
        for q, v in vals.items():
            qend = pd.Period(q, freq="Q").end_time.date()
            if _iso(qend) <= end:
                _append(rows, c, "DFC", qend, v,
                        _add_days(qend, RELEASE_RULES["DFC"]["lag_days"]))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("NO_NATIVE_OBSERVATIONS")
    frame = frame[frame["available_at"] <= end].copy()
    dup = frame.duplicated(["country", "source", "native_period"])
    if dup.any():
        raise ValueError(f"DUPLICATE_NATIVE_PERIOD:{int(dup.sum())}")
    return frame.sort_values(["country", "source", "period_end", "native_period"],
                             kind="stable").reset_index(drop=True)


def _levels_by_country_source(frame: pd.DataFrame) -> dict[tuple[str, str], list[NativeLevel]]:
    out: dict[tuple[str, str], list[NativeLevel]] = {}
    for (country, source), sub in frame.groupby(["country", "source"], sort=False):
        out[(country, source)] = [
            NativeLevel(period_id=str(r.native_period), period_end=str(r.period_end),
                        available_at=str(r.available_at), value=float(r.value))
            for r in sub.itertuples(index=False)
        ]
    return out


def _monthly_series(rows: list[NativeLevel], issue: str,
                    window_months: int = WINDOW_MONTHS) -> list[tuple[str, NativeLevel]]:
    """One latest as-of native snapshot per month, then one level per native period."""
    months = [add_months(issue, -k) for k in range(window_months - 1, -1, -1)]
    chosen: dict[str, tuple[str, NativeLevel]] = {}
    for m in months:
        cut = _month_end_iso(m)
        eligible = [r for r in rows if r.period_end <= cut and r.available_at <= cut]
        if not eligible:
            continue
        latest = max(eligible, key=lambda r: (r.period_end, r.available_at, r.period_id))
        chosen[latest.period_id] = (m, latest)
    return [chosen[k] for k in sorted(chosen, key=lambda z: (chosen[z][1].period_end,
                                                             chosen[z][1].period_id))]


def _prior_thresholds(rows: list[NativeLevel], issue: str) -> dict | None:
    """Label-free adjacent-delta Q75/Q35 from the 24 months before the active window."""
    window_start = _month_end_iso(add_months(issue, -(WINDOW_MONTHS - 1)))
    cut = _month_end_iso(issue)
    prior = [r for r in rows if r.period_end < window_start and r.available_at <= cut]
    by_id = {}
    for r in sorted(prior, key=lambda z: (z.period_end, z.available_at, z.period_id)):
        by_id[r.period_id] = r
    vals = [by_id[k].value for k in sorted(by_id, key=lambda z: (by_id[z].period_end, z))]
    deltas = np.asarray([abs(b - a) for a, b in zip(vals[:-1], vals[1:])], float)
    if len(deltas) < 4:
        return None
    hi = float(np.quantile(deltas, 0.75))
    lo = float(np.quantile(deltas, 0.35))
    if not np.isfinite(hi) or hi <= 0:
        return None
    return {"hi": hi, "lo": max(0.0, min(lo, hi)),
            "n_prior_deltas": int(len(deltas))}


def _individual_row(country: str, issue: str, source: str,
                    series: list[tuple[str, NativeLevel]], prof: dict | None) -> dict:
    base = {"country": country, "issue": issue, "source": source,
            "native_frequency": FREQUENCY[source], "profile_id": PROFILE_ID,
            "n_levels": len(series), "status": "NOT_EVALUATED", "vote": None,
            "reason": None}
    if prof is None:
        base["reason"] = "INSUFFICIENT_PRIOR_DELTAS"
        return base
    if len(series) < 2:
        base["reason"] = "N_LT_2_NATIVE_LEVELS"
        return base
    levels = [r.value for _, r in series]
    n = len(levels)
    m = n * (n - 1) // 2
    b = n - 1
    try:
        res = kernel_individual(levels, SIGNS[source], theta_hi=prof["hi"],
                                theta_lo=prof["lo"], min_bases=min(2, b),
                                count_k=max(1, math.ceil(0.5 * m)))
    except EligibilityError as exc:
        base["reason"] = str(exc)
        return base
    base.update({
        "status": "EVALUATED", "vote": int(res["vote"]),
        "M": int(m), "terminal_bases": int(b), "up": int(res["up"]),
        "down": int(res["down"]), "flat": int(res["flat"]),
        "strong": int(res["strong"]), "weak": int(res["weak"]),
        "count": int(res["count"]), "B_hi": int(res["B_hi"]),
        "B_lo": int(res["B_lo"]), "signed_count": int(res["signed_count"]),
        "terminal_mean": float(np.mean(res["terminal"])),
        "terminal_positive": int(sum(d > 0 for d in res["terminal"])),
        "theta_hi": prof["hi"], "theta_lo": prof["lo"],
        "min_bases": int(min(2, b)),
        "count_k": int(max(1, math.ceil(0.5 * m))),
        "n_prior_deltas": int(prof["n_prior_deltas"]),
        "native_periods": ";".join(r.period_id for _, r in series),
    })
    return base


def _matched(series_a: list[tuple[str, NativeLevel]],
             series_b: list[tuple[str, NativeLevel]], freq_a: str, freq_b: str):
    a = {m: r for m, r in series_a}
    b = {m: r for m, r in series_b}
    months = sorted(set(a) & set(b))
    matched, used = [], []
    last_master = None
    prev_a, prev_b = None, None
    for m in months:
        ra, rb = a[m], b[m]
        if "Q" in (freq_a, freq_b):
            master = ra if freq_a == "Q" else rb
            if master.period_id == last_master:
                continue
            last_master = master.period_id
        else:
            if ra.period_id == prev_a and rb.period_id == prev_b:
                continue
        matched.append((ra.value, rb.value))
        used.append((m, ra, rb))
        prev_a, prev_b = ra.period_id, rb.period_id
    return matched, used


def _joint_rows(country: str, issue: str, a: str, b: str,
                series_a: list[tuple[str, NativeLevel]],
                series_b: list[tuple[str, NativeLevel]],
                prof_a: dict | None, prof_b: dict | None) -> list[dict]:
    matched, used = _matched(series_a, series_b, FREQUENCY[a], FREQUENCY[b])
    base = {"country": country, "issue": issue, "pair": f"{a}-{b}",
            "profile_id": PROFILE_ID, "n_matched": len(matched),
            "native_periods_a": ";".join(r.period_id for _, r, _ in used),
            "native_periods_b": ";".join(r.period_id for _, _, r in used)}
    rows = []
    if prof_a is None or prof_b is None:
        reason = "INSUFFICIENT_PRIOR_DELTAS"
    elif len(matched) < 2:
        reason = "N_MATCHED_LT_2"
    else:
        reason = None
    for motif in MOTIFS:
        row = {**base, "motif": motif, "chosen": motif == "UU",
               "status": "NOT_EVALUATED", "vote": None,
               "gate1": None, "gate2": None, "paired_count": None}
        if motif in SIDEWAY:
            row["reason"] = "SIDEWAY_GATE1_NOT_IMPLEMENTED_IN_SHARED_KERNEL"
            rows.append(row)
            continue
        if reason is not None:
            row["reason"] = reason
            rows.append(row)
            continue
        sign_a = SIGNS[a] * dict(DIRECTIONAL)[motif][0]
        sign_b = SIGNS[b] * dict(DIRECTIONAL)[motif][1]
        n = len(matched)
        m_pairs = n * (n - 1) // 2
        try:
            res = kernel_joint(matched, (sign_a, sign_b),
                               high_a=prof_a["hi"], high_b=prof_b["hi"],
                               low_a=prof_a["lo"], low_b=prof_b["lo"],
                               gate1_a=prof_a["lo"], gate1_b=prof_b["lo"],
                               min_bases=min(2, n - 1),
                               count_k=max(1, math.ceil(0.5 * m_pairs)),
                               timing_valid=True)
        except EligibilityError as exc:
            row["reason"] = str(exc)
            rows.append(row)
            continue
        row.update({"status": "EVALUATED", "vote": int(res["vote"]),
                    "gate1": int(res["gate1"]), "gate2": int(res["gate2"]),
                    "paired_count": int(res["paired_count"]),
                    "paired_hi": int(res["paired_hi"]),
                    "paired_lo": int(res["paired_lo"]),
                    "theta_hi_a": prof_a["hi"], "theta_hi_b": prof_b["hi"],
                    "theta_lo_a": prof_a["lo"], "theta_lo_b": prof_b["lo"],
                    "reason": None})
        rows.append(row)
    return rows


def _pair_sources(key) -> list:
    """Registered source pair from a tuple or a 'A-B' packet key (never characters)."""
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return [str(x) for x in key]
    raw = str(key).strip().strip("()[]'\" ")
    for sep in ("-", ","):
        parts = [p.strip().strip("'\"") for p in raw.split(sep) if p.strip()]
        if len(parts) == 2:
            return parts
    return []


def pool_evidence_counts(individual_votes: dict, packet: dict) -> dict:
    """Separate source counts from pair/config counts in a Pool evidence row.

    A packet can select at most ten pair configurations; that number is not the
    number of evaluable Individual sources (at most five).  Keeping the fields
    disjoint prevents a pair count from being read as source coverage.
    """
    selected = packet.get("selected_config_ids", {})
    consumed = packet.get("consumed_config_ids", {})
    research = packet.get("research_config_ids", {})
    selected_sources = {s for key in selected for s in _pair_sources(key)}
    consumed_sources = {s for key in consumed for s in _pair_sources(key)}
    # only genuine binary votes count as evaluable sources (None / NaN is not a vote)
    n_ind = sum(1 for v in individual_votes.values() if v in (0, 1) and v is not None)
    sel_ids = set(selected.values())
    res_ids = set(research.values())
    joint_rows = packet.get("joint_rows", [])
    has_config_ids = any(row.get("config_id") for row in joint_rows)
    if has_config_ids:
        n_eval_sel = sum(1 for row in joint_rows
                         if row.get("config_id") in sel_ids
                         and row.get("status") in ("PASS", "FAIL"))
        n_eval_res = sum(1 for row in joint_rows
                         if row.get("config_id") in res_ids
                         and row.get("status") in ("PASS", "FAIL"))
    else:
        # legacy packet shape without per-config identity
        n_eval_sel = sum(1 for row in joint_rows
                         if row.get("motif") == packet.get("selected_motif", "UU")
                         and row.get("status") in ("PASS", "FAIL"))
        n_eval_res = n_eval_sel
    out = {
        "n_source_total": 5,
        "n_individual_evaluable": int(n_ind),
        "n_joint_selected_configs": int(packet.get("n_selected", len(selected))),
        "n_joint_evaluable_configs": int(n_eval_sel),
        "n_joint_consumed_configs": int(packet.get("n_consumed", len(consumed))),
        "n_research_feature_configs": int(packet.get("n_research_consumed", len(research))),
        "n_research_feature_evaluable": int(n_eval_res),
        "n_unique_sources_in_selected_joint": int(len(selected_sources)),
        "n_unique_sources_in_consumed_joint": int(len(consumed_sources)),
        "n_not_evaluated_sources": int(5 - n_ind),
    }
    if not (0 <= out["n_individual_evaluable"] <= 5
            and out["n_individual_evaluable"] <= out["n_source_total"]):
        raise AssertionError("INDIVIDUAL_EVALUABLE_OUTSIDE_SOURCE_DOMAIN")
    if not (0 <= out["n_unique_sources_in_selected_joint"] <= 5
            and 0 <= out["n_unique_sources_in_consumed_joint"] <= 5):
        raise AssertionError("PAIR_SOURCE_COUNT_OUTSIDE_SOURCE_DOMAIN")
    if not (0 <= out["n_joint_evaluable_configs"] <= out["n_joint_selected_configs"]
            and 0 <= out["n_joint_consumed_configs"] <= out["n_joint_selected_configs"]):
        raise AssertionError("JOINT_CONFIG_COUNTER_ORDER_VIOLATED")
    if out["n_joint_consumed_configs"] != out["n_joint_selected_configs"]:
        raise AssertionError("SELECTED_CONSUMED_CONFIG_COUNT_MISMATCH")
    return out


def build_native_tape(dataset_root, *, start: str = "2005-07",
                      end: str = "2019-12", selected_configs: dict | None = None,
                      research_configs: dict | None = None,
                      selected_pool: str | None = None) -> dict:
    """Build per-issue native Individual/Joint/Pool research rows.

    `selected_configs` is the frozen registry decision (possibly empty).  The
    Pool consumes exactly those configurations; when it is empty the issued Pool
    falls back to IND_ONLY and reports the fallback.  `research_configs` feeds
    the declared diagnostic packet-feature block only and is never called
    selected.
    """
    frame = native_frame(dataset_root, data_end=_month_end_iso(end))
    levels = _levels_by_country_source(frame)
    ind_rows, joint_rows, pool_rows = [], [], []
    legacy_joint_rows, legacy_pool_rows = [], []
    parity_rows, ne_reason_rows = [], []
    series_cache: dict[tuple[str, str, str], list] = {}
    prof_cache: dict[tuple[str, str, str], dict | None] = {}
    months = [add_months(start, k) for k in range(
        (int(end[:4]) - int(start[:4])) * 12 + int(end[5:7]) - int(start[5:7]) + 1)]
    for country in EA7:
        for issue in months:
            ind_issue = {}
            for source in SOURCES5:
                key = (country, source, issue)
                rows = levels.get((country, source), [])
                if key not in series_cache:
                    series_cache[key] = _monthly_series(rows, issue)
                    prof_cache[key] = _prior_thresholds(rows, issue)
                series = series_cache[key]
                prof = prof_cache[key]
                row = _individual_row(country, issue, source, series, prof)
                ind_rows.append(row)
                ind_issue[source] = row
            # Legacy sidecar rows retained for audit and rollback.
            legacy_joint_issue = {}
            for a, b in PAIRS10:
                sa = series_cache[(country, a, issue)]
                sb = series_cache[(country, b, issue)]
                pa = prof_cache[(country, a, issue)]
                pb = prof_cache[(country, b, issue)]
                for row in _joint_rows(country, issue, a, b, sa, sb, pa, pb):
                    legacy_joint_rows.append(row)
                    if row["motif"] == "UU":
                        legacy_joint_issue[(a, b)] = row
            legacy_individuals = {s: r["vote"] for s, r in ind_issue.items()
                                  if r["status"] == "EVALUATED"}
            legacy_joint_votes = {p: r["vote"] for p, r in legacy_joint_issue.items()
                                  if r["status"] == "EVALUATED"}
            for policy in ("IND_ONLY", "JOINT_ONLY", "IND_AND_JOINT", "IND_OR_JOINT",
                           "COMPLEMENT_AND_JOINT", "COMPLEMENT_OR_JOINT"):
                decision, status, reason = None, "NOT_EVALUATED", None
                try:
                    decision = int(kernel_pool(legacy_individuals, legacy_joint_votes, policy,
                                               registry=SOURCES5))
                    status = "EVALUATED"
                except EligibilityError as exc:
                    reason = str(exc)
                legacy_pool_rows.append({
                    "country": country, "issue": issue, "policy": policy,
                    "decision": decision, "status": status, "reason": reason,
                    "n_individual_evaluable": len(legacy_individuals),
                    "n_joint_evaluable": len(legacy_joint_votes),
                    "profile_id": PROFILE_ID})

            # Canonical packet path (approved B1): 10x9 motif Gate1/Gate2 + six Pools.
            pair_series, thresholds, state_owners, terminal_owners = {}, {}, {}, {}
            for s in SOURCES5:
                series = series_cache[(country, s, issue)]
                state_owners[s], terminal_owners[s] = research_state_and_terminal(
                    s, [r.value for _, r in series], prof_cache[(country, s, issue)])
            for a, b in PAIRS10:
                sa = series_cache[(country, a, issue)]
                sb = series_cache[(country, b, issue)]
                _, used = _matched(sa, sb, FREQUENCY[a], FREQUENCY[b])
                aa = [(r.period_id, r.value) for _, r, _ in used]
                bb = [(r.period_id, r.value) for _, _, r in used]
                pair_series[(a, b)] = (aa, bb)
                thresholds[(a, b)] = research_pair_thresholds(
                    min(len(aa), len(bb)), prof_cache[(country, a, issue)],
                    prof_cache[(country, b, issue)])
            pkt = build_issue_packet(
                country=country, issue=issue, pair_series=pair_series,
                individual_votes={s: r["vote"] for s, r in ind_issue.items()
                                  if r["status"] == "EVALUATED"},
                lag=LagPolicy("EXACT", 0, 0, None, LAG_OWNER),
                state_owners=state_owners, terminal_owners=terminal_owners,
                thresholds=thresholds, selected_configs=selected_configs,
                research_configs=research_configs, selected_pool=selected_pool)
            if pkt["config_parity"]["status"] != "PASS_SELECTED_EQUALS_CONSUMED_CONFIG_ID":
                raise AssertionError("PACKET_CONFIG_IDENTITY_MISMATCH:"
                                     f"{country}:{issue}:{pkt['config_parity']}")
            parity_rows.append({
                "country": country, "issue": issue,
                "status": pkt["config_parity"]["status"],
                "n_selected": pkt["n_selected"], "n_consumed": pkt["n_consumed"],
                "n_research": pkt["n_research_consumed"],
                "pool_fallback_used": pkt["pool_fallback_used"],
                "selected_pool_id": pkt["selected_pool_id"],
                "registry_pool_operator": pkt["registry_pool_operator"],
            })
            for code, count in pkt["not_evaluated_reason_counts"].items():
                ne_reason_rows.append({"country": country, "issue": issue,
                                      "motif_scope": "PAIR_MOTIF",
                                      "reason_code": code, "count": int(count)})
            for row in pkt["joint_rows"]:
                pair = row["pair"]
                pair_str = f"{pair[0]}-{pair[1]}" if isinstance(pair, (tuple, list)) else str(pair)
                gate1 = row.get("gate1")
                gate2 = row.get("gate2")
                g1 = 1 if gate1 == "PASS" else (0 if gate1 == "FAIL" else None)
                g2 = 1 if gate2 == "PASS" else (0 if gate2 == "FAIL" else None)
                n_bases = int(row.get("n_bases", 0) or 0)
                reason = (None if row["status"] in ("PASS", "FAIL")
                          else (row.get("reason") or "NOT_EVALUATED_UNCLASSIFIED"))
                joint_rows.append({
                    "country": country, "issue": issue, "pair": pair_str,
                    "profile_id": PROFILE_ID, "n_matched": n_bases + 1 if n_bases else 0,
                    "native_periods_a": ";".join(pid for pid, _ in pair_series[pair][0]),
                    "native_periods_b": ";".join(pid for pid, _ in pair_series[pair][1]),
                    "motif": row["motif"], "chosen": bool(row.get("selected")),
                    "selected": bool(row.get("selected")),
                    "research_consumed": bool(row.get("research_consumed")),
                    "config_id": row.get("config_id"),
                    "status": "EVALUATED" if row["status"] in ("PASS", "FAIL") else "NOT_EVALUATED",
                    "vote": row.get("vote"), "gate1": g1, "gate2": g2,
                    "gate1_mode": row.get("gate1_mode"),
                    "gate1_reason": row.get("gate1_reason"),
                    "gate2_reason": row.get("gate2_reason"),
                    "count_lower": row.get("count_lower"),
                    "count_upper": row.get("count_upper"),
                    "paired_count": row.get("paired_count"),
                    "paired_hi": None, "paired_lo": None,
                    "theta_hi_a": thresholds[pair].get("high_a"),
                    "theta_hi_b": thresholds[pair].get("high_b"),
                    "theta_lo_a": thresholds[pair].get("low_a"),
                    "theta_lo_b": thresholds[pair].get("low_b"),
                    "reason": reason, "reason_code": reason})
            counts = pool_evidence_counts(
                {s: r["vote"] for s, r in ind_issue.items() if r["status"] == "EVALUATED"}, pkt)
            for policy in ("IND_ONLY", "JOINT_ONLY", "IND_AND_JOINT", "IND_OR_JOINT",
                           "COMPLEMENT_AND_JOINT", "COMPLEMENT_OR_JOINT"):
                vote = pkt["pool_masks"].get(policy)       # absent = pool not determined (no vote)
                decision = 1 if vote == "PASS" else (0 if vote == "FAIL" else None)
                pool_rows.append({
                    "country": country, "issue": issue, "policy": policy,
                    "decision": decision,
                    "status": "EVALUATED" if decision is not None else "NOT_EVALUATED",
                    "reason": (None if decision is not None else
                               pkt["pool_reason_by_operator"].get(policy)
                               or "POOL_NOT_DERIVABLE"),
                    "is_issued_pool": bool(policy == pkt["selected_pool_id"]),
                    "issued_pool_id": pkt["selected_pool_id"],
                    "registry_pool_operator": pkt["registry_pool_operator"],
                    "pool_fallback_used": bool(pkt["pool_fallback_used"]),
                    "pool_status": pkt["pool_status"],
                    "pool_reason": pkt["pool_reason"],
                    **counts,
                    "profile_id": PROFILE_ID})
    joint_df = pd.DataFrame(joint_rows)
    pool_df = pd.DataFrame(pool_rows)
    # Packet feature block: one row per country-issue, one column per registered
    # research configuration plus coverage counters.  A NOT_EVALUATED vote is an empty cell (NaN) with
    # its reason in the joint tape; it is never a fabricated 0 and never a third class.
    feat = pd.DataFrame({"country": [c for c in EA7 for _ in months],
                         "month": [m for _ in EA7 for m in months]})
    res = joint_df[joint_df["research_consumed"] & joint_df["vote"].notna()].copy()
    research_cols: list = []
    if len(res):
        piv = res.pivot_table(index=["country", "issue"], columns="pair",
                              values="vote", aggfunc="first")
        piv.columns = [f"pkt_joint_{str(c).replace('-', '_')}" for c in piv.columns]
        research_cols = list(piv.columns)
        feat = feat.merge(piv.reset_index().rename(columns={"issue": "month"}),
                          on=["country", "month"], how="left")
    for col in research_cols:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
    feat["pkt_joint_not_evaluated_count"] = (feat[research_cols].isna().sum(axis=1)
                                 if research_cols else pd.Series(0, index=feat.index))
    feat["pkt_joint_research_pass_count"] = (feat[research_cols].sum(axis=1, skipna=True)
                                             if research_cols else pd.Series(0, index=feat.index))
    sel = joint_df[joint_df["selected"] & joint_df["vote"].notna()].copy()
    selected_cols: list = []
    if len(sel):
        piv = sel.pivot_table(index=["country", "issue"], columns="pair",
                              values="vote", aggfunc="first")
        piv.columns = [f"pkt_selected_{str(c).replace('-', '_')}" for c in piv.columns]
        selected_cols = list(piv.columns)
        feat = feat.merge(piv.reset_index().rename(columns={"issue": "month"}),
                          on=["country", "month"], how="left")
    feat["pkt_selected_pass_count"] = (feat[selected_cols].sum(axis=1, skipna=True)
                                       if selected_cols else pd.Series(0, index=feat.index))
    issued = pool_df[pool_df["is_issued_pool"]][
        ["country", "issue", "decision", "status", "pool_status"]].copy()
    issued = issued.rename(columns={"issue": "month", "decision": "pkt_pool_issued",
                                    "status": "pkt_pool_issued_status",
                                    "pool_status": "pkt_pool_issued_pool_status"})
    feat = feat.merge(issued, on=["country", "month"], how="left", validate="one_to_one")
    feat["pkt_pool_issued"] = pd.to_numeric(feat["pkt_pool_issued"], errors="coerce")
    if len(joint_df) and ((joint_df["status"] == "NOT_EVALUATED")
                          & joint_df["reason_code"].isna()).any():
        raise AssertionError("JOINT_NOT_EVALUATED_WITHOUT_REASON")
    return {"individual_tape": pd.DataFrame(ind_rows),
            "joint_tape": joint_df,
            "pool_tape": pool_df,
            "legacy_joint_tape": pd.DataFrame(legacy_joint_rows),
            "legacy_pool_tape": pd.DataFrame(legacy_pool_rows),
            "packet_features": feat,
            "packet_feature_joint_columns": research_cols,
            "packet_feature_selected_columns": selected_cols,
            "config_parity": pd.DataFrame(parity_rows),
            "not_evaluated_reason_ledger": pd.DataFrame(ne_reason_rows),
            "native_frame": frame,
            "profile": {"profile_id": PROFILE_ID, "window_months": WINDOW_MONTHS,
                        "prior_months": PRIOR_MONTHS,
                        "thresholds": "Q75/Q35 of prior adjacent native deltas",
                        "min_bases": "min(2,n-1)", "count_k": "ceil(0.5*M)",
                        "motif_primary": "REGISTRY_PER_PAIR",
                        "feature_block": ("SELECTED_REGISTRY" if selected_cols
                                          else "FULL_ROSTER_RESEARCH_PACKET"),
                        "status": "RESEARCH_NOT_OWNER_APPROVED"}}


def summarize(native: dict, labels: pd.DataFrame) -> dict:
    """Aggregate native tapes to the 01-02/03/04 summary shapes."""
    lab = labels[["country", "month", "label", "phase"]].rename(columns={"month": "issue"})
    ind = native["individual_tape"].merge(lab, on=["country", "issue"], how="left")
    jnt = native["joint_tape"].merge(lab, on=["country", "issue"], how="left")
    pol = native["pool_tape"].merge(lab, on=["country", "issue"], how="left")
    y = (ind["label"] != "N").astype(int)
    ind["_y"] = y
    native_obs = (native["native_frame"].groupby("source").size().to_dict()
                  if "native_frame" in native else {})
    ind_rows = []
    for source, sub in ind.groupby("source", sort=False):
        ev = sub[sub["status"] == "EVALUATED"]
        cm = _confusion(ev["_y"], ev["vote"])
        ind_rows.append({
            "source": source, "native_frequency": FREQUENCY[source],
            "n_source_issues": int(len(sub)),
            "n_native_observations": int(native_obs.get(source, 0)),
            "n_evaluated": int(len(ev)),
            "n_not_evaluated": int((sub["status"] != "EVALUATED").sum()),
            "vote_pass": int((ev["vote"] == 1).sum()),
            "vote_fail": int((ev["vote"] == 0).sum()),
            "pass_rate": (float((ev["vote"] == 1).mean()) if len(ev) else None),
            "triangle_pairs": int(ev["M"].sum()),
            "triangle_up": int(ev["up"].sum()), "triangle_down": int(ev["down"].sum()),
            "triangle_flat": int(ev["flat"].sum()),
            "terminal_mean_evaluated": (float(ev["terminal_mean"].mean()) if len(ev) else None),
            "n_comparator_N": int((sub["label"] == "N").sum()),
            "stress_TP": cm["TP"], "stress_FP": cm["FP"], "stress_FN": cm["FN"],
            "stress_TN": cm["TN"], "stress_FNR": cm["FNR"], "stress_FPR": cm["FPR"],
            "coverage": (float(len(ev) / len(sub)) if len(sub) else None),
            "status": "RESEARCH_NATIVE_PROFILE" if len(ev) else "NOT_EVALUATED",
            "profile_id": PROFILE_ID})
    jnt["_y"] = (jnt["label"] != "N").astype(int)
    for col, default in (("selected", False), ("research_consumed", False),
                         ("reason_code", None), ("gate1", None), ("gate2", None),
                         ("n_matched", 0)):
        if col not in jnt.columns:
            jnt[col] = default
    for col, default in (("reason", None), ("is_issued_pool", False),
                         ("issued_pool_id", None), ("pool_fallback_used", False)):
        if col not in pol.columns:
            pol[col] = default
    # pair-level complement of individual evidence: at least one member Individual PASS
    member = {}
    for source, sub in ind.groupby("source"):
        member[source] = sub.set_index(["country", "issue"])["vote"]
    jrows = []
    for (pair, motif), sub in jnt.groupby(["pair", "motif"], sort=False):
        a, b = pair.split("-")
        ia = pd.MultiIndex.from_frame(sub[["country", "issue"]])
        va = (member[a].reindex(ia).to_numpy() if a in member
              else np.full(len(sub), np.nan))
        vb = (member[b].reindex(ia).to_numpy() if b in member
              else np.full(len(sub), np.nan))
        ind_any = pd.Series(np.fmax(va, vb), index=sub.index)
        ev = sub[sub["status"] == "EVALUATED"].copy()
        ev_ind = ind_any.loc[ev.index].fillna(0).astype(int)
        rescued = int(((ev_ind == 0) & (ev["vote"] == 1) & (ev["_y"] == 1)).sum())
        newfp = int(((ev_ind == 0) & (ev["vote"] == 1) & (ev["_y"] == 0)).sum())
        u = sub[sub["status"] == "NOT_EVALUATED"]
        u_counts = u["reason_code"].fillna("NOT_EVALUATED_UNCLASSIFIED").value_counts().to_dict()
        jrows.append({
            "pair": pair, "motif": motif, "exact_lag": 0,
            "chosen": bool(sub["selected"].any()),
            "selected": int(sub["selected"].sum()),
            "research_consumed": int(sub["research_consumed"].sum()),
            "n_matched": int(sub["n_matched"].max()) if len(sub) else 0,
            "n_evaluated": int(len(ev)), "n_pass": int((ev["vote"] == 1).sum()),
            "gate1_pass": int((ev["gate1"] == 1).sum()), "gate2_pass": int((ev["gate2"] == 1).sum()),
            "rescued_fn": rescued, "new_fp": newfp,
            "n_comparator_N": int((sub["label"] == "N").sum()),
            "coverage": (float(len(ev) / len(sub)) if len(sub) else None),
            "n_not_evaluated": int(len(u)),
            "not_evaluated_reasons": json.dumps(u_counts, sort_keys=True),
            "not_evaluated_without_reason": int((u["reason_code"].isna()).sum()),
            "status": ("RESEARCH_NATIVE_9_MOTIF" if len(ev) else "NOT_EVALUATED"),
            "profile_id": PROFILE_ID})
    pol["_y"] = (pol["label"] != "N").astype(int)
    prows = []
    for policy, sub in pol.groupby("policy", sort=False):
        ev = sub[sub["status"] == "EVALUATED"]
        cm = _confusion(ev["_y"], ev["decision"])
        pre = sub[sub["phase"] == "PRE"]
        u = sub[sub["status"] == "NOT_EVALUATED"]
        prows.append({
            "operator": policy, "PASS": int((ev["decision"] == 1).sum()),
            "FAIL": int((ev["decision"] == 0).sum()),
            "N_NOT_EVALUATED": int(len(u)),
            "not_evaluated_reasons": json.dumps(u["reason"].fillna("NOT_EVALUATED_UNCLASSIFIED")
                                                .value_counts().to_dict(), sort_keys=True),
            "issued_pool": bool(sub["is_issued_pool"].any()),
            "issued_pool_id": (sub.loc[sub["is_issued_pool"], "issued_pool_id"].iloc[0]
                               if sub["is_issued_pool"].any() else None),
            "pool_fallback_used": bool(sub["pool_fallback_used"].any()),
            "pre_entry_TP": int(((pre["decision"] == 1) & (pre["label"] != "N")).sum()),
            "pre_entry_FP": int(((pre["decision"] == 1) & (pre["label"] == "N")).sum()),
            **{k: cm[k] for k in ("TP", "FP", "FN", "TN", "FNR", "FPR")},
            "coverage": (float(len(ev) / len(sub)) if len(sub) else None),
            "status": "RESEARCH_NATIVE_POOL" if len(ev) else "NOT_EVALUATED",
            "profile_id": PROFILE_ID})
    ne_ledger = (native.get("not_evaluated_reason_ledger") if "not_evaluated_reason_ledger" in native
                else pd.DataFrame())
    if len(ne_ledger):
        ne_reasons = (ne_ledger.groupby("reason_code", as_index=False)["count"].sum()
                     .sort_values("count", ascending=False))
    else:
        ne_reasons = pd.DataFrame(columns=["reason_code", "count"])
    parity = (native.get("config_parity") if "config_parity" in native
              else pd.DataFrame())
    parity_summary = {
        "n_issue_rows": int(len(parity)),
        "n_config_parity_pass": int((parity["status"] == "PASS_SELECTED_EQUALS_CONSUMED_CONFIG_ID").sum())
        if len(parity) else 0,
        "n_selected_configs": int(parity["n_selected"].max()) if len(parity) else 0,
        "n_consumed_configs": int(parity["n_consumed"].max()) if len(parity) else 0,
        "n_research_configs": int(parity["n_research"].max()) if len(parity) else 0,
        "pool_fallback_used": bool(parity["pool_fallback_used"].any()) if len(parity) else None,
    }
    return {"individual": pd.DataFrame(ind_rows), "joint": pd.DataFrame(jrows),
            "pools": pd.DataFrame(prows),
            "not_evaluated_reasons": ne_reasons, "config_parity": parity_summary}


def _confusion(y, pred) -> dict:
    y = pd.Series(list(y)).astype(int)
    p = pd.Series(list(pred)).astype(float)
    valid = p.notna()
    y, p = y[valid], p[valid]
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FNR": (fn / (tp + fn) if (tp + fn) else None),
            "FPR": (fp / (fp + tn) if (fp + tn) else None)}


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, source), sub in table.groupby(["country", "source"], sort=False):
        rows.append({
            "country": country, "source": source,
            "native_frequency": FREQUENCY[source],
            "n_native_obs": int(len(sub)),
            "native_first": str(sub["period_end"].min()),
            "native_last": str(sub["period_end"].max()),
            "n_distinct_periods": int(sub["native_period"].nunique()),
            "release_rules_status": "RESEARCH_ASOF_NOT_OFFICIAL_ISSUER_PIT",
        })
    return pd.DataFrame(rows)
