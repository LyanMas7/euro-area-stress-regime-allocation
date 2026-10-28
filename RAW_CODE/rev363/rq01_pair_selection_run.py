"""Device-level G0-G4 registry over all 90 pair x motif configurations.

The registry evaluates every ordered motif (not a hard-coded UU default) on the
DEV native tape, judges added value on the FINAL Pool mask (spec 7.3), selects
one registered configuration per pair, and runs the exact <=1024 pair-SET
search for each joint Pool policy.  A policy is then chosen on DEV only.

Two distinct outputs are kept apart:

* `selected_configs` - the registry decision consumed by the issued Pool;
* `research_configs` - one declared research feature configuration per pair,
  consumed only by the diagnostic M10/M11/Packet ablation.  It is never called
  a selected or owner-approved configuration.

No OOS row enters this module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_adapters.rq01.common import SOURCES
from research_adapters.vote_contract import decide_and, decide_or, is_nonvote, not_evaluated
from research_adapters.rq01.motif_issue_packet import (LAG_OWNER, OWNER, POLICY_ID,
                                                       RELEASE_RULE)
from research_adapters.rq01.pair_selection import (GATES, JointConfig,
                                                   config_counters,
                                                   evaluate_pair_gates, joint_id,
                                                   joint_threshold_domain,
                                                   select_pair_set)
from research_adapters.rq01.pairs import MOTIFS, PAIRS, LagPolicy
from research_adapters.rq01.pools import OPERATORS

RESEARCH_FP_BUDGET_FRACTION = 0.15
RESEARCH_SOURCE_CAP = 2
RESEARCH_RELEASE_SCOPE = RELEASE_RULE
JOINT_OPERATORS = tuple(op for op in OPERATORS if op != "IND_ONLY")
POOL_POLICY_ORDER = {op: i for i, op in enumerate(JOINT_OPERATORS)}
MOTIF_ORDER = {m: i for i, m in enumerate(MOTIFS)}

MECHANISMS = {
    ("CISS", "CLIFS"): "BANK_AND_CORPORATE_STRESS_COMOVEMENT",
    ("CISS", "DFC"): "BANK_STRESS_AND_DEBT_SERVICE_CAPACITY",
    ("CISS", "GFC"): "BANK_STRESS_AND_GLOBAL_FINANCIAL_CONDITIONS",
    ("CISS", "VIX"): "BANK_STRESS_AND_MARKET_VOLATILITY",
    ("CLIFS", "DFC"): "CORPORATE_STRESS_AND_DEBT_SERVICE_CAPACITY",
    ("CLIFS", "GFC"): "CORPORATE_STRESS_AND_GLOBAL_FINANCIAL_CONDITIONS",
    ("CLIFS", "VIX"): "CORPORATE_STRESS_AND_MARKET_VOLATILITY",
    ("DFC", "GFC"): "DEBT_SERVICE_CAPACITY_AND_GLOBAL_FINANCIAL_CONDITIONS",
    ("DFC", "VIX"): "DEBT_SERVICE_CAPACITY_AND_MARKET_VOLATILITY",
    ("GFC", "VIX"): "GLOBAL_FINANCIAL_CONDITIONS_AND_MARKET_VOLATILITY",
}


def _config(pair, motif: str) -> JointConfig:
    return JointConfig(pair[0], pair[1], motif,
                       LagPolicy("EXACT", 0, 0, None, LAG_OWNER),
                       "RISK_ORIENTED", POLICY_ID, RESEARCH_RELEASE_SCOPE,
                       OWNER).canonical()


def _tri(value, reason: str = "MASK_KEY_NOT_EVALUATED"):
    """Tape cell -> 'PASS'/'FAIL' for a binary vote, NonVote for an empty (not evaluated) cell.

    GLOBAL NO-U: an absent pair-mask entry isolates the key (no vote); an invalid token is rejected
    instead of being normalised to a third value."""
    if isinstance(value, str):
        u = value.upper()
        if u in ("PASS", "FAIL"):
            return u
        raise ValueError("INVALID_MASK_TOKEN_NOT_A_BINARY_VOTE:" + repr(value))
    if value is None:
        return not_evaluated(reason)
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("INVALID_MASK_VALUE:" + repr(value)) from exc
    if not np.isfinite(f):
        return not_evaluated(reason)
    if f in (0.0, 1.0):
        return "PASS" if f == 1.0 else "FAIL"
    raise ValueError("INVALID_MASK_VALUE_NOT_0_OR_1:" + repr(value))


def _or(values):
    vals = list(values)
    if not vals:
        return not_evaluated("NO_COMPONENTS")
    return decide_or(*vals)


def _and(a, b):
    return decide_and(a, b)


def _pair_key(value):
    if isinstance(value, str) and value.startswith("("):
        import ast
        value = ast.literal_eval(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return tuple(sorted((str(value[0]), str(value[1])), key=SOURCES.index))
    a, b = str(value).split("-")
    return tuple(sorted((a, b), key=SOURCES.index))


def _masks(native, panel_dev):
    keys = list(zip(panel_dev["country"], panel_dev["month"]))
    idx = pd.MultiIndex.from_tuples(keys, names=["country", "issue"])
    joint = native["joint_tape"].copy()
    joint["pair_key"] = [_pair_key(v) for v in joint["pair"]]
    pair_masks = {}
    for pair in PAIRS:
        for motif in MOTIFS:
            sub = joint[(joint["pair_key"] == pair) & (joint["motif"] == motif)]
            pivot = sub.pivot_table(index=["country", "issue"], values="vote",
                                    aggfunc="first").reindex(idx)
            col = pivot["vote"] if "vote" in pivot.columns else pd.Series(index=idx, dtype=float)
            pair_masks[(pair, motif)] = [_tri(v) for v in col.tolist()]

    ind = native["individual_tape"].copy()
    ip = ind.pivot_table(index=["country", "issue"], columns="source",
                         values="vote", aggfunc="first").reindex(idx)
    individual = {s: ([_tri(v) for v in ip[s].tolist()] if s in ip.columns
                      else [not_evaluated("SOURCE_ABSENT_FROM_INDIVIDUAL_TAPE")] * len(keys))
                  for s in SOURCES}
    y = (panel_dev["label"].astype(str) != "N").astype(int).tolist()
    return keys, pair_masks, individual, y


def _base_mask(individual, n):
    return [_or(individual[s][i] for s in SOURCES) for i in range(n)]


def _final_mask(operator, joint, base, comp):
    if operator == "JOINT_ONLY":
        return list(joint)
    if operator == "IND_AND_JOINT":
        return [_and(base[i], joint[i]) for i in range(len(joint))]
    if operator == "IND_OR_JOINT":
        return [_or((base[i], joint[i])) for i in range(len(joint))]
    if operator == "COMPLEMENT_AND_JOINT":
        return [_and(comp[i], joint[i]) for i in range(len(joint))]
    if operator == "COMPLEMENT_OR_JOINT":
        return [_or((comp[i], joint[i])) for i in range(len(joint))]
    raise ValueError("UNREGISTERED_POOL_OPERATOR")


def _confusion(y, pred, mask):
    tp = fp = fn = tn = 0
    for i, use in enumerate(mask):
        if not use:
            continue
        if y[i] == 1 and pred[i] == "PASS":
            tp += 1
        elif y[i] == 0 and pred[i] == "PASS":
            fp += 1
        elif y[i] == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _pool_impact(operator, y, base, joint, comp, fp_budget):
    final = _final_mask(operator, joint, base, comp)
    # matched evaluable keys: both the Individual-only and the final pool decision were issued
    common = [not is_nonvote(base[i]) and not is_nonvote(final[i]) for i in range(len(y))]
    n_eval = int(sum(common))
    cm_b = _confusion(y, base, common)
    cm_f = _confusion(y, final, common)
    # rescue / new FP / lost TP / avoided FP on the SAME matched evaluable keys only: a key whose Individual
    # (or final) decision was not issued is never read as a FAIL (REV373 counted it as one).
    rescue = sum(1 for i, yi in enumerate(y)
                 if common[i] and yi == 1 and final[i] == "PASS" and base[i] != "PASS")
    new_fp = sum(1 for i, yi in enumerate(y)
                 if common[i] and yi == 0 and final[i] == "PASS" and base[i] != "PASS")
    lost_tp = sum(1 for i, yi in enumerate(y)
                  if common[i] and yi == 1 and base[i] == "PASS" and final[i] != "PASS")
    avoided_fp = sum(1 for i, yi in enumerate(y)
                     if common[i] and yi == 0 and base[i] == "PASS" and final[i] != "PASS")

    def _same(a, b):
        # two non-issued decisions are the same (no decision); a vote never equals a non-vote
        if is_nonvote(a) or is_nonvote(b):
            return is_nonvote(a) and is_nonvote(b)
        return a == b
    equals = all(_same(base[i], final[i]) for i in range(len(y)))
    tp_gain = cm_f["tp"] > cm_b["tp"]
    fp_reduction = cm_f["fp"] < cm_b["fp"]
    added = bool(n_eval and not equals and (tp_gain or (cm_f["tp"] == cm_b["tp"] and fp_reduction)))
    if added:
        reason = "TP_GAIN" if tp_gain else "EQUAL_TP_FEWER_FP"
    elif n_eval == 0:
        reason = "NO_COMMON_EVALUABLE_KEYS"
    elif equals:
        reason = "OVERLAP_ONLY_NO_INCREMENT_OVER_INDIVIDUAL"
    else:
        reason = "NO_TP_GAIN_AND_NO_FP_REDUCTION"
    return {
        "comparison": "FINAL_POOL_MASK", "pool_operator": operator,
        "n_evaluable": n_eval, "tp_final": cm_f["tp"], "fp_final": cm_f["fp"],
        "fn_final": cm_f["fn"], "tn_final": cm_f["tn"],
        "tp_base": cm_b["tp"], "fp_base": cm_b["fp"],
        "rescue": int(rescue), "new_fp": int(new_fp),
        "lost_tp": int(lost_tp), "avoided_fp": int(avoided_fp),
        "fp_increase": int(cm_f["fp"] - cm_b["fp"]),
        "fp_budget": int(fp_budget), "fp_budget_ok": bool(new_fp <= fp_budget),
        "final_mask_equals_individual": bool(equals),
        "added_value": added, "added_value_reason": reason,
        "n_labelled": int(len(y)),
    }


def _gate_evidence(pair, motif, joint_rows, y, impact):
    n_levels = int(joint_rows["n_matched"].max()) if len(joint_rows) else 0
    m = n_levels * (n_levels - 1) // 2
    min_bases = max(1, min(2, n_levels - 1))
    count_k = max(1, int(math.ceil(0.5 * m))) if m else 1
    row = joint_rows.iloc[0] if len(joint_rows) else {}
    hi_a = float(row.get("theta_hi_a", 1.0) or 1.0) if len(joint_rows) else 1.0
    lo_a = float(row.get("theta_lo_a", 0.5) or 0.5) if len(joint_rows) else 0.5
    hi_b = float(row.get("theta_hi_b", 1.0) or 1.0) if len(joint_rows) else 1.0
    lo_b = float(row.get("theta_lo_b", 0.5) or 0.5) if len(joint_rows) else 0.5
    domain = joint_threshold_domain(max(n_levels, 1), min_bases, count_k, hi_a, lo_a)
    return {
        "mechanism": {"mechanism": MECHANISMS[pair], "testable": True},
        "native_clock": {
            "native_ids_verified": True,
            "release_verified": True,
            "release_verification_scope": RESEARCH_RELEASE_SCOPE,
            "n_matched_native_pairs": n_levels,
            "min_matched_pairs": 2,
        },
        "motif": {
            "lag": {"kind": "EXACT", "lag": 0, "delta": 0},
            "motif": motif,
            "count_method": "DISTINCT_NATIVE_ONE_TO_ONE",
        },
        "joint_gate": {
            "owner_present": True,
            "owner_scope": OWNER,
            "gate1_feasible": bool(domain["nonempty"]),
            "threshold_domain_nonempty": bool(domain["nonempty"]),
            "domain_reasons": domain["reasons"],
        },
        "added_value": impact,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def select_pairs(native, panel_dev, *, out_dir=None,
                 fp_budget_fraction=RESEARCH_FP_BUDGET_FRACTION,
                 source_cap=RESEARCH_SOURCE_CAP) -> dict:
    keys, pair_masks, individual, y = _masks(native, panel_dev)
    n = len(keys)
    if n == 0:
        return {"status": "NOT_EVALUATED", "reason": "NO_DEV_KEYS"}
    base = _base_mask(individual, n)
    n_neg = sum(1 for v in y if v == 0)
    fp_budget = int(round(fp_budget_fraction * n_neg))
    joint = native["joint_tape"].copy()
    joint["pair_key"] = [_pair_key(v) for v in joint["pair"]]
    configs = {(p, m): _config(p, m) for p in PAIRS for m in MOTIFS}

    # ---------------- every pair x motif x joint pool policy is evaluated
    nominal, executed = [], 0
    eligible = {}
    for pair in PAIRS:
        comp = [_or(individual[s][i] for s in SOURCES if s not in pair) for i in range(n)]
        for motif in MOTIFS:
            rows = joint[(joint["pair_key"] == pair) & (joint["motif"] == motif)]
            jm = pair_masks[(pair, motif)]
            cfg = configs[(pair, motif)]
            cid = joint_id(cfg)
            for operator in JOINT_OPERATORS:
                impact = _pool_impact(operator, y, base, jm, comp, fp_budget)
                gates = evaluate_pair_gates(
                    _gate_evidence(pair, motif, rows, y, impact))
                executed += 1
                eligible[cid] = bool(eligible.get(cid, False) or all(
                    gates["gates"][g] == "PASS" for g in GATES[:4]))
                nominal.append({
                    "pair": "-".join(pair), "pair_key": "-".join(pair),
                    "motif": motif, "config_id": cid, "pool_operator": operator,
                    **{k: v for k, v in gates["gates"].items()},
                    "reasons": json.dumps(gates["reasons"], sort_keys=True),
                    "pair_status": gates["pair_status"],
                    "pair_selected": gates["pair_selected"],
                    "research_only_candidate": gates["research_only_candidate"],
                    "joint_pass_at_issue": gates["joint_pass_at_issue"],
                    "n_matched_native": int(rows["n_matched"].max()) if len(rows) else 0,
                    "n_joint_pass_dev": int(sum(v == "PASS" for v in jm)),
                    "n_joint_not_evaluated_dev": int(sum(is_nonvote(v) for v in jm)),
                    **{k: impact[k] for k in
                       ("rescue", "new_fp", "lost_tp", "avoided_fp", "tp_final", "fp_final",
                        "tp_base", "fp_base", "n_evaluable", "fp_increase", "added_value",
                        "added_value_reason", "final_mask_equals_individual")},
                })
            eligible[cid] = gates["pair_status"] in ("SELECTED_CANDIDATE", "RESEARCH_ONLY")
    ledger = pd.DataFrame(nominal)
    executed_ids = [cid for cid, ok in eligible.items() if ok]
    counters = config_counters(
        [configs[k] for k in configs], eligible={cid: eligible[cid] for cid in eligible},
        masks={joint_id(configs[(p, m)]): pair_masks[(p, m)] for p in PAIRS for m in MOTIFS},
        executed=executed_ids)

    # ---------------- one registered configuration per pair, then exact pair-SET
    policy_results, per_operator_cfg = {}, {}
    sel_by_key = ledger.set_index(["pair_key", "motif", "pool_operator"])
    for operator in JOINT_OPERATORS:
        masks, statuses, cfgs = {}, {}, {}
        for pair in PAIRS:
            sub = ledger[(ledger.pair_key == "-".join(pair)) & (ledger.pool_operator == operator)
                         & (ledger.pair_status == "SELECTED_CANDIDATE")]
            if len(sub) == 0:
                continue
            sub = sub.sort_values(
                by=["tp_final", "fp_final", "lost_tp", "new_fp", "motif"],
                ascending=[False, True, True, True, True], kind="stable")
            best = sub.iloc[0]
            pkey = (pair[0], pair[1])
            masks[pkey] = pair_masks[(pkey, best["motif"])]
            statuses[pkey] = "SELECTED_CANDIDATE"
            cfgs[pkey] = configs[(pkey, best["motif"])]
        if not masks:
            policy_results[operator] = {
                "status": "NO_ELIGIBLE_CONFIG", "chosen": [], "n_candidate_pairs": 0,
                "final_confusion": {"tp": 0, "fp": n_neg, "fn": sum(y), "tn": 0},
                "n_subsets_enumerated": 1}
            continue
        res = select_pair_set(masks, y, fp_budget=fp_budget, source_cap=source_cap,
                              pool_operator=operator, individual_votes=individual,
                              pair_status=statuses, pair_configs=cfgs)
        policy_results[operator] = res
        per_operator_cfg[operator] = cfgs

    def _improves_on_base(res):
        fin = res.get("final_confusion") or {}
        bas = res.get("baseline_confusion") or {}
        if not fin or not bas:
            return False
        return bool(fin["tp"] > bas["tp"] or (fin["tp"] == bas["tp"] and fin["fp"] < bas["fp"]))

    ranked = sorted(
        ((op, r) for op, r in policy_results.items()
         if r.get("status") != "NO_ELIGIBLE_CONFIG" and r.get("chosen")
         and _improves_on_base(r)),
        key=lambda kv: (-kv[1]["final_confusion"]["tp"], kv[1]["final_confusion"]["fp"],
                        len(kv[1]["chosen"]), POOL_POLICY_ORDER[kv[0]]))
    policy_selected = bool(ranked)
    selected_pool = ranked[0][0] if policy_selected else "IND_OR_JOINT"
    win = policy_results.get(selected_pool, {"chosen": [], "status": "NO_ELIGIBLE_CONFIG"})
    chosen = tuple(win.get("chosen", ())) if policy_selected else ()
    selected_configs = ({p: per_operator_cfg[selected_pool][p] for p in chosen}
                        if policy_selected else {})
    selected_ids = {p: joint_id(selected_configs[p]) for p in selected_configs}

    # ---------------- declared research feature configurations (not selected)
    research_configs, research_ids = {}, {}
    for pair in PAIRS:
        sub = ledger[(ledger.pair_key == "-".join(pair))
                     & (ledger.pool_operator == "IND_OR_JOINT")
                     & (ledger.G0_MECHANISM == "PASS")
                     & (ledger.G1_NATIVE_CLOCK == "PASS")
                     & (ledger.G2_EXACT_MOTIF == "PASS")
                     & (ledger.G3_JOINT_GATE == "PASS")]
        if len(sub) == 0:
            continue
        sub = sub.sort_values(by=["rescue", "fp_final", "tp_final", "motif"],
                              ascending=[False, True, False, True], kind="stable")
        best = sub.iloc[0]
        research_configs[pair] = configs[(pair, best["motif"])]
        research_ids[pair] = joint_id(research_configs[pair])

    gate_rows = ledger[ledger.pool_operator == selected_pool].copy()
    out = {
        "status": "PAIR_SET_SELECTED" if chosen else "NO_PAIR_SELECTED",
        "scope": "DEV_ONLY_2005_07_2019_12",
        "selection_pool_operator": selected_pool,
        "n_dev_keys": n, "n_labelled": n, "n_negative": n_neg,
        "fp_budget_fraction": fp_budget_fraction, "fp_budget": fp_budget,
        "source_cap": source_cap,
        "n_nominal_configs": len(PAIRS) * len(MOTIFS),
        "n_config_policy_evaluations": len(nominal),
        "n_eligible_configs": int(counters["N_eligible"]),
        "n_unique_complete_masks": int(counters["N_unique_decision_masks"]),
        "n_candidate_pairs": int(win.get("n_candidate_pairs", 0)),
        "chosen_pairs": ["-".join(p) for p in chosen],
        "selected_config_ids": {f"{a}-{b}": selected_ids[(a, b)] for a, b in chosen},
        "research_feature_config_ids": {f"{a}-{b}": research_ids[(a, b)] for a, b in research_ids},
        "research_feature_motifs": {f"{a}-{b}": research_configs[(a, b)].motif
                                    for a, b in research_configs},
        "gate_status": {f"{a}-{b}": ("SELECTED_CANDIDATE"
                                     if any(gate_rows[(gate_rows.pair_key == f"{a}-{b}")
                                                      & (gate_rows.pair_selected)]) else
                                     ("REJECTED" if len(gate_rows[gate_rows.pair_key == f"{a}-{b}"])
                                      else "NOT_EVALUATED"))
                        for a, b in PAIRS},
        "selector": _json_safe({k: v for k, v in win.items() if k != "ledger"}),
        "pool_policy_comparison": _json_safe({
            op: {k: v for k, v in r.items() if k not in ("ledger", "config_ids")}
            for op, r in policy_results.items()}),
        "counters": counters,
        "selection_claim": "RESEARCH_REGISTRY_DECISION_NOT_OWNER_CERTIFIED_AND_NOT_JOINT_PASS_AT_ANY_ISSUE",
        "owner_certified": False,
        "no_oos": True,
    }
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        gate_rows.to_csv(out_path / "RQ01_PAIR_G0_G4_DEV.csv", index=False)
        ledger.to_csv(out_path / "RQ01_PAIR_MOTIF_POOL_LEDGER_DEV.csv", index=False)
        pd.DataFrame([{
            "pool_operator": op,
            "status": r.get("status"),
            "chosen_pairs": "|".join("-".join(p) for p in r.get("chosen", ())),
            "tp_final": r.get("final_confusion", {}).get("tp"),
            "fp_final": r.get("final_confusion", {}).get("fp"),
            "lost_tp": r.get("lost_tp"),
            "avoided_fp": r.get("avoided_fp"),
            "n_candidate_pairs": r.get("n_candidate_pairs"),
            "n_subsets_enumerated": r.get("n_subsets_enumerated"),
            "n_unique_final_masks": r.get("n_unique_final_masks"),
            "selected_policy": op == selected_pool,
        } for op, r in policy_results.items()]).to_csv(
            out_path / "RQ01_POOL_POLICY_COMPARISON_DEV.csv", index=False)
        (out_path / "RQ01_PAIR_SET_SELECTION_DEV.json").write_text(
            json.dumps(_json_safe(out), indent=2, default=str) + "\n", encoding="utf-8")
        registry = []
        for pair in PAIRS:
            for motif in MOTIFS:
                cid = joint_id(configs[(pair, motif)])
                registry.append({
                    "pair": "-".join(pair), "config_id": cid, "leader": pair[0],
                    "follower": pair[1], "motif": motif, "lag_kind": "EXACT",
                    "lag_months": 0, "delta": 0, "owner": OWNER,
                    "release_rule": RELEASE_RULE, "selected": pair in selected_configs
                    and selected_configs[pair].motif == motif,
                    "research_feature": pair in research_configs
                    and research_configs[pair].motif == motif,
                })
        (out_path / "RQ01_SELECTED_CONFIG_REGISTRY.json").write_text(
            json.dumps({"scope": "RESEARCH_DEV_ONLY", "owner_certified": False,
                        "selected_pool_operator": selected_pool,
                        "registry": registry}, indent=2) + "\n", encoding="utf-8")
        if win.get("ledger"):
            pd.DataFrame(win["ledger"]).to_csv(
                out_path / "RQ01_PAIR_SET_ENUMERATION_DEV.csv", index=False)
    out["selected_configs"] = selected_configs
    out["research_configs"] = research_configs
    return out
