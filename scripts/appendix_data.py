"""Rows for the public threshold appendix.

Each row is a claim about the executed configuration. `source` names the file and
line for code facts, or the selected-configuration file for registered values.
Values are copied verbatim; nothing is rounded or inferred.
"""
from __future__ import annotations

import json
from pathlib import Path

J01 = "THRESHOLD/RQ01/rq01_selected.json"
J02 = "THRESHOLD/RQ02/rq02_selected.json"


def _row(group, section, parameter, value, unit, condition, effect, source, status="active", evidence=None):
    return {"group": group, "section": section, "parameter": parameter, "value": value, "unit": unit,
            "condition": condition, "effect": effect, "source": source, "status": status, "evidence": evidence}


A1 = [
    _row("RQ01", "A.1", "Observation window", "12", "months", "every issue",
         "One latest as-of native level per month; the vote is formed at the issue month",
         "RAW_CODE/rev363/rq01_native_ijp.py:44"),
    _row("RQ01", "A.1", "Native frequency - CISS / CLIFS / DFC / GFC price / VIX", "W / M / Q / M / D", "frequency",
         "sampling", "Weekly and daily series are sampled; quarterly series are de-duplicated so a quarter is never "
         "counted twice", "RAW_CODE/rev363/rq01_native_ijp.py:49"),
    _row("RQ01", "A.1", "Direction of movement", "CISS +1, CLIFS +1, DFC -1, GFC price +1, VIX +1", "sign",
         "every delta", "D = sign x (level_end - level_base); DFC enters with a reversed sign, so a wider credit "
         "spread still means more stress", "RAW_CODE/rev363/rq01_native_ijp.py:50"),
    _row("RQ01", "A.1", "Prior-delta history", "deltas before issue minus 11 months; at least 4", "months / count",
         "every issue", "Reference distribution for the magnitude cuts; below four prior deltas the source is "
         "NOT_EVALUATED", "RAW_CODE/rev363/rq01_native_ijp.py:165"),
    _row("RQ01", "A.1", "Strong magnitude cut (theta_hi)", "75th percentile of prior absolute adjacent deltas",
         "native units", "per country x source x issue",
         "An endpoint difference above this cut counts as a strong confirmation",
         "RAW_CODE/rev363/rq01_native_ijp.py:175"),
    _row("RQ01", "A.1", "Weak magnitude cut (theta_lo)", "35th percentile of prior absolute adjacent deltas",
         "native units", "clamped to 0 <= lo <= hi", "Lower cut that feeds the count leg of the vote",
         "RAW_CODE/rev363/rq01_native_ijp.py:176"),
    _row("RQ01", "A.1", "Minimum native levels", "2", "count", "every issue",
         "Fewer than two levels cannot produce a delta; the source is flagged N_LT_2_NATIVE_LEVELS",
         "RAW_CODE/rev363/rq01_native_ijp.py:192"),
    _row("RQ01", "A.1", "Minimum confirming bases", "min(2, n-1)", "count", "strong and weak legs",
         "A leg confirms only when at least this many base pairs pass the magnitude test",
         "RAW_CODE/rev363/rq01_native_ijp.py:201"),
    _row("RQ01", "A.1", "Confirmation count threshold", "max(1, ceil(0.5 x M)), M = n(n-1)/2", "count",
         "count leg", "Half of the available base pairs must confirm", "RAW_CODE/rev363/rq01_native_ijp.py:202"),
    _row("RQ01", "A.1", "Magnitude predicate", "D > 0 AND D >= cut", "rule", "each base pair",
         "Equality confirms; a zero difference never confirms", "RAW_CODE/research_adapters/ijp_kernel.py:112"),
    _row("RQ01", "A.1", "Count predicate", "signed_count >= count_k", "rule", "count leg",
         "Counts signed confirming bases only", "RAW_CODE/research_adapters/ijp_kernel.py:115"),
    _row("RQ01", "A.1", "Individual vote", "Strong OR (Weak AND Count)", "rule", "every issue",
         "Single-issue decision with no persistence requirement", "RAW_CODE/research_adapters/ijp_kernel.py:116"),
]

A2 = [
    _row("RQ01", "A.2", "Pair roster and motif lag", "10 pairs, 9 ordered motifs, EXACT L = 0, delta = 0",
         "count / months", "pair registry",
         "Only the EXACT(0,0) lag is registered, so no pair may use a future publication",
         "RAW_CODE/rev363/rq01_pair_selection_run.py:61"),
    _row("RQ01", "A.2", "Selection gate G1 - matched native pairs", ">= 2", "count", "pair eligibility",
         "Below two matched native pairs the pair is INELIGIBLE", "RAW_CODE/research_adapters/rq01/pair_selection.py:181"),
    _row("RQ01", "A.2", "Selection gate G3 - domain check", "1 <= min_bases <= n-1; 0 <= count_k <= M; lo <= hi",
         "rule", "pair eligibility", "A pair whose count geometry is impossible is BLOCKED",
         "RAW_CODE/research_adapters/rq01/pair_selection.py:157"),
    _row("RQ01", "A.2", "Selection gate G4 - false-positive budget",
         "new_fp <= round(0.15 x n_neg); source cap 2; <= 1024 subsets", "count", "pair selection",
         "Budget that stops a pair from buying true positives with false alarms",
         "RAW_CODE/rev363/rq01_pair_selection_run.py:38"),
    _row("RQ01", "A.2", "Per-pair configuration choice",
         "tp_final down, fp_final up, lost_tp up, new_fp up, motif up", "lexicographic",
         "one configuration per pair", "Deterministic tie-break",
         "RAW_CODE/rev363/rq01_pair_selection_run.py:330"),
    _row("RQ01", "A.2", "Selected pairs in the registered rule", "OR pkt_joint_CLIFS_VIX; NOT pkt_joint_DFC_GFC",
         "packet flag", "registered rule",
         "Two pairs survive selection: one can only add stress, one can only remove it",
         "THRESHOLD/RQ01/rq01_selected.json"),
]

A3 = [
    _row("RQ01", "A.3", "Activation magnitude cuts", "high = Q75, low = Q35 per member", "native units",
         "selected pair, every issue",
         "The cuts used after selection are the member percentiles, not the selection gates",
         "RAW_CODE/research_adapters/rq01/motif_issue_packet.py:69"),
    _row("RQ01", "A.3", "Gate 1 - separate-base magnitude test",
         "PASS if a base has D_a >= gate1_a AND a base has D_b >= gate1_b", "rule", "member magnitudes",
         "The two members may confirm on different base pairs",
         "RAW_CODE/research_adapters/rq01/motif_issue_packet.py:186"),
    _row("RQ01", "A.3", "Gate 2 - paired count test",
         "strong >= min_bases OR (weak >= min_bases AND paired_count >= count_k)", "rule", "paired bases",
         "A pair needs either strong pairs or a majority of paired confirmations",
         "RAW_CODE/research_adapters/rq01/motif_issue_packet.py:139"),
    _row("RQ01", "A.3", "Joint activation", "Gate 1 PASS AND Gate 2 PASS", "rule", "every issue",
         "Either gate failing yields FAIL; insufficient data yields NOT_EVALUATED, never zero",
         "RAW_CODE/research_adapters/rq01/motif_issue_packet.py:231"),
    _row("RQ01", "A.3", "Synchronisation", "monthly intersection with a single quarterly anchor; >= 2 matched anchors",
         "count", "every issue", "A carried quarter cannot be double-counted",
         "RAW_CODE/rev363/rq01_native_ijp.py:260"),
]

A5 = [
    _row("RQ01", "A.5", "Normal state definition", "pooled decision calm; P_N = 1 - P_S", "probability",
     "every issue in the RQ03 overlay",
     "When the pooled rule is calm, P_S is the weekly hazard of the H6 quarterly pre-onset probability "
     "1-(1-p_H6)^(1/78), so P_N is the probability that stress does not start within the week. The published "
     "pre-onset choice is H6 quarterly (18-month model); the shipped overlay script reads its H12 column, which is "
     "disclosed in the audit (D15).", "THRESHOLD/RQ02/rq02_selected.json"),
_row("RQ01", "A.5", "Stress state definition", "pooled decision fires; P_N = weekly exit hazard", "probability",
     "every issue in the RQ03 overlay",
     "When the pooled rule fires, P_N is the weekly exit hazard of the 3-month exit probability 1-(1-p_3M)^(1/13), "
     "so P_S is the probability that the episode continues; P_N and P_S are complementary by construction",
     "THRESHOLD/RQ02/rq02_selected.json"),
_row("RQ01", "A.5", "Peak percentile p_E", "0.90", "quantile", "phase classification",
         "Peak branch requires a rise above this quantile of the normal-state reference",
         "RAW_CODE/rev363/ijp_phase.py:30"),
    _row("RQ01", "A.5", "Peak percentile p_P", "0.75", "quantile", "phase classification", "Peak level test",
         "RAW_CODE/rev363/ijp_phase.py:30"),
    _row("RQ01", "A.5", "De-escalation percentile p_D", "0.50", "quantile", "phase classification",
         "Falling below the median marks de-escalation", "RAW_CODE/rev363/ijp_phase.py:30"),
    _row("RQ01", "A.5", "Recovery percentile p_L", "0.25", "quantile", "phase classification",
         "Lower reference of the recovery band", "RAW_CODE/rev363/ijp_phase.py:30"),
    _row("RQ01", "A.5", "Tolerance percentiles p_H / p_X", "0.75 / 0.75", "quantile", "phase classification",
         "Exit tolerance band around the phase median", "RAW_CODE/rev363/ijp_phase.py:30"),
    _row("RQ01", "A.5", "Phase quorum - ENTRY / PEAK / DE-ESCALATION / RECOVERY / EXIT", "3 / 2 / 2 / 3 / 3",
         "votes of 5", "every issue",
         "K-of-5 bound rule: three bounds must agree for entry, recovery and exit, two for peak and de-escalation",
         "RAW_CODE/research_adapters/rq01/five_state.py:33"),
    _row("RQ01", "A.5", "Phase persistence", "2", "consecutive issues", "every phase change",
         "A phase is confirmed only when it repeats", "RAW_CODE/research_adapters/rq01/five_state.py:34"),
    _row("RQ01", "A.5", "Peak / Trough / Recovery as probabilities", "not implemented", "probability",
         "requested probabilistic states",
         "The executed path classifies phases by quantile, quorum and persistence. It emits no peak, trough or "
         "recovery probability and the three are not a distribution",
         "RAW_CODE/research_adapters/rq01/five_state.py:123", "unavailable"),
]

B1 = [
    _row("RQ02", "B.1", "Long-horizon change window", "27", "months", "entry features",
         "Signed change over 27 months, z-scaled", "RAW_CODE/rev363/rq02_features.py:35"),
    _row("RQ02", "B.1", "Short-horizon change window", "6", "months", "exit and bottom features",
         "Signed change over 6 months, z-scaled", "RAW_CODE/rev363/rq02_features.py:36"),
    _row("RQ02", "B.1", "Long / short magnitude threshold",
         "Q75 of strictly prior absolute values, at least 12 priors", "z units", "every issue",
         "A movement activates only when it exceeds the prior 75th percentile and is positive",
         "RAW_CODE/rev363/rq02_features.py:25"),
    _row("RQ02", "B.1", "Long pair set", "L_CG and L_DSR-P", "pair id", "entry features",
         "The two long-horizon confirmation pairs", "RAW_CODE/rev363/rq02_features.py:101"),
    _row("RQ02", "B.1", "Short pair set", "all 21 short pairs", "count", "exit and bottom features",
         "Complete short-horizon pair set", "RAW_CODE/rev363/rq02_features.py:101"),
]

B3 = [
    _row("RQ02", "B.3", "Short-system proxy trigger", "n_short_sources >= 4 AND n_short_pairs >= 5", "count",
         "every issue", "Transition-based trigger, not a level threshold", "RAW_CODE/rev363/rq02.py:218"),
    _row("RQ02", "B.3", "Last-short flag age", "<= 18", "months", "after the proxy trigger",
         "The flag stays valid for eighteen months and is the input to the RQ03 lock", "RAW_CODE/rev363/rq02.py:230"),
    _row("RQ02", "B.3", "S5 vote universe", "EW, IVOL, MVP, ERC, MaxDiv", "portfolio", "every issue",
         "Five portfolio votes, three tail measures each, 30 series in total",
         "RAW_CODE/research_adapters/rq02_s5_buyhold_var_es.py:24"),
    _row("RQ02", "B.3", "S5 tail parameters", "alpha 0.95 / 0.975 / 0.99; EWMA lambda 0.94; lookback 52",
         "level / weight / weeks", "every issue", "Peak detection uses the first argmax inside (H6, T]",
         "RAW_CODE/research_adapters/rq02_s5_buyhold_var_es.py:23"),
]

B4 = [
    _row("RQ02", "B.4", "Executed entry horizon ladder", "H15 / H12 / H10 / H8 / H6 = 45 / 36 / 30 / 24 / 18",
         "months", "entry training", "Four model families are fitted at each horizon",
         "RAW_CODE/rev363/config.py:24"),
    _row("RQ02", "B.4", "Executed entry selection rule",
         "feasible if FNR < 0.10 and FPR < 0.20; then min FN, then min FP", "rate", "threshold search",
         "Operating point chosen inside the feasible set", "RAW_CODE/rev363/w02_integrity.py:244"),
    _row("RQ02", "B.5", "Executed exit horizons", "3 / 6 / 12 / 18", "months", "exit training",
         "Exit is fitted at four horizons; the phase overlay uses three months", "RAW_CODE/rev363/rq02.py:34"),
    _row("RQ02", "B.5", "Executed exit selection rule",
         "feasible if FNR < 0.05 and FPR < 0.15; 0.5 confusion weight", "rate", "threshold search",
         "Operating point chosen inside the feasible set", "RAW_CODE/rev363/rq02.py:537"),
    _row("RQ02", "B.5", "Exit-A and Exit-B", "B = A + stacked RQ01 probability", "model", "exit models",
         "Two exit models exist; the bridge that would score Exit-B is never called, so B is NOT_EVALUATED",
         "RAW_CODE/rev363/spec_bridges.py:70", "inactive"),
]

B5 = [
    _row("RQ02", "B.6", "F1 - credit and stress level",
         "ddCLIFS <= -0.15 AND zCISS >= 0.5, OR phase in {PEAK, DE-ESCALATION, RECOVERY}",
         "level / z / phase", "every issue",
         "First of three independent bottom factors; the two branches are joined by OR",
         "RAW_CODE/research_adapters/rq02_bottom_factors.py:25"),
    _row("RQ02", "B.6", "F2 - peer cooling", ">= 2 of 6 peer countries in cooling phases", "count", "every issue",
         "The own country is excluded from the peer set", "RAW_CODE/research_adapters/rq02_bottom_factors.py:57"),
    _row("RQ02", "B.6", "F3 - volatility compression",
         "sigma13 < sigma52 AND 4-week return > 0, at least 53 levels", "vol / return / weeks",
         "per asset E1, E2, E3, B1, B2", "Third factor, computed separately for each asset",
         "RAW_CODE/research_adapters/rq02_bottom_factors.py:63"),
    _row("RQ02", "B.6", "Bottom vote and zone", "F1 + F2 + F3 >= 2; zone L < U <= issue + 3 months",
         "count / months", "every issue", "Two of three factors must agree before a bottom zone is issued",
         "RAW_CODE/research_adapters/rq02_bottom_factors.py:83"),
    _row("RQ02", "B.6", "Bottom evaluation timing",
         "issue = episode start + 3 months; truth = normal + 6 months, capped at the development end", "months",
         "evaluation only", "The truth window is defined ex post, so the scored hit rate is not a trading signal",
         "RAW_CODE/research_adapters/rq02_bottom_dev_harness.py:117"),
]


def RQ01_ROWS(repo: Path) -> list[dict]:
    d = json.loads((repo / J01).read_text(encoding="utf-8"))
    rows = [
        _row("RQ01", "A.4", "Pooled score model", d["model"], "model id", "every country-month",
             "Boosted-tree probability of a stressed month; the only input to the country threshold", J01),
        _row("RQ01", "A.4", "Threshold comparison operator", d["operator"], "operator", "every country-month",
             "Stress is raised when the score is at or above the country threshold", J01),
        _row("RQ01", "A.4", "Additive joint filter", d["joint_features"]["or"][0], "packet flag",
             "both members published before the decision month",
             "Adds stress even when the country threshold is not crossed", J01),
        _row("RQ01", "A.4", "Exclusion joint filter", d["joint_features"]["not"][0], "packet flag",
             "both members published before the decision month",
             "Removes stress raised by the other two layers (domestic-versus-global divergence)", J01),
        _row("RQ01", "A.4", "Pool operator", "selected pool, else COMPLEMENT_AND_JOINT; IND_ONLY fallback", "rule",
             "every issue", "Combination of the individual and joint layers",
             "RAW_CODE/research_adapters/rq01/motif_issue_packet.py:344"),
        _row("RQ01", "A.4", "Insufficient or conflicting signals", "NOT_EVALUATED, no vote; empty complement = FAIL",
             "rule", "every issue", "Missing evidence is never imputed as calm",
             "RAW_CODE/research_adapters/rq01/pools.py:64"),
    ]
    for c, t in d["thresholds"].items():
        rows.append(_row("RQ01", "A.4", f"Country decision threshold - {c}", t, "probability",
                         f"Ps_XGB_M01 >= {t}",
                         "Country-specific cut on the pooled score; the only country-level parameter",
                         J01))
    return rows + A1 + A2 + A3 + A5


def RQ02_ROWS(repo: Path) -> list[dict]:
    d = json.loads((repo / J02).read_text(encoding="utf-8"))
    e, x = d["entry"], d["exit"]
    rows = [
        _row("RQ02", "B.4", "Registered entry model", e["model"], "model id", "reported operating point",
             "Logistic L2 model on the long-horizon confirmation set", J02),
        _row("RQ02", "B.4", "Registered entry horizon", e["horizon_months"], "months", "reported operating point",
             "Forecast window of the entry question (H6 = 18 months)", J02),
        _row("RQ02", "B.4", "Registered entry threshold", e["threshold"], "probability",
             f"p_entry >= {e['threshold']}", "Reported out-of-sample operating point", J02),
        _row("RQ02", "B.5", "Registered exit model", x["model"], "model id", "reported operating point",
             "Logistic L2 model plus a duration term", J02),
        _row("RQ02", "B.5", "Registered exit horizon", x["horizon_months"], "months", "reported operating point",
             "Forecast window of the exit question", J02),
        _row("RQ02", "B.5", "Registered exit threshold", x["threshold"], "probability",
             f"p_exit >= {x['threshold']}", "Reported operating point, also used by the RQ03 phase overlay", J02),
        _row("RQ02", "B.3", "Short-system vote count", d["s5_short_votes"]["n_votes"], "votes", "every issue",
             "Five independent votes must agree before the last-short flag is issued", J02),
    ]
    for h, t in d["horizon_register"].items():
        rows.append(_row("RQ02", "B.4", f"Horizon-register threshold - {h}", t, "probability", f"p >= {t}",
                         "Registered threshold for that horizon in the ladder; only H6 is used by the registered "
                         "entry point", J02))
    return rows + B1 + B3 + B4 + B5


DISCREPANCIES = [
    {"id": "D1", "severity": "high", "area": "RQ03",
     "finding": "The executed allocation engine (W16_S4BOT_Z2) is not part of this repository. The overlay module "
                "points to an owner hand-off tree, so the RQ03 numbers are traceable to the selected configuration "
                "dump and to the executed ledger columns, not to shipped source.",
     "evidence": "RAW_CODE/rq03_standard/overlay_rq02_exit3_approved.py:20",
     "action": "Ship the engine with the next release, or treat the parameter table as the authoritative record."},
    {"id": "D2", "severity": "high", "area": "RQ03",
     "finding": "Two rows share the label 'Pre-bottom stop' with different values (0.10 and 0.08): one governs the "
                "pre-bottom supplement, the other the general equity sleeve.",
     "evidence": "results/rq03_parameters.csv data rows 10 and 32",
     "action": "Rename both rows so that each stop names the sleeve it protects."},
    {"id": "D3", "severity": "medium", "area": "RQ03",
     "finding": "The adapter contract sets a risky-share envelope of 0.50-0.75, while the executed policy holds 0.90 "
                "equity in S1 and 1.00 in S3. The adapter is not the executed allocator.",
     "evidence": "RAW_CODE/research_adapters/rq03/contract.py:43 versus the ledger phase means",
     "action": "Scope the adapter envelope to the research branch or remove it from the public tree."},
    {"id": "D4", "severity": "medium", "area": "RQ03",
     "finding": "The requested +/-5 percentage-point band around each benchmark's base weights is not implemented. "
                "The only 5 pp object in the code is a capital-grid step used by a research audit.",
     "evidence": "W05_RQ03_PRIORITY_END_TO_END_AUDITED_NEW_FUNCTIONS_REV10_20261023.py:528",
     "action": "State that benchmarks re-solve to their exact shape each week."},
    {"id": "D5", "severity": "medium", "area": "RQ03",
     "finding": "The 2% weekly expected-shortfall cap cannot be reproduced from the ledger: the recording column is "
                "empty and three weeks are flagged as bottom-state bypass violations.",
     "evidence": "results/rq03_standard/CHOSEN_STANDARD_LEDGER_DEV.csv, columns hist_es_cap_used and total_es_status",
     "action": "Recompute the cap from the engine and re-publish the ledger flag."},
    {"id": "D6", "severity": "medium", "area": "RQ03",
     "finding": "VaR override triggers (95th and 99th percentile flags that push equity to 1.00) exist only as ledger "
                "outputs; their threshold values are not in the repository.",
     "evidence": "ledger columns override_trigger95, override_trigger99, override_target_equity",
     "action": "Publish the override thresholds or drop the claim that the rule is documented."},
    {"id": "D7", "severity": "low", "area": "RQ03",
     "finding": "The cycle-reset probability gate is 1.01, above the maximum of a probability; the branch can never "
                "fire, so the cycle is re-armed by the 26-week clock alone.",
     "evidence": "results/rq03_parameters.csv data row 39",
     "action": "Keep the window and label the probability branch inert."},
    {"id": "D8", "severity": "low", "area": "RQ02",
     "finding": "The horizon register lists 0.1594707667685234 for H6, while the registered entry point is 0.111821 "
                "for a different model (M11). Both are correct for their own model; the register is not the selected "
                "threshold.",
     "evidence": "THRESHOLD/RQ02/rq02_selected.json",
     "action": "Keep both labelled by model so they are never read as one number."},
    {"id": "D9", "severity": "low", "area": "RQ03",
     "finding": "The RQ03 selected-threshold bundle has no chosen_standard block although the loader expects one.",
     "evidence": "RAW_CODE/canonical_thresholds.py:174 versus THRESHOLD/RQ03/rq03_selected.json",
     "action": "Add the block or relax the loader with an explicit message."},
    {"id": "D10", "severity": "high", "area": "RQ01",
     "finding": "The registered identification rule cannot be computed from the tape shipped here: the columns "
                "Ps_XGB_M01, pkt_joint_CLIFS_VIX and pkt_joint_DFC_GFC are absent, so the fidelity check returns "
                "NOT_COMPUTABLE_IN_THIS_PIPELINE. The published DEV and OOS confusion comes from the owner register, "
                "not from a run a reader can repeat.",
     "evidence": "RAW_CODE/canonical_fidelity.py:33",
     "action": "Ship the score tape or publish the rule as a registered, non-reproducible result."},
    {"id": "D11", "severity": "high", "area": "RQ01",
     "finding": "Peak, Trough and Recovery are phase classifications (percentile, quorum, persistence), not "
                "probabilities. No probability threshold exists for them and the three are not a distribution.",
     "evidence": "RAW_CODE/research_adapters/rq01/five_state.py:123",
     "action": "Describe them as phases; do not report a peak or recovery probability."},
    {"id": "D12", "severity": "high", "area": "RQ02",
     "finding": "The registered entry point (M11, 0.111821) and exit point (0.027174) are echoed by the loader but "
                "are not consumed by the executed RQ02 path, which selects its own thresholds under FNR/FPR "
                "constraints.",
     "evidence": "THRESHOLD/RQ02/rq02_selected.json versus RAW_CODE/rev363/w02_integrity.py:244",
     "action": "Label the registered points as reported operating points and show the executed rule beside them."},
    {"id": "D13", "severity": "medium", "area": "RQ02",
     "finding": "Exit-B is implemented but its specification bridge is never called; the run writes NOT_EVALUATED.",
     "evidence": "RAW_CODE/rev363/rq02.py:670",
     "action": "Report Exit-B as not evaluated rather than as a result."},
    {"id": "D15", "severity": "medium", "area": "RQ02/RQ03",
     "finding": "The stated pre-onset choice is the H6 quarterly model (18 months), but the shipped RQ03 overlay "
                "script reads its H12 column when rebuilding Pn/Ps. The public presentation uses H6 quarterly per "
                "the owner's choice; the script line is recorded here so the two are never silently merged.",
     "evidence": "RAW_CODE/rq03_standard/overlay_rq02_exit3_approved.py (H12_COL usage)",
     "action": "Align the overlay column with the stated choice or document the divergence in the next release."},
    {"id": "D14", "severity": "medium", "area": "RQ02",
     "finding": "The module rev363/rq02_next.py raises PermissionError before any computation, so the published RQ02 "
                "numbers must come from rev363/rq02.py.",
     "evidence": "RAW_CODE/rev363/rq02_next.py:152",
     "action": "Remove the blocked module or mark it as a guard."},
]
