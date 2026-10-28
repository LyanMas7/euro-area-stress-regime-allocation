"""Build the threshold appendix, its machine-readable inventory and its audit.

Every published row names a source. Rows sourced to a code line record the actual
text of that line, so a reader can check the claim; rows that cannot be traced to
the repository are published with an explicit status instead of a value presented
as executable fact.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SECTION_TITLES = {
    "A.1": ("A.1 Individual signal thresholds",
            "Confirmation geometry for the five native indicators. The magnitude cuts are percentiles of each "
            "country's own prior deltas, so they carry no common unit across indicators."),
    "A.2": ("A.2 Joint pair selection thresholds",
            "Gates used to decide whether a pair may enter the rule. These are not the thresholds that activate the "
            "pair afterwards; activation is in A.3."),
    "A.3": ("A.3 Joint activation thresholds",
            "Conditions applied to a selected pair at each issue. Magnitudes are member-level percentiles; counts are "
            "base-pair counts."),
    "A.4": ("A.4 Pool integration thresholds",
            "How individual and joint layers are combined and the country cuts applied to the pooled score."),
    "A.5": ("A.5 State and phase thresholds",
            "Peak, trough and recovery are phase classifications in the executed code, not probabilities. Percentiles, "
            "quorums and persistence are reported as they are implemented."),
    "B.1": ("B.1 Individual thresholds, long and short",
            "Long-horizon features feed entry; short-horizon features feed exit and the bottom rule."),
    "B.2": ("B.2 Joint pairs in RQ02",
            "RQ02 reuses the RQ01 joint specification, with the same EXACT(0,0) publication lag."),
    "B.3": ("B.3 Pool and short-system thresholds",
            "The five-vote short system and the last-short flag that feeds the allocation stage."),
    "B.4": ("B.4 Entry and exit forecasting thresholds",
            "Registered operating points are reported next to the selection rule the executed path actually uses."),
    "B.5": ("B.5 Asset-price bottom thresholds",
            "Three independent factors, a two-of-three vote and an ex-post truth window."),
    "C.1": ("C.1 Main model risk thresholds",
            "Tail limits, feasibility and trading assumptions of the executed policy."),
    "C.2": ("C.2 Benchmark construction",
            "The five passive rules, their constraints and the relative band that was requested but not implemented."),
    "C.3": ("C.3 Restructuring thresholds",
            "State targets, gates, the profit ladder and the cycle reset."),
}

INTRO = {
    "RQ01": "Identification converts five native stress indicators into one binary country-month decision. The "
            "registered rule is a country threshold on a boosted-tree probability, one joint confirmation that can "
            "only add stress and one that can only remove it.",
    "RQ02": "Forecasting asks three questions: whether stress starts, whether it ends and when the equity market "
            "reaches its bottom. Entry and exit are thresholded probabilities; the bottom is a zone rule.",
    "RQ03": "Allocation is a four-state policy. The Main Model is compared with five passive benchmarks built from the "
            "same six assets and with a fully invested equity portfolio.",
}


def clean(text) -> str:
    s = str(text).replace("\ufffd", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", s).strip()


def row(group, section, parameter, value, unit, condition, effect, source, status="active"):
    return {"group": group, "section": section, "parameter": clean(parameter), "value": clean(value),
            "unit": clean(unit), "condition": clean(condition), "effect": clean(effect),
            "source": source, "status": status}


def rq03_section(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("confirmation", "recovery", "bottom-state", "post-exit", "ladder", "tranche",
                            "holding", "trail", "runner", "stop", "cycle", "lookback", "forced", "supplement",
                            "pre-bottom", "profit")):
        return "C.3"
    return "C.1"


def rq03_rows(csv_path: Path) -> list[dict]:
    d = pd.read_csv(csv_path)
    out = []
    for i, r in d.iterrows():
        name = clean(r["Parameter / Threshold"])
        value = clean(r["Actual Value"])
        status = "active"
        if (name == "Risk-tolerance scaling" or name == "Forced bottom entry") and value.lower() == "off":
            status = "inert"
        if name == "Cycle reset normal-probability":
            status = "inert"
        if name == "Pre-bottom ES overlay" and value.upper() == "NONE":
            status = "inert"
        out.append(row("RQ03", rq03_section(name), name, value, clean(r["Unit"]),
                       clean(r["Activation Condition"]), clean(r["Effect on Portfolio"]),
                       f"results/rq03_parameters.csv data row {i + 2}", status))
    return out


def benchmark_rows() -> list[dict]:
    src = "RAW_CODE/research_adapters/rq03/benchmarks.py"
    base = [("EW", "Equal weight", "v_i = 1/5 inside the risky sleeve", 47),
            ("IVOL", "Inverse volatility", "v_i proportional to 1/sigma_i, SAMPLE_DEV estimator", 64),
            ("MVP", "Minimum variance", "argmin v'Av on the simplex, active-set and KKT", 130),
            ("ERC", "Equal risk contribution", "RC_i = 1/5 by Newton iteration", 146),
            ("MaxDiv", "Maximum diversification", "max sigma'v / sqrt(v'Av)", 137)]
    out = []
    for short, name, rule, line in base:
        out.append(row("RQ03", "C.2", f"{short} - base allocation rule", name, "rule", "every weekly rebalance",
                       f"{rule}; long-only and fully invested across the five risky sleeves",
                       f"{src}:{line}"))
    out.append(row("RQ03", "C.2", "Benchmark risk limits", "none", "limit",
                   "not part of the benchmark rule",
                   "The five benchmarks carry no VaR or ES cap and no state-dependent tolerance; the Main Model cap "
                   "does not apply to them", "RAW_CODE/research_adapters/rq03/risk_tolerance.py:198"))
    out.append(row("RQ03", "C.2", "Benchmark weight constraints", "simplex", "constraint", "every rebalance",
                   "w >= 0 and sum(w) = 1; no per-asset cap and no relative band",
                   "RAW_CODE/research_adapters/rq03/contract.py:74"))
    out.append(row("RQ03", "C.2", "Benchmark rebalancing and cost", "weekly, 10 bp one-way", "frequency / bps",
                   "weekly issue", "Full re-solve to the exact target each week on the same cost schedule",
                   f"{src}:179"))
    out.append(row("RQ03", "C.2", "Relative allocation band around base weights", "not implemented",
                   "percentage points", "requested +/-5 pp band",
                   "No such band exists in the executed or adapter code; the only 5 pp object is a research grid step",
                   "RAW_CODE/research_adapters/rq03/W05_RQ03_PRIORITY_END_TO_END_AUDITED_NEW_FUNCTIONS_REV10_20261023.py:528",
                   "unavailable"))
    return out


def verify(rows: list[dict], repo: Path) -> list[dict]:
    out = []
    for r in rows:
        src = r["source"]
        r = dict(r)
        r["source_line"] = ""
        r["check"] = "selected-configuration"
        m = re.match(r"^([^:]+\.py):(\d+)$", src)
        if m:
            path, line = repo / m.group(1), int(m.group(2))
            if not path.exists():
                r["check"] = "missing-file"
            else:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line > len(lines):
                    r["check"] = "line-out-of-range"
                else:
                    text = lines[line - 1].strip()
                    r["source_line"] = text
                    r["check"] = "verified" if text and not text.startswith("#") else "blank-line"
        out.append(r)
    return out


def md(rows: list[dict]) -> str:
    out = "| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |\n|---|---|---|---|---|\n"
    for r in rows:
        out += (f"| {r['parameter']} | {r['value']} | {r['unit']} | {r['condition']} | {r['effect']} |\n")
    return out


def html_rows(rows: list[dict]) -> str:
    cells = []
    for r in rows:
        cls = " class=\"inert\"" if r["status"] != "active" else ""
        cells.append(f"<tr{cls}><td>{r['parameter']}</td><td class=\"num\">{r['value']}</td><td>{r['unit']}</td>"
                     f"<td>{r['condition']}</td><td>{r['effect']}</td></tr>")
    return ("<div class=\"tablewrap\"><table class=\"thr\"><thead><tr><th>Parameter / Rule</th>"
            "<th>Selected value</th><th>Unit</th><th>Activation condition</th><th>Interpretation</th></tr></thead>"
            "<tbody>" + "".join(cells) + "</tbody></table></div>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    R = Path(a.repo)
    from appendix_data import RQ01_ROWS, RQ02_ROWS, DISCREPANCIES

    rows = verify(RQ01_ROWS(R) + RQ02_ROWS(R) + rq03_rows(R / "results/rq03_parameters.csv") + benchmark_rows(), R)
    problems = [r for r in rows if r["check"] not in ("verified", "selected-configuration")]
    summary = {"rows": len(rows), "verified_code_rows": sum(r["check"] == "verified" for r in rows),
               "selected_configuration_rows": sum(r["check"] == "selected-configuration" for r in rows),
               "unverified": len(problems), "active": sum(r["status"] == "active" for r in rows),
               "inert": sum(r["status"] == "inert" for r in rows),
               "unavailable": sum(r["status"] == "unavailable" for r in rows)}
    (R / "results/THRESHOLD_INVENTORY.json").write_text(
        json.dumps({"summary": summary, "discrepancies": DISCREPANCIES, "rows": rows}, indent=2), encoding="utf-8")

    md_out, html_out = [], []
    LETTER = {"RQ01": "A", "RQ02": "B", "RQ03": "C"}
    for grp in ("RQ01", "RQ02", "RQ03"):
        label = {"RQ01": "A - RQ01 Financial stress identification",
                 "RQ02": "B - RQ02 Transition forecasting",
                 "RQ03": "C - RQ03 Portfolio risk and restructuring"}[grp]
        md_out.append(f"\n### Appendix {label} thresholds\n")
        md_out.append(f"{INTRO[grp]}\n")
        html_out.append(f"<h3>Appendix {LETTER[grp]} - RQ{grp[1:]} "
                        f"{'Financial Stress Identification' if grp == 'RQ01' else 'Transition Forecasting' if grp == 'RQ02' else 'Portfolio Risk and Restructuring'} thresholds</h3>")
        html_out.append(f"<p>{INTRO[grp]}</p>")
        for sec in sorted({r["section"] for r in rows if r["group"] == grp}):
            title, blurb = SECTION_TITLES[sec]
            part = [r for r in rows if r["section"] == sec]
            md_out.append(f"\n#### {title}\n\n{blurb}\n\n{md(part)}")
            html_out.append(f"<h4>{title}</h4><p>{blurb}</p>{html_rows(part)}")

    md_out.append("\n### Threshold completeness audit\n")
    md_out.append(f"{summary['rows']} rows are published: {summary['verified_code_rows']} traced to a code line, "
                  f"{summary['selected_configuration_rows']} traced to a selected-configuration file, "
                  f"{summary['unverified']} unverified. Status counts: {summary['active']} active, "
                  f"{summary['inert']} inert, {summary['unavailable']} not implemented or not available.\n")
    md_out.append("| Id | Area | Severity | Finding | Action |\n|---|---|---|---|---|\n")
    for d in DISCREPANCIES:
        md_out.append(f"| {d['id']} | {d['area']} | {d['severity']} | {d['finding']} | {d['action']} |\n")
    html_out.append(f"<h3>Threshold completeness audit</h3><p>{summary['rows']} rows are published: "
                    f"{summary['verified_code_rows']} traced to a code line, {summary['selected_configuration_rows']} "
                    f"traced to a selected-configuration file, {summary['unverified']} unverified.</p>")
    html_out.append("<div class=\"tablewrap\"><table class=\"thr\"><thead><tr><th>Id</th><th>Area</th>"
                    "<th>Severity</th><th>Finding</th><th>Action</th></tr></thead><tbody>" +
                    "".join(f"<tr><td>{d['id']}</td><td>{d['area']}</td><td>{d['severity']}</td>"
                            f"<td>{d['finding']}</td><td>{d['action']}</td></tr>" for d in DISCREPANCIES) +
                    "</tbody></table></div>")

    (R / "results/appendix.md").write_text("".join(md_out), encoding="utf-8")
    (R / "results/appendix.html").write_text("".join(html_out), encoding="utf-8")
    print("rows:", summary)
    for p in problems:
        print("  UNVERIFIED:", p["section"], p["parameter"], "->", p["check"], p["source"])


if __name__ == "__main__":
    main()
