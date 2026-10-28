"""Generate a single-page GitHub Pages presentation from the verified result files."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

CSS = """
:root{--ink:#16202c;--muted:#5b6b7c;--line:#d8e0e8;--accent:#b1121b;--bg:#ffffff;--soft:#f4f7fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:31px;line-height:1.22;margin:0 0 10px;letter-spacing:-.01em}
h2{font-size:22px;margin:52px 0 6px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:17px;margin:28px 0 6px}
p{margin:10px 0;color:var(--ink)}
.lede{font-size:18px;color:var(--muted);margin:0 0 22px}
.meta{font-size:13px;color:var(--muted);letter-spacing:.02em;text-transform:uppercase}
.k{font-weight:600;color:var(--accent)}
table{border-collapse:collapse;width:100%;margin:14px 0 8px;font-size:14px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{border-top:1.6px solid var(--ink);border-bottom:1.2px solid var(--ink);font-weight:600}
tbody tr:last-child td{border-bottom:1.6px solid var(--ink)}
figure{margin:22px 0 6px}
img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
figcaption{font-size:13px;color:var(--muted);margin-top:8px}
.callout{background:var(--soft);border-left:4px solid var(--accent);padding:14px 16px;border-radius:4px;margin:18px 0;font-size:15px}
.note{font-size:14px;color:var(--muted)}
a{color:#0b5cad}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:16px 0}
.card{background:var(--soft);border-radius:6px;padding:14px 16px}
.card b{display:block;font-size:22px;margin-bottom:2px}
.card span{font-size:13px;color:var(--muted)}
code{background:var(--soft);padding:1px 5px;border-radius:3px;font-size:94%}
"""


def table(df, cols, fmt=None):
    out = '<div class="tablewrap"><table class="thr"><thead><tr>' + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"
    for _, row in df.iterrows():
        out += "<tr>"
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cell = "—" if pd.isna(v) else (fmt or {}).get(c, "{:.2f}").format(v)
            else:
                cell = str(v)
            out += f"<td>{cell}</td>"
        out += "</tr>"
    return out + "</tbody></table></div>"


EXTRA_CSS = """
.gh{margin:2px 0 6px;text-align:left}
.gh a{display:inline-flex;align-items:center;gap:6px;color:#24292f;text-decoration:none;font-size:.78rem}
.gh a:hover{text-decoration:underline}
figure{margin:14px 0 18px}
figure img{width:100%;height:auto;display:block;border:1px solid #e5e5e5}
figcaption{font-size:.85rem;color:#333;margin-top:6px;line-height:1.35}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:10px 0 16px;border:1px solid #e5e5e5}
table.thr{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff}
table.thr th{background:#f2f2f2;text-align:left;font-weight:600;padding:6px 7px;border-bottom:1px solid #ccc;white-space:nowrap}
table.thr td{padding:5px 7px;border-bottom:1px solid #eee;vertical-align:top}
table.thr td.num{white-space:nowrap;font-variant-numeric:tabular-nums}
table.thr tr.inert td{color:#666;background:#fafafa}
h4{margin:18px 0 6px;font-size:1rem}
@media(max-width:640px){table.thr{font-size:12px}figure{margin:10px 0 14px}}
"""

def fig_block(num, title, path, reading, limitation):
    return ('<figure><a href="' + path + '"><img src="' + path + '" alt="' + title + '"></a>'
            '<figcaption><b>Figure ' + str(num) + '.</b> ' + title + '</figcaption></figure>'
            '<p><b>Interpretation.</b> ' + reading + '</p>'
            '<p><b>Limitation.</b> ' + limitation + '</p>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    R = Path(a.repo); res = R / "results"; docs = R / "docs"
    (docs / "assets").mkdir(parents=True, exist_ok=True)
    for f in res.glob("*.png"):
        shutil.copy2(f, docs / "assets" / f.name)

    r1s = pd.read_csv(res / "rq01_summary.csv")
    r1a = pd.read_csv(res / "rq01_ablation.csv")
    r2 = pd.read_csv(res / "rq02_entry_exit.csv")
    r2["task"] = r2["task"].map(lambda v: "ENTRY (18m)" if str(v).upper().startswith("ENTRY") else "EXIT (3m)")
    r2b = pd.read_csv(res / "rq02_bottom_zone.csv")
    r3 = pd.read_csv(res / "rq03_nav_terminal.csv")
    par = pd.read_csv(res / "rq03_parameters.csv")
    risk = pd.read_csv(res / "rq03_risk.csv")

    d3 = r3[r3.period == "DEV"].iloc[0]; o3 = r3[r3.period == "OOS"].iloc[0]
    dev = r1s[r1s.period == "DEV"].iloc[0]; oos = r1s[r1s.period == "OOS"].iloc[0]

    appendix_html = (R / "results/appendix.html").read_text(encoding="utf-8")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stress identification, transition forecasting and state-dependent allocation</title>
<style>{CSS}{EXTRA_CSS}</style></head><body><div class="wrap">

<div class="gh">
<a href="https://github.com/LyanMas7/rq01-rq02-rq03-research-code/blob/main/cv/Nguyen_Hoang_Lam_TikTok_AI_Model_Data_Operation.pdf"
   target="_blank" rel="noopener" aria-label="Open the CV on GitHub">
<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" focusable="false">
<path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
</svg>
<span>CV - Nguyen Hoang Lam</span></a></div>

<div class="meta"><div class="meta">Research summary · seven euro-area countries · 2005–2024</div>
<h1>Financial stress identification, transition forecasting and state-dependent portfolio allocation</h1>
<p class="lede">A three-stage decision chain, evaluated separately at each stage so that a failure in one link cannot hide
behind success in another.</p>

<div class="grid">
  <div class="card"><b>{dev.FNR:.2f}% / {dev.FPR:.2f}%</b><span>RQ01 miss / false-alarm rate, development window</span></div>
  <div class="card"><b>{oos.FNR:.2f}% / {oos.FPR:.2f}%</b><span>RQ01 miss / false-alarm rate, out of sample</span></div>
  <div class="card"><b>{d3.POLICY:.4f}</b><span>Main model NAV, development window</span></div>
  <div class="card"><b>{o3.POLICY:.4f}</b><span>Main model NAV, out of sample</span></div>
</div>

<h2>1 · Project overview</h2>
<p>The chain answers three questions in sequence. <b>RQ01</b> asks whether the economy is currently in a financially
stressful state and where the stress comes from — domestic, global or both. <b>RQ02</b> asks whether stress will start,
whether it will end, and when the equity market reaches its bottom. <b>RQ03</b> asks how a six-asset portfolio should be
restructured given those signals, and whether the restructuring adds value against five passive benchmarks and a fully
invested equity portfolio.</p>
<p><b>Data.</b> Monthly financial-condition indicators for Belgium, Germany, Spain, Finland, France, Italy and the
Netherlands; weekly total-return indices for three equity and two bond groups plus a cash proxy; European Systemic Risk
Board stress labels, used only to score the result, never as an input.</p>
<p><b>Periods.</b> Development runs from July 2005 to December 2019 (1,218 country-months, 800 calm). Evaluation runs from
January 2020 to December 2024; labels stop in April 2021, so only 112 of 420 out-of-sample months can be scored. Months
without a label are excluded from scoring and are never treated as calm.</p>
<div class="callout"><b>Disclosure.</b> The out-of-sample window was not a blind holdout: the country thresholds and the
portfolio overlay were selected after that window had been opened. The 2020–2024 figures describe behaviour in that window;
they are not independent confirmation.</div>

<h2>2 · RQ01 — identifying financial stress</h2>
<p>Three layers build one binary decision, and they are not equally weighted. <b>Individual</b> signals compare each
indicator's standardised level, its three- and twelve-month change and its drawdown with source-specific quantiles, and
vote up, down or sideways; a source is flagged strong when its level passes the strong quantile. <b>Joint</b> signals pair
two indicators that are observed in the same month and whose movements match a motif, carrying both members' thresholds
and the applicable publication lag. <b>Pool</b> integrates what is available and then applies the final rule — a country
threshold on the pooled score, one joint confirmation that can only add stress, and one joint confirmation that can only
remove it.</p>
{table(r1s, ["period", "TP", "FN", "FP", "TN", "FNR", "FPR"], {'TP': '{:.0f}', 'FN': '{:.0f}', 'FP': '{:.0f}', 'TN': '{:.0f}', 'FNR': '{:.2f}%', 'FPR': '{:.2f}%'})}
<p>The layered structure is not decorative. Keeping only the country threshold lowers the false-alarm rate to 15.00% and
raises the miss rate to 5.74%; adding the confirmation pair buys {r1a.iloc[1].FNR:.2f}% miss rate for a much higher
false-alarm rate; the exclusion pair then removes 27 alarms and costs 5.03 points of miss rate. The rule behaves as
"threshold plus two filters", not as a vote.</p>
<figure><a href="assets/fig_ijp01_rq01_individual.png"><img src="assets/fig_ijp01_rq01_individual.png" alt="Individual Signal - Six-Month Observation Window"></a><figcaption>Individual Signal - Six-Month Observation Window</figcaption></figure><p><b>Interpretation.</b> Seven monthly observations create up to 21 pairwise comparisons and one endpoint difference; a comparison confirms only when it is positive and reaches a magnitude cut, and the vote needs one strong movement or a weak movement backed by half of the comparisons.</p><p><b>Limitation.</b> The cuts are percentiles of each country's own prior deltas, so they carry no common unit across indicators.</p>
<figure><a href="assets/fig_ijp02_rq01_joint.png"><img src="assets/fig_ijp02_rq01_joint.png" alt="Joint Signal - Two Synchronized Indicators"></a><figcaption>Joint Signal - Two Synchronized Indicators</figcaption></figure><p><b>Interpretation.</b> Each member keeps its own magnitude cut; paired movements are counted on synchronized bases with no publication lag. Selection gates are applied once; activation gates run at every issue afterwards.</p><p><b>Limitation.</b> Insufficient data yields NOT_EVALUATED, never a calm vote.</p>
<figure><a href="assets/fig_ijp03_rq01_pool.png"><img src="assets/fig_ijp03_rq01_pool.png" alt="Pool Integration - From Signals to a Stress Decision"></a><figcaption>Pool Integration - From Signals to a Stress Decision</figcaption></figure><p><b>Interpretation.</b> The pool is not an average and not a vote: a boosted-tree probability is compared with the country threshold, one joint packet can only add stress and one can only remove it.</p><p><b>Limitation.</b> The removal packet can cancel stress raised by either other layer.</p>

{table(r1a, ["variant", "TP", "FN", "FP", "TN", "FNR", "FPR", "alarms"], {'TP': '{:.0f}', 'FN': '{:.0f}', 'FP': '{:.0f}', 'TN': '{:.0f}', 'FNR': '{:.2f}%', 'FPR': '{:.2f}%', 'alarms': '{:.0f}'})}
<figure><a href="assets/fig01_rq01_phase_error.png"><img src="assets/fig01_rq01_phase_error.png" alt="Identification error by phase"></a>
<figcaption>Financial Stress Identification - DEV vs OOS</figcaption></figure>
<p><b>Interpretation.</b> The rule is silent in the calm, entry and peak phases and pays for that silence in recovery, where
43.78% of calm months raise an alarm. The pooled 10.05% and 24.25% average a rule that is either perfectly quiet or strongly
over-sensitive.</p>
<p><b>Limitation.</b> A development-window diagnostic; the out-of-sample score tape needed to repeat it is not part of this
release.</p>
<p>The rule is conservative while the economy is calm or entering stress — no misses in the pre-entry, entry and peak
phases — and pays for that in recovery, where 43.8% of calm months raise an alarm. Out of sample the false-alarm rate is
stable ({oos.FPR:.2f}%) while the miss rate rises to {oos.FNR:.2f}%: the rule lost sensitivity, not discipline.
Only 112 labelled months support that statement, and the per-phase split cannot be reproduced because the out-of-sample
score tape is not published here.</p>

<h2>3 · RQ02 — forecasting transitions and the bottom</h2>
<p>Entry asks whether stress starts within a fixed horizon; the horizon ladder runs from 45 down to 18 months and the
published operating point uses 18. Exit asks whether the episode ends within three months and adds a duration term,
because the probability of exit rises with the age of the episode. The bottom is not a probability model but a zone rule
that requires independent factors to agree. In all three, <b>Individual</b> movements are accumulated over the horizon,
<b>Joint</b> confirmations are only admitted when both members are published before the issue date, and <b>Pool</b> is the
combination actually thresholded. A slow <i>Long</i> system feeds entry; a five-vote <i>Short</i> system, whose last vote is
a last-short flag, feeds exit and bottom.</p>
<figure><a href="assets/fig_ijp04_rq02_short.png"><img src="assets/fig_ijp04_rq02_short.png" alt="RQ02 Short Signal - The H15 to H12 Observation Interval"></a><figcaption>RQ02 Short Signal - The H15 to H12 Observation Interval</figcaption></figure><p><b>Interpretation.</b> The representative short interval is the nine months between the 45-month and 36-month milestones. The comparison count depends on the indicator's frequency; the numbers shown are structural capacity, not confirmations.</p><p><b>Limitation.</b> The count is computed on eligible bases only.</p>
<figure><a href="assets/fig_ijp05_rq02_long.png"><img src="assets/fig_ijp05_rq02_long.png" alt="RQ02 Long Signal - Progressive Milestones H15 to H6"></a><figcaption>RQ02 Long Signal - Progressive Milestones H15 to H6</figcaption></figure><p><b>Interpretation.</b> The long system accumulates confirmations at five milestones. Each later milestone adds information to an already-issued forecast; it never rewrites what an earlier issue would have seen.</p><p><b>Limitation.</b> Forecasts use data published by the issue date only.</p>
<figure><a href="assets/fig_ijp06_rq02_joint.png"><img src="assets/fig_ijp06_rq02_joint.png" alt="RQ02 Joint Confirmation - Long and Short Pairs"></a><figcaption>RQ02 Joint Confirmation - Long and Short Pairs</figcaption></figure><p><b>Interpretation.</b> Long pairs feed the entry pool; the short pair set feeds exit and the bottom factors. Paired bases are counted only when both members pass their own cuts and both are published before the issue.</p><p><b>Limitation.</b> A carried quarter is never counted twice.</p>

{table(r2, ["task", "period", "threshold", "n", "TP", "FN", "FP", "TN", "FNR", "FPR"], {'n': '{:.0f}', 'TP': '{:.0f}', 'FN': '{:.0f}', 'FP': '{:.0f}', 'TN': '{:.0f}', 'FNR': '{:.2f}%', 'FPR': '{:.2f}%', 'threshold': '{:.6f}'})}
<figure><a href="assets/fig02_rq02_transition_error.png"><img src="assets/fig02_rq02_transition_error.png" alt="Entry and exit error and bottom-zone hit rate"></a>
<figcaption>Entry and Exit Forecasting Performance</figcaption></figure>
<p><b>Interpretation.</b> Entry buys its low miss rate with false alarms in development and cannot be scored on false alarms out
of sample, where all 14 observations are positives. Exit degrades to 28.57% and 45.71%. The bottom zone keeps the trough inside
the zone in one development forecast in four and one evaluation forecast in fifteen.</p>
<p><b>Limitation.</b> Two of the five rates are undefined rather than zero, and the evaluation block rests on 14, 91 and 90
scorable observations.</p>
<p>Both models are calibrated towards recall: no episode is missed in development, at the cost of a false alarm in
roughly one in four calm observations. Out of sample entry keeps a low miss rate (14.29%) but its false-alarm rate
<b>cannot be defined</b> — all 14 scorable observations are positives, so an empty denominator is reported as missing,
never as zero. Exit degrades on both dimensions (28.57% and 45.71%). The bottom rule hits
{r2b[r2b.period == 'DEV'].iloc[0].n_hit:.0f} of {r2b[r2b.period == 'DEV'].iloc[0].n_issued_matured:.0f} matured forecasts in development (23.33%) with a negative median lead
time, meaning it identifies the zone but on average after the trough. Out of sample the block is scored: 6 of 90 matured
forecasts contained the trough (6.67%), against 14 of 60 (23.33%) in development.</p>

<h2>4 · RQ03 — state-dependent allocation</h2>
<p>The Main Model is a four-state machine in which the <i>permission</i> to restructure changes with the state. Normal
(S1) runs a band allocator: equity centred at 90% inside an 80–100% band driven by the volatility z-score and the stress
probability, bonds 0–20%, cash the rest. Stress without a confirmed bottom (S2) withdraws that allocator and holds
50–80% bonds, the remainder in cash, with no equity. Short confirmation is what triggers the lock: between that week and
bottom confirmation the portfolio is not re-optimised on new information, which prevents selling into the final decline
and buying back higher, at the cost of 596 of 781 development weeks spent in that posture. Bottom confirmation (S3)
activates the only fully invested state — equity 100% of NAV, held — after which a profit ladder of four 20% trims at
+8%, +16%, +25% and +40%, a four-week cooldown, a 52-week minimum hold and a 10–15% trail returns the portfolio through
the post-exit band (S4) to normal, with 26 consecutive normal weeks re-arming the cycle.</p>
<figure><a href="assets/fig03_rq03_restructuring_flow.png"><img src="assets/fig03_rq03_restructuring_flow.png" alt="Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom"></a><figcaption>Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom</figcaption></figure><p><b>Interpretation.</b> RQ01 identifies the state, RQ02 supplies Pn and Ps from the H6 quarterly pre-onset model and the 3-month exit model, and RQ03 applies the risk cap, volatility and z-score constraints. The portfolio is locked between last short and bottom confirmation, so a favourable signal produces no trade while the lock is active.</p><p><b>Limitation.</b> Bottom confirmation does not guarantee a positive return: the realised equity trough arrived 0 to 20 weeks after the model's confirmation across the seven episodes.</p><div class="tablewrap"><table class="thr"><thead><tr><th>Phase</th><th>RQ01</th><th>RQ02</th><th>RQ03</th><th>Allocation</th></tr></thead><tbody><tr><td>Normal</td><td>pooled rule calm</td><td>P_S = weekly hazard of the H6 quarterly pre-onset probability</td><td>equity band 80-100%</td><td>equity 80-100%, bonds 0-20%, cash residual</td></tr><tr><td>Last short</td><td>state persists</td><td>five-vote short system fires; P_N = 1 - P_S</td><td>lock: band allocator withdrawn</td><td>bonds 50-80%, no equity, restructuring prohibited</td></tr><tr><td>Bottom</td><td>state persists</td><td>gates P_N &gt;= 0.995, P_S &lt;= 0.02, z &gt;= 2, drawdown &lt;= 5%; F1-F3 two of three</td><td>ES cap bypassed in S3</td><td>equity 100%, profit ladder active</td></tr><tr><td>Post-bottom</td><td>state updated</td><td>exit 3M p &gt;= 0.027174 or ladder complete</td><td>S4 band</td><td>equity 50-100%, bonds 0-25%; 26 normal weeks re-arm the cycle</td></tr></tbody></table></div><p><b>Chronology check.</b> Measured on the executed ledger, the realised equity trough arrived between 0 and 20 weeks after the model's bottom confirmation across the seven episodes; in 2007-09 and 2022 the confirmation preceded the trough by 18 and 15 weeks. The restructuring lock has lasted between 18 and 300 weeks. Confirmation dates are model-issued decisions; trough dates are retrospective market events.</p>
<p>Four constraints act on every target: a hard weekly expected-shortfall cap of 2%, the state band clamps, a zero floor on
cash (so no leverage), and a 10 basis point one-way transaction cost on realised turnover. Two parameters are often
assumed to be active and are not: the risk-tolerance scaling table is disabled, and the probability branch of the cycle
reset is inert by construction, so the cycle is re-armed by time rather than by a probability condition.</p>
<figure><a href="assets/fig04_rq03_allocation.png"><img src="assets/fig04_rq03_allocation.png" alt="Portfolio composition and regime evolution, 2005-2024"></a>
<figcaption>Six-Asset Capital Allocation, 2005-2024</figcaption></figure>
<p><b>Interpretation.</b> The policy is equity-heavy by design: the three equity sleeves hold about three quarters of NAV for most
of the period, cash reaches 100% in ten weeks only, and the regime band shows a cycle rather than a one-way de-risking. The lock
accounts for 783 of 1,041 weeks.</p>
<p><b>Limitation.</b> Weekly weights are re-based to 100% for display, and the band records the states the executed configuration
produced.</p>
<h3>Parameter register</h3>
{table(par, ["Parameter / Threshold", "Actual Value", "Unit", "Class"])}
<figure><a href="assets/fig05_rq03_nav_paths.png"><img src="assets/fig05_rq03_nav_paths.png" alt="Main model NAV against five benchmarks and full equity"></a>
<figcaption>Main Model NAV Against Five Benchmarks and Full Equity</figcaption></figure>
<p><b>Interpretation.</b> The main model compounds 3.9015 in development and 1.8059 in evaluation, against 2.5960 and 1.2611 for
full equity and 1.678/1.087 for equal weights. The gap opens in the drawdowns, where the policy keeps equity and the benchmarks
do not.</p>
<p><b>Limitation.</b> Development benchmarks are terminal values because only end-of-window levels exist; the evaluation window
was open when the model was chosen.</p>
{table(r3, ["period", "weeks", "POLICY", "EW", "IVOL", "MVP", "ERC", "MaxDiv", "EQUITY100"],
       {"weeks": "{:.0f}", "POLICY": "{:.4f}", "EW": "{:.4f}", "IVOL": "{:.4f}", "MVP": "{:.4f}", "ERC": "{:.4f}", "MaxDiv": "{:.4f}", "EQUITY100": "{:.4f}"})}
<p>The model beats all five passive benchmarks and the fully invested equity portfolio in both windows: 3.9015 against
1.2934–1.6784 and 2.5960 in development, 1.8059 against 0.9892–1.0867 and 1.2611 out of sample, compounding to 5.6981
over the continuous path. The premium is paid for with risk: maximum drawdown of 32.84% and 14.30%, and the highest weekly
expected shortfall of the six strategies in both windows. This is a return engine with a defensive overlay, not a
risk-control device. Benchmarks are passive construction rules and are not subject to the lock.</p>
{table(risk, ["strategy", "MDD_DEV", "MDD_OOS", "ES95_DEV", "ES95_OOS"], {"MDD_DEV": "{:.4f}", "MDD_OOS": "{:.4f}", "ES95_DEV": "{:.4f}", "ES95_OOS": "{:.4f}"})}
<figure><a href="assets/fig06_rq03_terminal_nav.png"><img src="assets/fig06_rq03_terminal_nav.png" alt="Terminal NAV by strategy and window"></a>
<figcaption>Terminal NAV by Strategy and Window</figcaption></figure>
<p><b>Interpretation.</b> Terminal NAV ranks the main model first in both windows, with full equity second in both and the five
passive rules near one.</p>
<p><b>Limitation.</b> Terminal NAV ignores the path: the premium is earned with the deepest drawdown and the highest weekly
expected shortfall of the group.</p>

<h2>5 · Conclusions</h2>
<p><b>Identification is the reliable link:</b> 10% miss rate at a 25% false-alarm rate in development, with a stable
false-alarm rate out of sample. <b>Forecasting is the weak link:</b> entry survives on the miss dimension but cannot be
scored on false alarms, and exit and bottom degrade. <b>Allocation adds value on return and costs value on risk.</b>
<b>The lock is the distinctive design choice</b> — refusing to restructure between short confirmation and bottom
confirmation is what separates this policy from a band rebalancer. The evidence does not support independent
out-of-sample validation, deployment without an owner-approved risk budget, or use of the bottom forecast as a timing
signal.</p>

<h2>6 · Reproducibility and limitations</h2>
<p>Code, selected thresholds and the exact tables behind every figure are in the repository:
<code>README.md</code>, <code>RAW_CODE/</code>, <code>THRESHOLD/</code>, <code>results/</code>. The dataset is
not redistributed. Costs are fixed at 10 basis points one-way with no market impact, no borrow cost and no liquidity
constraint; the sample covers seven countries and one out-of-sample shock.</p>
<p class="note">Research output, not an investment recommendation.</p>
<h2>7 · Complete threshold and assumption appendix</h2>
<p>Every active threshold, rule and parameter used by the three research questions, in the standard form Parameter / Selected value / Unit / Activation condition / Interpretation. Values are copied from the executed configuration or the selected-threshold files and nothing is rounded or inferred. Rows whose rule lives in code cite the file and line. The audit at the end reports every value that could not be traced, every duplicated label and every requested object that is not implemented.</p>
{appendix_html}
</div></body></html>"""
    (docs / "index.html").write_text(html, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print("pages bytes:", len(html), "| assets:", len(list((docs / 'assets').glob('*.png'))))


if __name__ == "__main__":
    main()
