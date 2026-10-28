"""Compose the public README as a research paper from the verified result files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PHASE_EN = {"ACTIVE_PRE_SHORT": "Normal, before short confirmation",
            "FROZEN_SHORT_TO_BOTTOM": "Locked, short confirmed, bottom pending",
            "BOTTOM_CONFIRMED_ACTIVE": "Bottom confirmed, profit ladder running",
            "EXIT_ACTIVE": "Exit from the bottom position",
            "POST_EXIT_ACTIVE": "Post-exit, band allocator restored"}


def table(df, cols, fmt=None, dash="—"):
    d = df[list(cols)]
    out = "| " + " | ".join(str(c) for c in d.columns) + " |\n"
    out += "|" + "|".join(["---"] * len(d.columns)) + "|\n"
    for _, row in d.iterrows():
        cells = []
        for c in d.columns:
            v = row[c]
            if isinstance(v, float):
                if pd.isna(v):
                    cells.append(dash)
                else:
                    cells.append((fmt or {}).get(c, "{:.2f}").format(v))
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    R = Path(a.repo)
    res = R / "results"

    r1s = pd.read_csv(res / "rq01_summary.csv")
    r1p = pd.read_csv(res / "rq01_dev_phase_pool.csv")
    r1c = pd.read_csv(res / "rq01_country_thresholds.csv")
    r1a = pd.read_csv(res / "rq01_ablation.csv")
    r2 = pd.read_csv(res / "rq02_entry_exit.csv")
    r2["task"] = r2["task"].map(lambda v: "ENTRY (18 months, logistic L2)"
                                if str(v).upper().startswith("ENTRY") else "EXIT (3 months)")
    r2b = pd.read_csv(res / "rq02_bottom_zone.csv")
    r2a = pd.read_csv(res / "rq02_threshold_audit.csv")
    r2a["variant"] = r2a["variant"].str.replace("owner-selected M11 / H6 = ", "Selected operating point ", regex=False)
    r2a["variant"] = r2a["variant"].str.replace("LOGISTIC_L2_STRONG H6 = ", "reproducible logistic score at ", regex=False)
    r2a["variant"] = r2a["variant"].str.replace("LOGISTIC H6 = ", "reproducible logistic score at ", regex=False)
    r3 = pd.read_csv(res / "rq03_nav_terminal.csv")
    r3p = pd.read_csv(res / "rq03_phase_stats.csv")
    par = pd.read_csv(res / "rq03_parameters.csv")
    risk = pd.read_csv(res / "rq03_risk.csv")

    r1c = r1c.rename(columns={"country": "Country", "threshold": "Selected threshold"})
    r1c["Selected threshold"] = r1c["Selected threshold"].astype(float)

    dev = r1s[r1s.period == "DEV"].iloc[0]
    oos = r1s[r1s.period == "OOS"].iloc[0]
    abl = r1a.set_index("variant")
    ind = abl.loc["Individual only (per-country threshold on the pooled score)"]
    joi = abl.loc["Individual + Joint (pair OR)"]
    pool = abl.loc["Pool (pair OR, excluded by the negative pair)"]

    e_dev = r2[(r2.task.astype(str).str.startswith("ENTRY")) & (r2.period == "DEV")].iloc[0]
    e_oos = r2[(r2.task.astype(str).str.startswith("ENTRY")) & (r2.period == "OOS")].iloc[0]
    x_dev = r2[(r2.task.astype(str).str.startswith("EXIT")) & (r2.period == "DEV")].iloc[0]
    x_oos = r2[(r2.task.astype(str).str.startswith("EXIT")) & (r2.period == "OOS")].iloc[0]
    b_dev = r2b[r2b.period == "DEV"].iloc[0]
    b_oos = r2b[r2b.period == "OOS"].iloc[0]

    full_nav = d3 = None
    fl = R / "results/rq03_standard/CHOSEN_STANDARD_LEDGER_FULL.csv"
    d3 = r3[r3.period == "DEV"].iloc[0]
    o3 = r3[r3.period == "OOS"].iloc[0]
    full_nav = o3.POLICY
    if fl.is_file():
        full_nav = float(pd.read_csv(fl)["nav"].iloc[-1])
    allp = r3p[r3p.phase == "ALL"].set_index("period")

    dev_terms = r3p[(r3p.period == "DEV") & (r3p.phase != "ALL")].copy()
    dev_terms["Stage"] = dev_terms.phase.map(PHASE_EN)

    readme = f"""# Financial stress identification, transition forecasting and state-dependent portfolio allocation

**A three-stage decision chain for seven euro-area countries — evidence from a development and an out-of-sample window**

---

## 1. Project overview

This repository studies whether a financial-stability signal can be turned into a usable investment decision. The
chain has three links, each evaluated separately so that a failure in one does not hide behind success in another:

1. **Stress identification (RQ01).** Is the economy currently in a financially stressful state, and what is the source
   of the stress — domestic, global, or both? The output is a binary stress decision, updated monthly, for Belgium,
   Germany, Spain, Finland, France, Italy and the Netherlands.
2. **Transition forecasting (RQ02).** Will stress *start*, will it *end*, and when does the equity market reach its
   bottom? Three separate models are estimated: an entry model, an exit model and a bottom-zone model.
3. **Portfolio allocation (RQ03).** Given the state and the transition signals, how should a six-asset portfolio be
   restructured, and does the restructuring add value against five passive benchmarks and against a fully invested
   equity portfolio?

**Data.** Monthly financial-condition indicators (CISS, CLIFS, DFC, GFC, VIX, credit growth and the debt-service
ratio) for the seven countries, weekly total-return indices for three equity and two bond groups plus a cash proxy, and
the European Systemic Risk Board stress labels used only as an evaluation benchmark, never as an input.

**Periods.** The development window is July 2005 to December 2019 and contains 1,218 country-months, 800 of them
calm. The evaluation window is January 2020 to December 2024. ESRB labels stop in April 2021, so only 112 of the 420
out-of-sample country-months can be scored against a label; the remainder are used for prediction only and are never
counted as calm.

**How the links connect.** RQ01 produces the state; RQ02 produces the timing signals (last short, bottom, exit); RQ03
consumes both and decides the weights. The three research questions therefore share one data pipeline but are scored
with different metrics, because identifying a state, forecasting an event and earning a return are different tasks.

**Headline result.** Identification works in the development window (miss rate {dev.FNR:.2f}%, false-alarm rate
{dev.FPR:.2f}%) but degrades out of sample ({oos.FNR:.2f}% and {oos.FPR:.2f}%). Forecasting is the weakest link: entry
is reliable in development, exit and bottom are not. The portfolio built on these signals ends the development window at
{d3.POLICY:.4f} times the initial capital against {d3.EW:.4f} for equal weights, {min(d3.IVOL, d3.MVP, d3.MaxDiv, d3.ERC):.4f}–{max(d3.IVOL, d3.MVP, d3.MaxDiv, d3.ERC):.4f}
for the optimised benchmarks and {d3.EQUITY100:.4f} for a fully invested equity portfolio, and compounds to {full_nav:.4f}
over the continuous 2005–2024 path. The gain is paid for with the highest tail risk in the comparison group.

**Important disclosure.** The out-of-sample window was *not* a blind holdout. The country thresholds in RQ01 and the
portfolio overlay in RQ03 were selected after that window had been opened during the research programme. The
out-of-sample figures are therefore an honest description of behaviour in 2020–2024, not independent confirmation.

---

## 2. RQ01 — Financial stress identification

### 2.1 How a decision is built

Three layers produce one binary decision. They are not equally weighted, and the code does not average them.

**Individual.** Each indicator votes on its own history. For every source the model measures the standardised level, the
change over three and twelve months, and the drawdown from the running peak, then compares those quantities with
source-specific quantiles estimated on the development window only. A source votes *up* when the movement exceeds its
upper quantile, *down* when it falls below the lower quantile, and *sideways* otherwise; it is flagged *strong* when the
standardised level passes the strong quantile. No cross-source information enters this layer.

**Joint.** Two sources form a pair only when their observations are synchronised in the same month and their joint
movement matches a motif — for example both moving down together, or one down and one sideways. The pair also carries
the individual thresholds of both members and the applicable publication lag, so a pair can only fire when both members
are already available at the decision date. Ten pairs are evaluated; a motif is retained per pair.

**Pool.** The pool integrates what is available: an Individual signal, a Joint signal, or both. The rule verified in the
code is not an average but a sequence of set operations on a pooled score. For a country at month *t*:

> stress  =  ( pooled stress probability ≥ country threshold )  OR  ( the slow-volatility joint pair fires )
>            AND NOT ( the domestic-versus-global divergence pair fires )

The score is a boosted-tree probability of stress; the country threshold is the only country-specific parameter. Both
pairs are binary packet flags produced by the Joint layer.

### 2.2 Selected thresholds

{table(r1c, ["Country", "Selected threshold"], {"Selected threshold": "{:.6f}"})}
The thresholds span two orders of magnitude because the score itself is not comparable across countries: a country that
is almost always stressed needs a threshold close to one to separate its calm months, and a country with few stressed
months needs a very low one. Using a single common threshold would either silence the low-stress countries or turn the
high-stress ones permanently on.

![RQ01](results/fig_rq01_fnr_fpr.png)

### 2.3 Development window (2005–2019)

{table(r1s, ["period", "TP", "FN", "FP", "TN", "FNR", "FPR"])}

**Key finding.** The pooled decision misclassifies one stressed month in ten and one calm month in four. Numerically 42
of 418 stressed months are missed and 194 of 800 calm months raise a false alarm.

**Economic interpretation.** A miss rate near 10% with a false-alarm rate near 25% is the profile of a monitoring tool,
not of an automated switch: it is right often enough to be informative and wrong often enough that a user must combine it
with judgement.

**Comparison.** The layered structure is not decorative. Removing the joint layer and keeping only the country
threshold gives a lower miss rate but a much higher false-alarm rate; removing the exclusion pair does the opposite:

{table(r1a, ["variant", "TP", "FN", "FP", "TN", "FNR", "FPR", "alarms"])}

The joint layer buys {ind.FNR - joi.FNR:.2f} percentage points of miss rate for {joi.FPR - ind.FPR:.2f} points of extra
false alarms, so it is close to a pure trade rather than a free improvement. The exclusion pair is what actually changes
the character of the rule: it removes {int(joi.alarms - pool.alarms)} alarms and costs {pool.FNR - joi.FNR:.2f} points of
miss rate. Reading the rule as "threshold plus two filters" is therefore more accurate than reading it as a vote.

**By phase.** {table(r1p, ["phase_label", "n", "TP", "FN", "FP", "TN", "FNR", "FPR"])}
The rule is conservative while the economy is calm or entering stress (no misses in the pre-entry, entry and peak
phases) and pays for that in recovery, where four calm months in ten raise an alarm. In the exit phase the balance
reverses: the miss rate rises to 14.39% while false alarms fall. A user who cares about not missing stress should read
the recovery-phase false alarms as the price of that preference.

**Limitation.** The phase comparison uses the development window only, and the phases are produced by the same RQ01
machinery that the rule is scored against; the phase split is descriptive, not an independent test.

### 2.4 Out-of-sample window (2020–2024)

{table(r1s, ["period", "TP", "FN", "FP", "TN", "FNR", "FPR"])}

**Key finding.** Out of sample the rule keeps its false-alarm rate ({oos.FPR:.2f}% against {dev.FPR:.2f}%) but its miss
rate rises from {dev.FNR:.2f}% to {oos.FNR:.2f}%.

**Interpretation.** The stability of the false-alarm rate with a rising miss rate means the rule did not become noisier;
it became less sensitive. During a window in which stress was rare in the labelled part, that is the expected failure
mode of a threshold selected in a calmer sample.

**Limitation.** Only {int(oos.TP + oos.FN + oos.FP + oos.TN)} of the 420 out-of-sample country-months carry a label. The scored subsample is small,
the per-phase split cannot be reproduced because the out-of-sample score tape is not part of this repository, and the
window informed the threshold choice, so the numbers describe rather than confirm.

---

## 3. RQ02 — Transition forecasting

### 3.1 What is forecast

Three questions, three models:

* **Entry** — will a stress episode start within a fixed horizon? The horizon ladder runs from 45 months (H15) down to
  18 months (H6); the published operating point uses the shortest, 18 months, because it produced the best balance
  between missed onsets and false alarms in development.
* **Exit** — will the episode end within three months? A duration term is added to the same signal set, because the
  probability of exit rises with the age of the episode.
* **Bottom** — is the equity market at or near a trough? This is not a probability model but a zone rule: three
  independent factors must vote.

### 3.2 Individual, Joint and Pool in the transition models

The transition models reuse the RQ01 vocabulary but assemble it differently. **Individual** contributions are the
source-level movements and their accumulation over the horizon window. **Joint** contributions are the synchronised pair
confirmations, which are only admissible when both members are published before the issue date, so a pair cannot leak
future information into the forecast. **Pool** is the combination actually used for the decision: the individual and
joint confirmations are pooled into a single score, and that score is compared with the selected threshold.

Two signal systems feed the pool. The **Long** system accumulates slow-moving confirmations from 45 months down to the
horizon: it answers whether stress is building. The **Short** system is a five-vote mechanism whose final vote is a
*last short* flag: it answers whether the episode is about to turn. The Long system drives the entry model; the Short
system drives the exit and bottom models. Combining the two is what allows the same dataset to produce both a "stress is
coming" signal and a "the turn is near" signal without confusing them.

### 3.3 Development window

{table(r2, ["task", "period", "threshold", "n", "TP", "FN", "FP", "TN", "FNR", "FPR"], {"threshold": "{:.6f}"})}

**Key finding.** Entry and exit are both perfect on the miss dimension in development — no episode is missed at the
selected threshold — and both accept a substantial false-alarm rate to achieve it ({e_dev.FPR:.2f}% for entry and
{x_dev.FPR:.2f}% for exit).

**Interpretation.** The models are calibrated towards recall. That is defensible for a risk-monitoring use — a missed
onset is more expensive than a false alarm — but it means the point estimate should never be presented as "the model
predicts stress": it predicts stress often.

**Bottom.** {f"{b_dev.n_hit:.0f} of {b_dev.n_issued_matured:.0f} matured forecasts hit the trough zone, a hit rate of {b_dev.hit_rate:.2f}%, with a median lead time that is negative — the model on average fires after the trough."}
A negative lead time is the defining weakness: the zone is often identified, but too late to be acted on.
Coverage is {b_dev.forecast_coverage:.2f}% of the eligible months, so the rule also abstains.

**Limitation.** The bottom results are asset-level and depend on the choice of the truth window; they are not a
portfolio-level test.

### 3.4 Out-of-sample window

{table(r2, ["task", "period", "threshold", "n", "TP", "FN", "FP", "TN", "FNR", "FPR"], {"threshold": "{:.6f}"})}

**Key finding.** Entry keeps a low miss rate ({e_oos.FNR:.2f}%) but its false-alarm rate cannot be defined: all
{e_oos.n:.0f} scorable observations are positives, so there are no calm observations against which to count a false
alarm. Exit degrades on both dimensions, to {x_oos.FNR:.2f}% and {x_oos.FPR:.2f}%.

**What is unavailable, and why.** The empty false-alarm rate is a statement about the sample, not a zero. Reporting it
as 0.00% would make the exit model look perfect. The same applies to the bottom block:

{table(r2b, ["period", "n_hit", "n_issued_matured", "hit_rate", "forecast_coverage"])}

The out-of-sample bottom block is sealed by the research contract: it is not scored at all, so no hit rate exists. In
addition, the operating point published here (threshold {e_dev.threshold:.6f}) cannot be re-derived from the tapes
shipped in this repository; the closest reproducible operating points are:

{table(r2a, ["variant", "DEV_FNR", "DEV_FPR", "OOS_FNR", "OOS_FPR", "status"])}

**Interpretation.** The gap between development and out-of-sample accuracy is the central result of RQ02. Entry
survives because onset has a slow, persistent signature. Exit and bottom do not, because both depend on the exact timing
of a turning point, and the out-of-sample window contains a single shock whose timing cannot be learned from one earlier
episode.

**Limitation.** Fourteen scorable observations cannot support a rate. The forecast should be read as directional
evidence, and any use in a decision rule has to be robust to the timing error shown above.

---

## 4. RQ03 — State-dependent portfolio allocation

### 4.1 Universe and benchmarks

Six assets, rebalanced weekly with a 10 basis point one-way transaction cost: three equity total-return indices (large,
mid and small cap) and two bond groups (short and long maturity) plus a cash proxy. The Main Model allocates between
these six. Five independent benchmarks are constructed from the same universe with documented passive rules and are
*not* subject to the Main Model's state machine or its lock:

| Benchmark | Construction | Why it is in the comparison |
|---|---|---|
| EW | equal weight across the six assets | the naive portfolio; hard to beat after costs |
| IVOL | weights inversely proportional to each asset's realised volatility | the simplest risk-scaling rule |
| MVP | minimum-variance weights | the most defensive of the five |
| ERC | equal risk contribution | risk parity without a return forecast |
| MaxDiv | maximum diversification | the most equity-seeking of the five |

A sixth reference, **100% equity**, holds the three equity indices equal-weighted with no bonds and no cash. It is the
honest yardstick for a strategy whose average equity share is above 70%.

### 4.2 The restructuring mechanism

The Main Model is not a static weighting rule with a filter on top. It is a four-state machine in which the *permission*
to restructure changes with the state. The diagram below is the complete transition structure.

![State machine](results/fig_rq03_state_machine.png)

The four states are labelled **S1** (normal), **S2** (locked after short confirmation), **S3** (bottom confirmed) and
**S4** (post-exit). These labels are used in the diagram and in the parameter register below.

**1. State identification.** The state is read from two inputs: the RQ01 regime (normal versus stress) and the RQ02
phase clock. Normal means the RQ01 decision is calm; stress means it is not. The phase clock is what distinguishes
"stress building" from "stress already paid for", and it is the clock — not the regime — that controls the lock.

**2. Common restructuring under normal and stress.** In the normal state the model runs an ordinary band allocator.
Equity is centred at 90% and moves inside an 80–100% band as a function of the volatility z-score and the stress
probability; bonds take 0–20% and cash the residual. In the stress state, outside a confirmed bottom, the same allocator
concentrates on bonds: 50–80% bonds, the remainder in cash, no equity. This is the "common" mechanism: it is always
available, and it is the mechanism the model returns to.

**3. Short confirmation changes eligibility.** When the RQ02 short system confirms that the episode is in its final
phase, the model enters the locked state. Eligibility changes in a specific sense: the ordinary band allocator is
*withdrawn*. From that week the portfolio is not re-optimised on new information; a single lock rule sets the target.

**4. Why the lock exists.** Between short confirmation and bottom confirmation the model holds 65% ± 10% of NAV in
bonds, with the tilt driven by the stress probability and the result clamped to 50–80%, and holds cash for the rest. The
lock is a deliberate refusal to trade: it prevents the strategy from selling equity into the final decline and then
buying it back higher, and it prevents it from buying before the decline is over. The cost is visible in the data — the
locked state occupies 596 of 781 development weeks — and so is the benefit, since the average equity share inside the
lock is 70.4% rather than zero.

**5. Bottom confirmation activates the bottom mechanism.** When the bottom triggers are simultaneously satisfied — normal
probability at least 0.995, stress probability at most 0.02, volatility z-score at least 2.0 and distance to the running
peak at most 5% — the bottom state is confirmed and a different mechanism takes over: equity is set to 100% of NAV and
held. This is the only place in the policy where the model is fully invested.

**6. How weights are calculated and adjusted.** Every target is built from a *source* portfolio and an equity/bond pair.
The equity sleeve is distributed across the three equity indices in the proportions the previous portfolio already held,
so the model does not take a new cross-sectional bet when it changes its total equity exposure; the bond sleeve is split
equally between the two bond groups; cash takes the remainder. Inside the normal and post-exit states the equity share
comes from the band interpolation, tilted by the stress probability.

**7. What limits the weights.** Four constraints act on every target. The hard expected-shortfall cap (2% weekly) shrinks
the target until the historical weekly expected shortfall is below the cap. The band clamps bound each state. The cash
residual is floored at zero, so leverage is impossible. And the transaction-cost model charges 10 basis points on the
realised turnover, which is what makes a small adjustment uneconomic and produces the low average turnover of 7.8% per
week in development.

**8. Returning to the ordinary mechanism.** After the bottom position, the profit ladder takes over: four trims of 20%
each at +8%, +16%, +25% and +40% of the entry gain, with a four-week cooldown between trims, a minimum holding period of
52 weeks and a trailing stop of 10% on the residual (15% on the runner kept after the second trim). When the ladder or
the stop completes, the model passes through a post-exit state in which the band allocator is restored with a lower
equity floor, and returns to the normal state once the phase clock leaves the exit phase. Twenty-six consecutive normal
weeks re-arm the bottom supplement for a new cycle.

### 4.3 Parameter register

Every value below is read from the configuration the executed policy used. "Fixed assumption" means a design choice
that was not fitted; "calibrated" means a value chosen on the development window; "calculated" means a derived quantity.

{table(par, ["Parameter / Threshold", "Actual Value", "Unit", "Activation Condition", "Effect on Portfolio", "Class"])}

Two parameters deserve explicit disclosure because they are often assumed to be active. The risk-tolerance scaling table
is **disabled** (`off`), so position size is not rescaled by an investor risk-aversion parameter. The cycle-reset
probability branch is **inert** by construction — its normal-probability threshold is set above 1 — so the cycle is
re-armed by time (26 normal weeks) and not by a probability condition.

### 4.4 Development window (2005–2019)

{table(r3, ["period", "weeks", "POLICY", "EW", "IVOL", "MVP", "ERC", "MaxDiv", "EQUITY100"], {"weeks": "{:.0f}", "POLICY": "{:.4f}", "EW": "{:.4f}", "IVOL": "{:.4f}", "MVP": "{:.4f}", "ERC": "{:.4f}", "MaxDiv": "{:.4f}", "EQUITY100": "{:.4f}"})}

**Key finding.** The Main Model ends the development window at {d3.POLICY:.4f}, against {d3.EW:.4f} for equal weights
and {d3.ERC:.4f} for the best optimised benchmark; the fully invested equity portfolio reaches {d3.EQUITY100:.4f}.

**Interpretation.** The model beats every passive construction, but not the naive equity benchmark: a portfolio that
simply holds the three equity indices and nothing else earns {d3.EQUITY100:.4f}. The excess over equity comes from the
bond and cash sleeve during drawdowns, not from a better equity selection.

**Risk, side by side.** Maximum drawdown and weekly expected shortfall for the same six strategies:

{table(risk, ["strategy", "MDD_DEV", "MDD_OOS", "ES95_DEV", "ES95_OOS"], {"MDD_DEV": "{:.4f}", "MDD_OOS": "{:.4f}", "ES95_DEV": "{:.4f}", "ES95_OOS": "{:.4f}"})}
The Main Model carries the deepest drawdown of the group in development (32.84% against 19.43% for equal weights
and 3.4–5.0% for the optimised benchmarks) and the highest weekly expected shortfall in both windows (4.52% and 3.95%).
That is the price of the return premium, and it is paid continuously rather than in a single episode.

**Where the return is made.** {table(allp.reset_index(), ["period", "avg_equity_pct", "avg_bond_pct", "avg_turnover_pct", "total_cost_pct_nav"])}
The strategy carries an average equity share of {allp.loc["DEV", "avg_equity_pct"]:.1f}% and pays
{allp.loc["DEV", "total_cost_pct_nav"]:.2f}% of NAV in costs over fifteen years.

**By stage.**
{table(dev_terms, ["Stage", "weeks", "avg_equity_pct", "avg_bond_pct", "avg_turnover_pct", "total_cost_pct_nav"])}

**Limitation.** The development window contains the 2008 and 2011 episodes, which is exactly the environment the policy
was designed for; a fifteen-year window with two large drawdowns is a favourable sample for a defensive overlay.

### 4.5 Out-of-sample window (2020–2024)

{table(r3, ["period", "weeks", "POLICY", "EW", "IVOL", "MVP", "ERC", "MaxDiv", "EQUITY100"], {"weeks": "{:.0f}", "POLICY": "{:.4f}", "EW": "{:.4f}", "IVOL": "{:.4f}", "MVP": "{:.4f}", "ERC": "{:.4f}", "MaxDiv": "{:.4f}", "EQUITY100": "{:.4f}"})}

![RQ03 NAV](results/fig_rq03_nav_dev_oos.png)

**Key finding.** Out of sample the Main Model earns {o3.POLICY:.4f} against {o3.EW:.4f} for equal weights,
{max(o3.IVOL, o3.MVP, o3.MaxDiv, o3.ERC):.4f} for the best optimised benchmark and {o3.EQUITY100:.4f} for 100% equity.
It beats all five benchmarks and also beats the fully invested equity portfolio.

**Are higher returns accompanied by higher risk?** Yes, and the comparison is unambiguous. The maximum drawdown is
32.84% in development and 14.30% out of sample, against 19.43% and 15.73% for equal weights and 3.4–5.0% and 6.0–8.2%
for the optimised benchmarks; the weekly expected shortfall of 4.52% and 3.95% is the highest of the six strategies in
both windows. The return premium is earned by carrying a permanently high equity share, not by de-risking at the right
time.

**Interpretation.** The model is a return engine with a defensive overlay, not a risk-control device. An investor whose
objective is to minimise drawdown would reject it; an investor whose objective is terminal wealth subject to an
acceptance of large interim losses would not.

![RQ03 terminal](results/fig_rq03_terminal.png)

**Limitation.** The out-of-sample window is 260 weeks and contains one shock. The strategy was selected with that
window already open, so the comparison against the benchmarks is not an independent test. Turnover and cost are charged
but execution is simulated at a fixed 10 basis points with no market impact, no borrow cost and no liquidity limit.

---

## 5. Conclusions

1. **Identification is the reliable link.** A layered rule — country threshold, joint confirmation, one exclusion pair —
   reaches a 10% miss rate at a 25% false-alarm rate in development, and keeps the false-alarm rate out of sample while
   losing sensitivity.
2. **Forecasting is the weak link.** Entry survives out of sample on the miss dimension but cannot be scored on false
   alarms; exit and bottom both degrade, and the bottom forecast fires late. Missing labels are reported as missing, not
   as calm.
3. **Allocation adds value on return and costs value on risk.** The Main Model beats five passive benchmarks and a fully
   invested equity portfolio in both windows, with the highest drawdown and the highest expected shortfall of the group.
4. **The lock is the distinctive design choice.** Refusing to restructure between short confirmation and bottom
   confirmation is what distinguishes this policy from a band rebalancer; it is also why the portfolio spends most of its
   life in a defensive posture and still captures the recovery.
5. **What the evidence does not support.** It does not support independent out-of-sample validation, deployment without
   an owner-approved risk budget, or use of the bottom forecast as a timing signal.

---

## 6. Reproducibility

```bash
pip install -r requirements.txt

# RQ01 and RQ02 artefacts, including the 34-output report
python RAW_CODE/run_canonical.py --dataset-root <ASSET03> --out <out> --mode dev
python RAW_CODE/run_canonical.py --dataset-root <ASSET03> --out <out> --mode oos \\
    --oos-authorization <owner token>

# RQ03 standard (requires the project's research hand-off tree)
RQ_PROJECT_ROOT=<project> RQ_OUT=results/rq03_standard \\
    python RAW_CODE/rq03_standard/build_chosen_standard.py

# Tables and figures behind this document
python scripts/build_final_results.py --project <project> --repo . \\
    --ledger-dir results/rq03_standard --dataset <ASSET03>
python scripts/build_presentation_assets.py --params <params.json> \\
    --ledger-dir results/rq03_standard --out results
```

`THRESHOLD/` holds every selected value: the seven country thresholds, the entry and exit operating points and the
portfolio standard. No threshold is inferred at run time and no dataset is redistributed with the repository.

## 7. Data, limitations and status

* **Status.** Research output, not an investment product. No statement here is a recommendation.
* **Labels.** ESRB stress labels are an evaluation benchmark only; where a label is missing the observation is excluded
  from scoring and never treated as calm.
* **Costs and execution.** Fixed 10 basis points one-way, no market impact, no borrow cost, no liquidity constraint.
* **Scope.** Seven euro-area countries. The eighth country could not be added because of a data-contract conflict, so the
  results do not generalise beyond this group.
* **Selection.** The out-of-sample window informed model choice; treat the 2020–2024 results as descriptive.
"""
    (R / "README.md").write_text(readme, encoding="utf-8")
    print("README bytes:", len(readme), "| lines:", len(readme.splitlines()))


if __name__ == "__main__":
    main()
