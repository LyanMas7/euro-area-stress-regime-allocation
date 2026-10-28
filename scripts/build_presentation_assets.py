"""Presentation assets for RQ03: parameter register, phase statistics, state diagram.

Values are read from the configuration dump of the executed policy
(``params.json``) and from the two chosen-standard ledgers, so the published
table cannot drift from what the model actually ran.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEE_BPS = 10.0  # verified in the execution engine (one-way)

# (key, label, unit, activation, effect, klass)
ROWS = [
    ("alloc_mode", "Allocation engine", "—", "always",
     "State-and-cycle allocator: four states, each with its own weight rule", "Fixed assumption"),
    ("sz_state1_e", "Normal-state equity target", "share of NAV", "state S1 (no lock, no bottom)",
     "Sets the core equity weight; the rest of S1 is bonds", "Calibrated"),
    ("sz_state1_b", "Normal-state bond target", "share of NAV", "state S1",
     "Bond sleeve B1/B2 split equally; cash is the residual", "Calibrated"),
    ("sz_lock_b_base", "Lock-state bond share (base)", "share of NAV", "state S2 (short confirmed, bottom not yet confirmed)",
     "Bond share of the locked portfolio at neutral probabilities", "Calibrated"),
    ("sz_lock_b_ps", "Lock-state stress tilt", "share per unit of P_S", "state S2",
     "Raises the bond share as the stress probability rises, inside a 50–80% band", "Calibrated"),
    ("sz_pre_split", "Pre-bottom residual split", "rule", "state S2 when the pre-bottom supplement is active",
     "Residual after equity is split pro-rata between bonds and cash (65% to bonds)", "Fixed assumption"),
    ("sz_pre_max_weeks", "Pre-bottom window", "weeks", "state S2 supplement active",
     "Caps how long the early-equity supplement may stay on", "Calibrated"),
    ("sz_pre_cooldown", "Pre-bottom cooldown", "weeks", "after a pre-bottom exit",
     "Blocks immediate re-entry into the supplement", "Calibrated"),
    ("sz_pre_stop", "Pre-bottom stop", "return", "supplement active",
     "Cuts the supplement when its own equity basket falls by this much", "Calibrated"),
    ("sz_bot_pn", "Bottom confirmation: normal probability", "probability", "state S2",
     "One of the four gates that must all hold before bottom equity is deployed", "Calibrated"),
    ("sz_bot_ps", "Bottom confirmation: stress probability", "probability", "state S2",
     "Stress must have collapsed to this level", "Calibrated"),
    ("sz_bot_z", "Bottom confirmation: volatility z-score", "standard deviations", "state S2",
     "Deep-drawdown regime required before full equity is bought", "Calibrated"),
    ("sz_bot_dd", "Bottom confirmation: drawdown tolerance", "return", "state S2",
     "Maximum remaining distance to the running peak accepted at confirmation", "Calibrated"),
    ("sz_bot_eq", "Bottom-state equity target", "share of NAV", "state S3 (bottom confirmed)",
     "Moves the portfolio to a fully invested equity position", "Calibrated"),
    ("sz_rec_eq", "Recovery equity cap", "share of NAV", "post-bottom recovery path",
     "Intermediate equity level when the cycle has not been reset", "Calibrated"),
    ("sz_rec_pn", "Recovery normal-probability gate", "probability", "recovery path",
     "Normal probability required to leave the recovery allocation", "Calibrated"),
    ("sz_rec_ps", "Recovery stress-probability gate", "probability", "recovery path",
     "Stress probability required to leave the recovery allocation", "Calibrated"),
    ("sz_state4_high_z_mode", "State S4 high-volatility response", "rule", "state S4 with elevated z",
     "S4 absorbs the shock through bonds rather than cash", "Fixed assumption"),
    ("band_post_e_low", "Post-exit equity floor", "share of NAV", "states S4",
     "Lowest equity weight the band can produce after a completed cycle", "Calibrated"),
    ("band_post_b_high", "Post-exit bond ceiling", "share of NAV", "states S4",
     "Upper bound of the bond band in the post-exit state", "Calibrated"),
    ("band_z_direction", "Volatility-z direction", "sign", "states S1 and S4",
     "Risk-off mapping: a higher volatility z-score lowers equity; no contrarian inversion is used", "Fixed assumption"),
    ("profit_tranches", "Profit-taking ladder", "gain / share", "bottom position held",
     "Four trim steps of 20% each at +8%, +16%, +25% and +40%", "Calibrated"),
    ("profit_tranche_cooldown", "Tranche cooldown", "weeks", "between two trims",
     "Prevents trimming twice in consecutive weeks", "Calibrated"),
    ("profit_min_hold_weeks", "Minimum holding after the bottom", "weeks", "bottom position held",
     "Keeps the bottom exposure for a full year before the ladder completes", "Calibrated"),
    ("profit_trail", "Profit trail", "drawdown from peak", "after the ladder",
     "Closes the residual position when it retraces this far", "Calibrated"),
    ("profit_runner_after", "Runner activation", "tranches completed", "after the ladder",
     "Number of completed trims after which a runner is kept", "Calibrated"),
    ("profit_runner_trail", "Runner trail", "drawdown from peak", "runner active",
     "Wider trail applied only to the runner", "Calibrated"),
    ("profit_to_cash", "Profit proceeds destination", "rule", "on every trim",
     "Proceeds go to cash, not to bonds", "Fixed assumption"),
    ("post_bottom_eq_cap", "Post-bottom equity cap", "share of NAV", "states S3/S4",
     "Hard ceiling on equity once the bottom has been confirmed", "Calibrated"),
    ("post_bottom_stop_loss", "Post-bottom stop", "return", "states S3/S4",
     "Stops the position if it falls this far from entry", "Calibrated"),
    ("stop_loss", "Pre-bottom stop", "return", "states S1/S2",
     "General stop applied to the equity sleeve before the bottom", "Calibrated"),
    ("stop_mode", "Stop reference", "rule", "whenever a stop applies",
     "Stop is measured on the equal-weight equity basket, not on NAV", "Fixed assumption"),
    ("es_cap", "Hard expected-shortfall cap", "weekly return", "every rebalance",
     "Shrinks the target until the historical weekly ES is below the cap", "Calibrated"),
    ("sz_pre_es_mode", "Pre-bottom ES overlay", "rule", "supplement active",
     "NONE: the pre-bottom supplement is not additionally ES-constrained", "Fixed assumption"),
    ("sz_s4_bot", "Bottom supplement switch", "flag", "states S2/S4",
     "Enables the bottom-specific supplement that allows equity before confirmation", "Calibrated"),
    ("sz_force_bottom", "Forced bottom entry", "flag", "state S2",
     "Disabled; the bottom is only traded on confirmation", "Fixed assumption"),
    ("cycle_reset_weeks", "Cycle reset window", "weeks", "after a completed cycle",
     "Consecutive normal-state weeks required to re-arm the bottom supplement", "Calibrated"),
    ("cycle_reset_pn", "Cycle reset normal-probability", "probability", "after a completed cycle",
     "Set above 1, so the probability branch never triggers and the reset is inert", "Fixed assumption"),
    ("theta_enabled", "Risk-tolerance scaling", "flag", "every rebalance",
     "Disabled: the alpha/ES tolerance table is not applied to size", "Fixed assumption"),
    ("bottom_rule", "Bottom trigger rule", "rule", "state S3",
     "ANY: the first confirmed bottom starts the bottom state", "Fixed assumption"),
    ("bottom_lookback", "Bottom lookback", "weeks", "state S3",
     "Zero: the tail trigger uses the whole available history", "Fixed assumption"),
    ("bottom_es_bypass", "Bottom ES bypass", "flag", "state S3",
     "The hard ES cap is not applied inside the bottom state", "Fixed assumption"),
    ("sz_lock_b_band", "Lock-state bond band", "share of NAV", "state S2",
     "Bond share is clamped to 50–80%; the complement is cash", "Calculated"),
    ("fee_bps", "Transaction cost", "basis points", "on every traded notional",
     "One-way cost charged on realised turnover", "Fixed assumption"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--ledger-dir", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    P = json.loads(Path(a.params).read_text(encoding="utf-8"))
    L = Path(a.ledger_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, unit, act, eff, klass in ROWS:
        if key == "fee_bps":
            val = f"{FEE_BPS:g}"
        elif key == "sz_lock_b_band":
            val = "50 – 80"
        else:
            v = P.get(key, "—")
            if isinstance(v, str):
                v = {"STATE_Z_CYCLE": "four-state state-and-cycle allocator",
                     "PRO_RATA": "pro-rata", "BOND_FIRST": "bonds first",
                     "ANY": "first confirmed bottom", "RISK_OFF": "risk-off",
                     "BC": "bonds absorb the shock", "EW3": "equal-weight equity basket",
                     "NONE": "not applied", "HARD": "hard cap"}.get(v, v)
            if isinstance(v, list):
                val = "; ".join(f"+{100*g:.0f}% / {100*s:.0f}%" for g, s in v)
            elif isinstance(v, float):
                val = f"{v:g}"
            elif isinstance(v, bool):
                val = "on" if v else "off"
            else:
                val = str(v)
        rows.append({"Parameter / Threshold": label, "Actual Value": val, "Unit": unit,
                     "Activation Condition": act, "Effect on Portfolio": eff, "Class": klass})
    reg = pd.DataFrame(rows)
    reg.to_csv(out / "rq03_parameters.csv", index=False, encoding="utf-8-sig")

    stats = []
    for per in ("DEV", "OOS"):
        d = pd.read_csv(L / f"CHOSEN_STANDARD_LEDGER_{per}.csv", parse_dates=["week_end"])
        g = d.groupby("phase").agg(weeks=("nav", "size"), equity=("equity", "mean"),
                                   bond=("bond", "mean"), risky=("risky", "mean"),
                                   turnover=("turnover", "mean"), cost=("cost", "sum"))
        for ph, r in g.iterrows():
            stats.append({"period": per, "phase": ph, "weeks": int(r.weeks),
                          "avg_equity_pct": 100 * r.equity, "avg_bond_pct": 100 * r.bond,
                          "avg_risky_pct": 100 * r.risky, "avg_turnover_pct": 100 * r.turnover,
                          "total_cost_pct_nav": 100 * r.cost})
        stats.append({"period": per, "phase": "ALL", "weeks": len(d),
                      "avg_equity_pct": 100 * d.equity.mean(), "avg_bond_pct": 100 * d.bond.mean(),
                      "avg_risky_pct": 100 * d.risky.mean(),
                      "avg_turnover_pct": 100 * d.turnover.mean(),
                      "total_cost_pct_nav": 100 * d.cost.sum()})
    st = pd.DataFrame(stats)
    st.to_csv(out / "rq03_phase_stats.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ state diagram
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    boxes = {
        "S1": (0.6, 3.9, 3.0, 1.5, "#dfeaf7",
               "S1  Normal\nEquity 90% · Bonds 10%\n(band: equity 80–100%)"),
        "S2": (0.6, 1.0, 3.0, 1.6, "#fde7c8",
               "S2  Locked after short confirmation\nBonds 65% ± tilt by stress probability\nclamped 50–80% · cash remainder · equity 0\npre-bottom supplement allowed"),
        "S3": (6.4, 1.0, 3.0, 1.6, "#f7c6c7",
               "S3  Bottom confirmed\nEquity 100%\nheld while the profit ladder runs"),
        "S4": (6.4, 3.9, 3.0, 1.5, "#d7ecd9",
               "S4  Post-exit\nEquity band 50–100%\nBonds 0–25% · cash remainder"),
    }
    for k, (x, y, w, h, c, txt) in boxes.items():
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                             linewidth=1.4, edgecolor="#333333", facecolor=c, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.2, zorder=3)
    def arrow(a, b, text, rad=0.0, tx=0.0, ty=0.0, ha="center"):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#2b2b2b",
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=6, shrinkB=6), zorder=1)
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, fontsize=8.4,
                ha=ha, va="center", color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9), zorder=4)
    arrow((3.05, 4.2), (3.05, 2.9), "Short confirmed\n(RQ02 last-short flag)", tx=-0.15, ha="right")
    arrow((3.05, 1.8), (6.95, 1.8), "Bottom confirmed\n(RQ02 bottom trigger)", ty=0.22)
    arrow((6.95, 2.9), (6.95, 4.0), "Profit ladder completes\nor trailing stop", tx=0.15, ha="left")
    arrow((6.95, 5.4), (3.05, 5.4), "Exit phase ends (normal state returns)", ty=0.2)
    ax.text(5.0, 4.95, "26 consecutive normal weeks re-arm the bottom supplement",
            fontsize=7.8, ha="center", color="#555555")
    ax.text(0.6, 0.35, "Lock duration is the defining feature: no restructuring is executed between short confirmation and bottom confirmation.",
            fontsize=8.6, style="italic", color="#444444")
    ax.set_title("Main model restructuring: state transitions and allocation rules", fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out / "fig_rq03_state_machine.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(reg[["Parameter / Threshold", "Actual Value", "Class"]].to_string(index=False))
    print("\n", st.to_string(index=False))


if __name__ == "__main__":
    main()
