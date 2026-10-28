"""Restructuring decision flow: RQ01 state, RQ02 transitions, RQ03 constraints.

All formulas and thresholds on this canvas come from the executed overlay and the
selected configuration; the layers are connected with dependency arrows, not
economic-causality claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt2006.figure_canvas import FigurePage  # noqa: E402
from make_figures import finish  # noqa: E402

INK = "#000000"


def box(page: FigurePage, x: float, y: float, w: float, h: float, text: str,
        size: float = 7.8, fill: str | None = None, bold_head: bool = False) -> None:
    page.fig.add_artist(Rectangle((x / page.W, 1 - (y + h) / page.H), w / page.W, h / page.H,
                                  transform=page.fig.transFigure, facecolor=fill or "white",
                                  edgecolor=INK, linewidth=0.6, zorder=1))
    page.fig.text((x + w / 2) / page.W, 1 - (y + h / 2) / page.H, text, ha="center", va="center",
                  fontsize=size, fontfamily="Garamond", linespacing=1.32, zorder=2,
                  fontweight="bold" if bold_head else "normal")


def arrow(page: FigurePage, x1: float, y1: float, x2: float, y2: float) -> None:
    page.fig.add_artist(FancyArrowPatch((x1 / page.W, 1 - y1 / page.H), (x2 / page.W, 1 - y2 / page.H),
                                        transform=page.fig.transFigure, arrowstyle="-|>",
                                        mutation_scale=10, linewidth=0.9, color=INK))


def main() -> None:
    page = FigurePage(number=None, top_in=0.80,
                      title="Restructuring Decision Flow - How RQ01, RQ02 and RQ03 Control Allocation",
                      paragraph="Three verified layers feed one decision engine. The engine first reads the "
                                "financial state, then the transition and bottom information, then the risk and "
                                "weight constraints; only then does it choose to maintain, restructure under the "
                                "common rules, restructure under the bottom rules, or refuse to restructure at all.")

    box(page, 0.95, 2.30, 3.55, 1.62,
        "LAYER A - RQ01 FINANCIAL STATE\n"
        "P_S = stress probability;  P_N = 1 - P_S by construction\n"
        "normal state:  P_S = weekly entry hazard 1-(1-p12)^(1/52)\n"
        "stress state:  P_N = weekly exit hazard 1-(1-p_exit)^(1/13)\n"
        "state source: pooled rule (country threshold + two joint packets)",
        size=7.6, fill="#DCE6F1", bold_head=True)
    box(page, 0.95, 4.30, 3.55, 1.62,
        "LAYER B - RQ02 TRANSITION AND BOTTOM\n"
        "entry: 12-month model, applies outside phases\n"
        "last-short flag: five-vote short system, valid 18 months\n"
        "lock: short confirmed -> bottom not yet confirmed\n"
        "bottom gates: P_N >= 0.995, P_S <= 0.02, z >= 2, drawdown <= 5%\n"
        "bottom factors F1-F3: two of three must agree\n"
        "exit: p_exit >= 0.027174",
        size=7.6, fill="#FCE4D6", bold_head=True)
    box(page, 0.95, 6.30, 3.55, 1.62,
        "LAYER C - RQ03 RISK AND WEIGHTS\n"
        "weekly expected-shortfall cap 2% (bypassed only inside the bottom state)\n"
        "VaR/ES confidence set 95 / 97.5 / 99%\n"
        "state bands: S1 80-100% equity, S2 50-80% bonds, S4 50-100% equity\n"
        "profit ladder +8/+16/+25/+40%, 20% trims, 4-week cooldown\n"
        "weekly rebalancing, 10 bp one-way cost, long-only",
        size=7.6, fill="#E2EFDA", bold_head=True)

    box(page, 5.10, 3.55, 2.45, 2.30,
        "RESTRUCTURING\nDECISION ENGINE\n\n"
        "1 verify state\n2 check lock\n3 common vs bottom eligibility\n4 risk caps\n5 feasibility and clamps\n"
        "6 costs and turnover\n7 execute or hold",
        size=7.8, bold_head=True)
    arrow(page, 4.50, 3.10, 5.10, 4.20)
    arrow(page, 4.50, 5.10, 5.10, 4.60)
    arrow(page, 4.50, 7.10, 5.10, 5.20)

    actions = [("MAINTAIN\ncurrent allocation", 0.95, "#FFFFFF"),
               ("COMMON\nrestructuring", 2.85, "#DCE6F1"),
               ("BOTTOM-SPECIFIC\nrestructuring", 4.75, "#F8CBAD"),
               ("RESTRUCTURING\nPROHIBITED (lock)", 6.65, "#F4CCCC")]
    for text, x, fill in actions:
        box(page, x, 7.30, 1.80, 0.85, text, size=7.4, fill=fill)
    arrow(page, 6.32, 5.85, 6.32, 6.95)
    arrow(page, 6.32, 6.95, 1.85, 7.30)
    arrow(page, 6.32, 6.95, 3.75, 7.30)
    arrow(page, 6.32, 6.95, 5.65, 7.30)
    arrow(page, 6.32, 6.95, 7.55, 7.30)

    finish(page, Path("results"), "fig03_rq03_restructuring_flow",
           "Notes: P_N and P_S are complementary by construction in the executed overlay, not independent "
           "probabilities. A favorable signal does not produce a trade while the lock is active or when the risk "
           "caps make the target infeasible. Bottom confirmation does not guarantee a subsequent positive return: "
           "the realized equity trough arrived 0 to 20 weeks after confirmation across the seven episodes.",
           "Source: author's construction from the executed overlay and the selected configuration. "
           "Thresholds: Appendix C.3.", 8.75)


if __name__ == "__main__":
    main()
