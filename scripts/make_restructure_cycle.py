"""Restructuring cycle: Normal -> Last Short -> Bottom -> Post-Bottom -> Normal.

Four clearly separated cards with the transition arrows in the gaps. RQ01
identifies the state, RQ02 supplies Pn/Ps from the H6 quarterly pre-onset model
and the 3-month exit model, and RQ03 supplies the risk cap, volatility and
z-score constraints.
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
        size: float = 7.0, fill: str | None = None) -> None:
    page.fig.add_artist(Rectangle((x / page.W, 1 - (y + h) / page.H), w / page.W, h / page.H,
                                  transform=page.fig.transFigure, facecolor=fill or "white",
                                  edgecolor=INK, linewidth=0.6, zorder=1))
    page.fig.text((x + w / 2) / page.W, 1 - (y + h / 2) / page.H, text, ha="center", va="center",
                  fontsize=size, fontfamily="Garamond", linespacing=1.25, zorder=2)


def seg(page: FigurePage, x1: float, y1: float, x2: float, y2: float, arrow_end: bool = True) -> None:
    page.fig.add_artist(FancyArrowPatch((x1 / page.W, 1 - y1 / page.H), (x2 / page.W, 1 - y2 / page.H),
                                        transform=page.fig.transFigure, arrowstyle="-|>" if arrow_end else "-",
                                        mutation_scale=9, linewidth=0.8, color=INK))


def main() -> None:
    page = FigurePage(number=None, top_in=0.80,
                      title="Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom",
                      paragraph="RQ01 identifies the current state, RQ02 supplies the probabilities Pn and Ps from "
                                "the H6 quarterly pre-onset model and the 3-month exit model, and RQ03 applies the "
                                "risk cap, volatility and z-score constraints. The portfolio is locked between last "
                                "short and bottom confirmation.")

    W, GAP, Y, H = 1.48, 0.22, 3.80, 2.75
    x0, x1, x2, x3 = 0.95, 0.95 + W + GAP, 0.95 + 2 * (W + GAP), 0.95 + 3 * (W + GAP)
    ymid = Y + H / 2

    box(page, x0, Y, W, H,
        "NORMAL  (S1)\n\nRQ01: calm\nP_S = weekly hazard of\nthe H6 quarterly\npre-onset probability\n\nallocation\n80-100% equity\n0-20% bonds",
        size=6.9, fill="#DCE6F1")
    box(page, x1, Y, W, H,
        "LAST SHORT  (S2)\n\nfive-vote short\nsystem fires\nP_N = 1 - P_S\n\nlock: restructuring\nprohibited\n\nallocation\n50-80% bonds\nno equity",
        size=6.9, fill="#FCE4D6")
    box(page, x2, Y, W, H,
        "BOTTOM  (S3)\n\ngates\nP_N >= 0.995\nP_S <= 0.02\nz >= 2\ndrawdown <= 5%\nF1-F3: 2 of 3 agree\n\nallocation\n100% equity",
        size=6.9, fill="#F8CBAD")
    box(page, x3, Y, W, H,
        "POST-BOTTOM\n(S4)\n\nexit 3M\np >= 0.027174\nor ladder complete\n\nallocation\n50-100% equity\n0-25% bonds",
        size=6.9, fill="#E2EFDA")

    seg(page, x0 + W, ymid, x1, ymid)
    seg(page, x1 + W, ymid, x2, ymid)
    seg(page, x2 + W, ymid, x3, ymid)
    seg(page, x3 + W / 2, Y + H, x3 + W / 2, 7.00, arrow_end=False)
    seg(page, x3 + W / 2, 7.00, x0 + W / 2, 7.00, arrow_end=False)
    seg(page, x0 + W / 2, 7.00, x0 + W / 2, Y + H)
    page.fig.text(((x3 + W / 2) + (x0 + W / 2)) / 2 / page.W, 1 - 7.13 / page.H,
                  "26 consecutive normal weeks re-arm the cycle", ha="center", va="top",
                  fontsize=7.4, fontfamily="Garamond", style="italic")

    box(page, 0.84, 2.20, 1.70, 0.80, "RQ01 - STATE\npooled rule:\ncalm or stress", size=7.0)
    box(page, 3.99, 2.20, 2.20, 0.80, "RQ02 - Pn / Ps\npre-onset: H6 quarterly\nexit: 3M, p >= 0.027174", size=7.0)
    box(page, 6.04, 2.20, 1.50, 0.80, "RQ03 - RISK\nES cap 2%\nvolatility, z\n10 bp cost", size=7.0)
    seg(page, x0 + W / 2, 3.00, x0 + W / 2, Y)
    seg(page, x2 + W / 2, 3.00, x2 + W / 2, Y)
    seg(page, x3 + W / 2, 3.00, x3 + W / 2, Y)

    page.fig.text(0.5, 1 - 7.75 / page.H,
                  "weekly entry hazard 1-(1-p_H6)^(1/78)      weekly exit hazard 1-(1-p_3M)^(1/13)",
                  ha="center", va="top", fontsize=9.0, fontfamily="Garamond", fontweight="bold")
    finish(page, Path("results"), "fig03_rq03_restructuring_flow",
           "Notes: the lock forbids restructuring between last short and bottom confirmation; a favourable signal "
           "produces no trade while the lock is active. Bottom confirmation is a state change, not a return "
           "guarantee: the realised equity trough arrived 0 to 20 weeks after confirmation across the seven "
           "episodes. The complete threshold set is in Appendix C.3.",
           "Source: author's construction from the selected configuration and the executed ledger. "
           "Thresholds: Appendix C.3.", 8.55)


if __name__ == "__main__":
    main()
