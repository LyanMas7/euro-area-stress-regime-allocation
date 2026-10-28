"""Number formatting and text utilities for the academic standard."""

from __future__ import annotations

import textwrap

import numpy as np

PROOF_BANNER = (
    "STRUCTURAL LAYOUT PROOF \u2014 PLACEHOLDER DATA \u2014 NOT EMPIRICAL RESULTS"
)
PROOF_NOTE = (
    "Layout proof only: all numeric tokens on this page are placeholders; no statistic is "
    "empirical or citable."
)
STAR_NOTE = (
    "Significance stars are displayed for layout verification only; no significance test "
    "was evaluated for this proof."
)
BOLD_NOTE = (
    "Bold cells are displayed for layout verification only; no registered comparison rule "
    "was evaluated for this proof."
)


def fmt_num(value, decimals: int = 3, thousands: bool = False, na: str = "NA") -> str:
    if value is None:
        return na
    if isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return na
    if isinstance(value, (int, np.integer)):
        return f"{value:,}" if thousands else str(value)
    text = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"
    if text.startswith("-0.") and float(text) == 0:
        text = text[1:]
    return text


def fmt_p(value) -> str:
    if value is None:
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def stars(p, symbols=None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    symbols = symbols or {"0.10": "*", "0.05": "**", "0.01": "***"}
    if p < 0.01:
        return symbols["0.01"]
    if p < 0.05:
        return symbols["0.05"]
    if p < 0.10:
        return symbols["0.10"]
    return ""


def fmt_est_se(estimate, se, decimals: int = 4, na: str = "NA") -> str:
    if estimate is None:
        return na
    est = fmt_num(estimate, decimals, na=na)
    if se is None:
        return est
    return f"{est} ({fmt_num(se, decimals, na=na)})"


def wrap(text: str, width: int = 104) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def escape_rl(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def signed(text: str) -> str:
    """Use a true minus sign for display while keeping ASCII in data exports."""
    return text.replace("-", "\u2212") if text.startswith("-") else text
