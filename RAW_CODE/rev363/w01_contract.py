"""W01 RQ01 contract and owner gates. No unilateral user-authority amendment.

Hard scientific scope is EA8; locked dataset's fitted scope is EA7.
The mismatch must be resolved before a *certified* fitted research run.

SCIENTIFIC_COUNTRIES (the EA8 owner contract, incl. IE) is a RECORD used only to diagnose the conflict
DATA_CONTRACT_CONFLICT(EA7_RESEARCH_SCOPE, EA8_OFFICIAL_OWNER).  Nothing that fits, scores or reports an EA7
result may iterate it: use `assert_ea7_only` on the country tuple of every active computation (IE and PT are refused).
"""
from __future__ import annotations

from dataclasses import dataclass

SCIENTIFIC_COUNTRIES = ("BE", "DE", "ES", "FI", "FR", "IE", "IT", "NL")
DATASET_FIT_COUNTRIES = ("BE", "DE", "ES", "FI", "FR", "IT", "NL")
FORBIDDEN_ACTIVE_COUNTRIES = ("IE", "PT")
DEV_FIRST, DEV_LAST = "2005-01", "2019-12"
OOS_FIRST, OOS_LAST = "2020-01", "2024-12"
# Explicit user authorization recorded in AI_HANDOFF DEC-015 (2026-10-25).
OOS_AUTHORIZATION_TOKEN = "USER_AUTHORIZED_OOS_20261025_DEC015"


class ContractBlockedError(RuntimeError):
    """Raised rather than allowing an unauthorized research run."""


@dataclass(frozen=True)
class ContractDiagnosis:
    status: str
    scientific_scope: tuple[str, ...]
    dataset_scope: tuple[str, ...]
    missing_in_data_scope: tuple[str, ...]
    requirement: str


def diagnose_scope(*, scientific=SCIENTIFIC_COUNTRIES, dataset=DATASET_FIT_COUNTRIES):
    missing = tuple(sorted(set(scientific) - set(dataset)))
    extra = tuple(sorted(set(dataset) - set(scientific)))
    return ContractDiagnosis(
        status="DATA_CONTRACT_CONFLICT" if missing or extra else "PASS",
        scientific_scope=tuple(scientific), dataset_scope=tuple(dataset),
        missing_in_data_scope=missing,
        requirement="Explicit authorized amendment or corrected EA8 eligible feature/owner provenance; no EA7-as-EA8 promotion."
    )


def assert_ea7_only(countries) -> tuple:
    """Scope guard for every ACTIVE (fit / score / report) computation: exactly BE,DE,ES,FI,FR,IT,NL."""
    got = tuple(countries)
    if (set(got) & set(FORBIDDEN_ACTIVE_COUNTRIES) or len(set(got)) != len(got)
            or set(got) != set(DATASET_FIT_COUNTRIES)):
        raise ContractBlockedError("EA7_ONLY_SCOPE_VIOLATION: " + ",".join(got))
    return got


def enforce_execution(*, oos_requested=False, supervised_refit=False,
                      amendment_reference=None, research_only=False,
                      oos_authorization=None):
    """Never treat an audit report or user goal as a formal contract amendment.

    OOS remains fail-closed unless the exact user-authorized token recorded in
    DEC-015 is supplied. Supervised expanding/refit remains blocked.
    """
    if supervised_refit:
        raise ContractBlockedError("BLOCKED_BY_CONTRACT: supervised OOS expanding/refit is prohibited")
    if oos_requested:
        if oos_authorization != OOS_AUTHORIZATION_TOKEN:
            raise ContractBlockedError("BLOCKED_BY_CONTRACT: independent OOS access not authorized")
        return "PASS_OOS_AUTHORIZED_USER_DEC015"
    if not research_only and diagnose_scope().status != "PASS":
        raise ContractBlockedError("DATA_CONTRACT_CONFLICT: EA8 authority vs EA7 fitted dataset")
    return "PASS_RESEARCH_ONLY" if research_only else "PASS"
