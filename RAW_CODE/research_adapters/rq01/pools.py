"""RQ01-15: six Individual/Joint pool operators and the nominal 72/108 reference registry.

  IND_ONLY = I;  JOINT_ONLY = J;  IND_AND_JOINT = I AND J;  IND_OR_JOINT = I OR J,
where I = OR_j I_j over sources and J = OR over the chosen pairs.  The two COMPLEMENT
operators are PAIR-INDEXED (spec IJP D2 / RQ01 PRE 3.3):
  COMPLEMENT_AND_JOINT = OR_pair [ J_pair AND OR_{h not in pair} I_h ]
  COMPLEMENT_OR_JOINT  = OR_pair [ J_pair OR  OR_{h not in pair} I_h ]
A source that belongs to ANOTHER chosen pair is still outside THIS pair and may support it; a
single global "sources outside every chosen pair" set is NOT the specified operator.
Nominal REV261 reference families: quantile q in {.25,.50,.75} x terminal confirm
{1,2} x triangle K {9,12,15} x 4 non-complement operators = 72, plus the 2 complement
operators = 108. These are REFERENCE policy families, never 'fitted' lattices.
GLOBAL NO-U: votes are binary PASS/FAIL.  A source/pair that is absent or non-evaluable is a NonVote
(external eligibility reason), never 0 or a third vote; a pool decision that cannot be proven from the
evaluable components is a NOT_EVALUATED record without any vote (work order section 2.4).  The binary
eligible-only kernel lives in research_adapters/ijp_kernel.py.
"""
from __future__ import annotations
from itertools import product

from ..vote_contract import NonVote, component, not_evaluated
from .common import SOURCES, decision_result, k_and, k_or, result

OPERATORS = ('IND_ONLY', 'JOINT_ONLY', 'IND_AND_JOINT', 'IND_OR_JOINT', 'COMPLEMENT_AND_JOINT', 'COMPLEMENT_OR_JOINT')
NOMINAL_REGISTRY = {'q': (.25, .50, .75), 'terminal_confirm': (1, 2), 'triangle_K': (9, 12, 15)}


def complement_for_pair(pair) -> tuple:
    """Sources outside ONE pair (the support set that may confirm that pair)."""
    pair = tuple(pair)
    if len(pair) != 2 or pair[0] == pair[1] or any(s not in SOURCES for s in pair):
        raise ValueError('INVALID_PAIR_IDENTITY')
    return tuple(s for s in SOURCES if s not in pair)


def complement_sources(chosen_pairs) -> tuple:
    """Sources that belong to NO chosen pair (descriptive only; not the pool's complement)."""
    used = {s for p in chosen_pairs for s in p}
    return tuple(s for s in SOURCES if s not in used)


def combine(operator: str, i_vote, j_vote, i_comp_vote):
    """Scalar form over binary votes (or NonVote components).  For the COMPLEMENT operators
    `i_comp_vote` must already be the support of the SAME pair that produced `j_vote`; use
    pool_decision for the pair-indexed operator."""
    if operator not in OPERATORS:
        raise ValueError('UNREGISTERED_POOL_OPERATOR')
    i_vote, j_vote, i_comp_vote = (component(x) for x in (i_vote, j_vote, i_comp_vote))
    return {'IND_ONLY': i_vote, 'JOINT_ONLY': j_vote, 'IND_AND_JOINT': k_and(i_vote, j_vote),
            'IND_OR_JOINT': k_or(i_vote, j_vote), 'COMPLEMENT_AND_JOINT': k_and(i_comp_vote, j_vote),
            'COMPLEMENT_OR_JOINT': k_or(i_comp_vote, j_vote)}[operator]


def _source_vote(individual_votes: dict, s: str):
    if s not in individual_votes:
        return not_evaluated('SOURCE_INDIVIDUAL_NOT_EVALUATED:' + s)
    return component(individual_votes[s], 'SOURCE_INDIVIDUAL_NOT_EVALUATED:' + s)


def pool_decision(operator: str, individual_votes: dict, joint_votes: dict, chosen_pairs) -> dict:
    """individual_votes: {source: PASS/FAIL | NonVote}; joint_votes: {pair: PASS/FAIL | NonVote}.

    An absent source is an eligibility gap (not negative); a known positive elsewhere can still
    determine OR positively.  An undetermined decision is a NOT_EVALUATED record with no vote."""
    if operator not in OPERATORS:
        raise ValueError('UNREGISTERED_POOL_OPERATOR')
    chosen = tuple(tuple(p) for p in chosen_pairs)
    if set(joint_votes) != set(chosen):
        raise ValueError('JOINT_VOTES_MUST_MATCH_CHOSEN_PAIRS')
    joints = {p: component(v, 'JOINT_PAIR_NOT_EVALUATED:' + '|'.join(p)) for p, v in joint_votes.items()}
    i_vote = k_or(*[_source_vote(individual_votes, s) for s in SOURCES])
    j_vote = k_or(*joints.values()) if joints else 'FAIL'
    comp_by_pair = {}
    for p in chosen:
        outside = complement_for_pair(p)
        comp_by_pair[p] = k_or(*[_source_vote(individual_votes, s) for s in outside]) if outside else 'FAIL'
    if operator in ('COMPLEMENT_AND_JOINT', 'COMPLEMENT_OR_JOINT'):
        join = k_and if operator == 'COMPLEMENT_AND_JOINT' else k_or
        vote = k_or(*[join(comp_by_pair[p], joints[p]) for p in chosen]) if chosen else 'FAIL'
    else:
        vote = combine(operator, i_vote, j_vote, 'FAIL')
    complement = k_or(*comp_by_pair.values()) if comp_by_pair else 'FAIL'
    summary = {k: v for k, v in (('individual', i_vote), ('joint', j_vote), ('complement', complement))
               if isinstance(v, str)}
    return decision_result(vote, source_ids=list(SOURCES),
                           components={'individual': i_vote, 'joint': j_vote, 'complement': complement},
                           operator=operator, **summary,
                           complement_by_pair={'|'.join(p): v for p, v in comp_by_pair.items()
                                               if isinstance(v, str)},
                           complement_not_evaluated_by_pair={'|'.join(p): v.reason for p, v in comp_by_pair.items()
                                                             if isinstance(v, NonVote)},
                           complement_sources_by_pair={'|'.join(p): list(complement_for_pair(p)) for p in chosen},
                           empty_complement_rule='FAIL_NO_SOURCE_OUTSIDE_THE_PAIR')


def truth_tables() -> dict:
    """Six operators over the 8 binary combinations of ELIGIBLE (I, J, I_complement) votes."""
    vals = ('PASS', 'FAIL')
    return {op: {(i, j, c): combine(op, i, j, c) for i, j, c in product(vals, vals, vals)} for op in OPERATORS}


def nominal_registry(include_complement: bool = True) -> dict:
    ops = OPERATORS if include_complement else OPERATORS[:4]
    rows = [{'q': q, 'terminal_confirm': b, 'triangle_K': k, 'operator': o}
            for q, b, k, o in product(*NOMINAL_REGISTRY.values(), ops)]
    return result('PASS', n_nominal=len(rows), rows=rows, role='REFERENCE_POLICY_FAMILY_NOT_FITTED',
                  composition='3 q x 2 confirm x 3 K x ' + str(len(ops)) + ' operators')
