"""Fail-closed RQ01 selected-versus-full-roster Joint feature ownership.

The existing fitted four-M model is a FULL_ROSTER_DEV_DIAGNOSTIC. This module
prevents reinterpreting all 10 exploratory pairs as approved selected pairs.
It deliberately does not mutate fitted coefficients or any source threshold.
"""
from __future__ import annotations
from hashlib import sha256
import json


def pair_consumer_gate(selected_pairs, consumed_pairs, *, mode='SELECTED_POLICY'):
    """Classify legitimate consumer mode; no H0/H1 metric and no relabeling.

    selected_pairs are N-clean Gate1/2 accepted pair identities, not merely
    PASS votes on a retrospectively observed stress month.
    """
    if mode not in ('SELECTED_POLICY','FULL_ROSTER_DIAGNOSTIC'):
        raise ValueError('UNKNOWN_JOINT_CONSUMER_MODE')
    if selected_pairs is None or consumed_pairs is None:
        # configuration BLOCKED (missing selected/consumed pair registry); never a vote
        return {'status':'BLOCKED','execution_status':'BLOCKED','reason':'SELECTED_AND_CONSUMED_LISTS_REQUIRED'}
    selected={tuple(sorted(x)) for x in selected_pairs}
    consumed={tuple(sorted(x)) for x in consumed_pairs}
    if any(len(k)!=2 or k[0]==k[1] or not all(isinstance(s,str) and s for s in k)
           for k in selected | consumed):
        raise ValueError('INVALID_PAIR_IDENTITY')
    if len(selected)!=len(selected_pairs) or len(consumed)!=len(consumed_pairs):
        raise ValueError('DUPLICATE_UNORDERED_PAIR')
    mismatch=sorted(consumed.symmetric_difference(selected))
    if mode=='SELECTED_POLICY':
        if mismatch:
            status='BLOCKED_SELECTED_CONSUMER_MISMATCH'
        else:
            status='PASS_STRUCTURE_ONLY_NOT_OWNER_CERTIFIED'
    else:
        status='RESEARCH_ONLY_FULL_ROSTER_NOT_SELECTED_POLICY'
    packet={'status':status,'mode':mode,'n_selected_pairs':len(selected),
            'n_consumed_pairs':len(consumed),'selected_ids':sorted(selected),
            'consumed_ids':sorted(consumed),'symmetric_difference':mismatch,
            'owner_selected_approved':False,'new_thresholds_used':False}
    packet['sha256']=sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest()
    return packet
