"""Transfer-compatible function-first ConnectedExecutor, without hidden fitting.

This *minimal adapter* is intentionally NOT a clone of the archived complex
REV358 estimator/pool replayer. It provides a real deterministic connection of
caller-supplied RQ01 research stage functions, verified issue keys and complete
final mask deduplication. No historical labels, OOS refit, or owner changes.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import date
from hashlib import sha256
import inspect
import json

from .shared_threshold_contract import complete_decision_mask_sha, deduplicate_complete_timeline_decision_masks

from .vote_contract import NON_VOTE_STATUSES

STAGES = ('INDIVIDUAL','JOINT','POOL','PHASE','FORECAST')
DECISIONS = frozenset(('PASS','FAIL'))      # binary ModelVote; non-executed keys carry execution_status


def _checksum(obj):
    return sha256(json.dumps(obj, sort_keys=True, default=str, separators=(',',':')).encode()).hexdigest()


class ConnectedExecutor:
    """Link the five stages on identical DEV country/issue/horizon/target keys.

    Each stage has signature ``stage(row, context) -> dict`` and must preserve
    country, issue, horizon, target. FORECAST returns a binary final decision (PASS/FAIL) or a
    non-executed record (decision absent, execution_status NOT_EVALUATED/NOT_APPLICABLE/BLOCKED + reason).
    Source-readiness / native-period proofs are the responsibility of caller's
    stages and must be in their proof dictionaries; omission blocks execution.
    """
    def __init__(self, stage_functions, *, contract_id, code_owner_hash):
        if set(stage_functions) != set(STAGES) or not all(callable(stage_functions[s]) for s in STAGES):
            raise ValueError('EXACT_FIVE_REGISTERED_CALLABLE_STAGES_REQUIRED')
        if not contract_id or not code_owner_hash:
            raise ValueError('CONTRACT_AND_CODE_OWNER_HASH_REQUIRED')
        self._functions = dict(stage_functions)
        self.contract_id = str(contract_id)
        self.code_owner_hash = str(code_owner_hash)

    def execute(self, issued_inputs, *, context):
        """Execute same-key immutable row snapshots, with no fitting or tuning."""
        if not isinstance(context, dict) or not context.get('asof_policy_id'):
            raise ValueError('ASOF_POLICY_ID_REQUIRED')
        inputs = deepcopy(list(issued_inputs))
        original_hash = _checksum(inputs)
        seen, out, traces = set(), [], []
        for raw in inputs:
            key = tuple(str(raw[k]) for k in ('country','issue','horizon','target'))
            if key in seen:
                raise ValueError('DUPLICATE_ISSUE_TARGET')
            seen.add(key)
            issue = date.fromisoformat(key[1][:10] if len(key[1])>=10 else key[1]+'-01')
            if not date(2005,1,1) <= issue <= date(2019,12,31):
                raise ValueError('DEV_ONLY_OOS_SEALED')
            if not raw.get('source_readiness_proof') or not raw.get('source_vintage_sha'):
                raise ValueError('ASOF_NATIVE_READINESS_PROOF_REQUIRED')
            row = deepcopy(raw)
            stage_traces = []
            for stage in STAGES:
                prior_key = tuple(str(row[k]) for k in ('country','issue','horizon','target'))
                candidate = self._functions[stage](deepcopy(row), deepcopy(context))
                if not isinstance(candidate,dict) or not candidate.get('stage_proof_sha'):
                    raise ValueError('STAGE_PROOF_SHA_REQUIRED:'+stage)
                new_key = tuple(str(candidate[k]) for k in ('country','issue','horizon','target'))
                if prior_key != new_key:
                    raise ValueError('STAGE_CHANGED_ISSUED_TARGET:'+stage)
                if candidate.get('available_at') and str(candidate['available_at'])[:10] > issue.isoformat():
                    raise ValueError('FUTURE_STAGE_INFORMATION:'+stage)
                stage_traces.append({'stage':stage,'output_sha256':_checksum(candidate),
                                     'proof_sha':candidate['stage_proof_sha']})
                row = candidate
            base={k:v for k,v in zip(('country','issue','horizon','target'),key)}
            if row.get('decision') in DECISIONS:
                out.append(base | {'decision':row['decision'],'execution_status':'EXECUTED'})
            elif row.get('decision') is None and row.get('execution_status') in NON_VOTE_STATUSES \
                    and row.get('reason'):
                out.append(base | {'execution_status':row['execution_status'],'reason':row['reason']})
            else:
                raise ValueError('FINAL_BINARY_DECISION_OR_NON_VOTE_RECORD_REQUIRED:'+repr(row.get('decision')))
            traces.append({'key':key,'stages':stage_traces})
        if not out or original_hash != _checksum(inputs):
            raise ValueError('EMPTY_OR_MUTATED_ISSUED_INPUTS')
        return {'status':'RESEARCH_COMPONENT_ONLY_NOT_FITTED', 'contract_id':self.contract_id,
                'code_owner_hash':self.code_owner_hash, 'mask_sha256':complete_decision_mask_sha(out),
                'n_issued':len(out),'decision_rows':out,'stage_proof_traces':traces,
                'promoted':False,'OOS':'SEALED'}

    @staticmethod
    def deduplicate(candidates):
        return deduplicate_complete_timeline_decision_masks(candidates)
