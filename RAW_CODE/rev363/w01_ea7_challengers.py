"""W01 REV366: standalone four-M RQ01 challenger mathematics and fit gates.

EA7 research only. Does not import historical Rev55/Rev64 outputs. A synthetic
unit-test fit is not evidence that the EA7 scientific scope/PIT are certified.
DEV is one indivisible block; no splitting, expanding, or OOS fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd

from rev363 import estimators as EST
from rev363 import metrics as MET
from rev363.w01_ea7_research import EA7, DEV_START, DEV_END

CLASSES = ('N','D','G','BOTH')
NAMES = ('M00','M01','M10','M11')


class UnverifiedChallengerInputs(RuntimeError):
    pass


def block_columns(*, base, individual, joint):
    """Matched factorial research design. Common BASE reused across all 4 M."""
    base=list(dict.fromkeys(base))
    ind=list(dict.fromkeys(individual))
    joi=list(dict.fromkeys(joint))
    if not base or not ind or not joi:
        raise ValueError('M00_M01_M10_M11_REQUIRES_REAL_FEATURE_BLOCKS')
    if (set(base)&set(ind)) or (set(base)&set(joi)) or (set(ind)&set(joi)):
        raise ValueError('OVERLAPPING_FACTORS_COMPROMISE_FOUR_M_ABLATION')
    return {'M00':base,'M01':base+ind,'M10':base+joi,'M11':base+ind+joi}


def _validate_population(frame, *, synthetic_only=False):
    req={'country','issue','gold_label'}
    if not req.issubset(frame):raise ValueError('MISSING_ISSUE_COUNTRY_GOLD_LABEL')
    if frame[['country','issue']].duplicated().any():raise ValueError('DUPLICATE_COUNTRY_ISSUE')
    if not frame.gold_label.isin(CLASSES).all():raise ValueError('UNAVAILABLE_GOLD_MUST_BE_MASKED_NOT_N')
    if not synthetic_only:
        if not frame.country.isin(EA7).all():raise ValueError('OUTSIDE_EA7_REQUESTED_RESEARCH')
        if not frame.issue.between(DEV_START,DEV_END).all():
            raise ValueError('OOS_OR_OUTSIDE_DEV_BLOCK_DENIED')
    if len(frame)==0: raise ValueError('EMPTY_ELIGIBLE_POPULATION')
    if set(frame.gold_label.unique()) != set(CLASSES):
        raise ValueError('ALL_FOUR_CLASSES_REQUIRED_FOR_MATCHED_FIT')


def _require_fit_contract(gates, *, synthetic_only):
    if synthetic_only:
        return 'TESTED_SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE'
    requirements=('scope_contract_amendment_id','native_release_ledger_sha256',
                  'label_eligibility_ledger_sha256','eligible_rows_sha256')
    if not isinstance(gates,dict) or any(not gates.get(k) for k in requirements):
        raise UnverifiedChallengerInputs('GATE_A_DORMANT: EA7 authorization, issuer PIT and verified label ledgers required')
    if not all(gates.get(k) is True for k in ('native_release_verified','label_join_verified',
                                             'full_dev_population_verified')):
        raise UnverifiedChallengerInputs('GATE_A_DORMANT: mandatory verification false')
    if gates.get('OOS_accessed_for_selection', True) is not False:
        raise UnverifiedChallengerInputs('OOS_SELECTION_NOT_ALLOWED')
    # Ledger hashes are required but are not cryptographic proof that the
    # issuer published on claimed dates; separate independent audit needed.
    return 'DEV_FIT_SUBJECT_TO_PROVENANCE_REVIEW'


def _binary_search(p4, y4, *, weight=1., fnr_cap=.10, fpr_cap=.20):
    """Full final decision: Stress from sum(p_D,p_G,p_BOTH) after every cut.

    Exhaust all observed continuous score decision breakpoints and extremes;
    penalty accounts for multiple candidate masks but does NOT de-bias DEV.
    """
    p4=np.asarray(p4,dtype=float);y4=np.asarray(y4,dtype=int)
    if p4.ndim!=2 or p4.shape!=(len(y4),4) or not np.isfinite(p4).all():
        raise ValueError('INVALID_P4_SHAPE_OR_NONFINITE')
    if np.any(p4<0) or np.any(p4>1) or not np.allclose(p4.sum(axis=1),1,atol=1e-6):
        raise ValueError('P4_NOT_ON_SIMPLEX')
    y=(y4!=0).astype(int)
    if not y.any() or not (y==0).any():raise ValueError('ONE_CLASS_NO_BINARY_FNR_FPR')
    stress=1.0-p4[:,0]
    unique=np.unique(stress)
    cuts=np.r_[np.inf,unique[::-1],-np.inf]
    masks=set();rows=[]
    for cut in cuts:
        pred=(stress>=cut).astype(np.int8)
        mhash=hashlib.sha256(pred.tobytes()).hexdigest()
        masks.add(mhash)
        cm=MET.confusion_binary(y,pred)
        # Registered scientific research penalty, not a fitted economic cost.
        rows.append({'threshold':float(cut),'decision_mask_sha256':mhash,
                     'FNR':cm['FNR'],'FPR':cm['FPR'],
                     'TP':cm['TP'],'FN':cm['FN'],'FP':cm['FP'],'TN':cm['TN']})
    multiplicity_penalty=weight*np.log(max(len(masks),1))/len(y)
    for x in rows:
        x['feasible']=bool(x['FNR']<fnr_cap and x['FPR']<fpr_cap)
        x['selection_penalty']=float(multiplicity_penalty)
        x['research_objective']=float(x['FNR']+x['FPR']+multiplicity_penalty)
    feasible=[r for r in rows if r['feasible']]
    selected=min(feasible,key=lambda r:(r['research_objective'],r['FP'],r['FN'])) if feasible else None
    return {'N_nominal':len(cuts),'N_executed':len(rows),
            'N_unique_complete_decision_masks':len(masks),
            'threshold_selected':None if selected is None else selected['threshold'],
            'selected_DEV_counts':selected, 'status':'FEASIBLE_DEV_RESEARCH' if feasible else 'NO_FEASIBLE_DEV_GATE',
            'selection_caveat':'APPARENT_DEV_NOT_INDEPENDENT_VALIDATION'}


def fit_dev_four_m(frame, blocks, *, gate_evidence=None, synthetic_only=False,
                   families=('LOGISTIC','XGBOOST')):
    """Common-key, common-block DEV two-family fit, closed by default.

    A missing Gate A forbids training on actual sources. synthetic_only is
    expressly for synthetic unit tests and never qualifies real W01 tickets.
    """
    _validate_population(frame,synthetic_only=synthetic_only)
    gate_status=_require_fit_contract(gate_evidence,synthetic_only=synthetic_only)
    if tuple(blocks)!=NAMES:raise ValueError('MUST_FIT_ALL_FOUR_M_ON_SAME_KEYS')
    y4=frame.gold_label.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(dtype=int)
    n=len(frame);results=[]
    for family in families:
        for name in NAMES:
            columns=blocks[name]
            if any(c not in frame.columns for c in columns):raise ValueError('MISSING_FEATURE_COLUMNS')
            x=frame[columns].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            if not np.isfinite(x).any(axis=0).all():
                raise ValueError('EMPTY_FEATURE_COLUMN_IN_SAME_KEY_POPULATION')
            mdl=EST.fit_two_family(x,columns,y4,family=family,name=name)
            p=mdl.p4(x)
            th=_binary_search(p,y4)
            results.append({'family':family,'model':name,'n_fit':n,'n_features':len(columns),
                            'model_digest':mdl.digest, 'source_gates':gate_status,
                            'logloss_DEV_fit':MET.multiclass_logloss(y4,p),
                            'brier_DEV_fit':MET.multiclass_brier(y4,p),
                            'threshold_full_policy':th,
                            'evidence_status':'TESTED_SYNTHETIC' if synthetic_only else 'EXECUTED_DEV_SUBJECT_TO_OWNER_REVIEW'})
    return {'status':gate_status, 'n':n,'models':results,
            'OOS_scored':False,'method':'NO_DEV_FOLDS_ONE_FULL_DEV_FIT'}
