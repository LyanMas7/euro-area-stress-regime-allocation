"""REV369: dataset-direct EA7 exploratory DEV research, never an issued forecast.

Uses ONLY supplied ASSET03 dataset; no internet/issuer-vintage verification
precondition for retrospective association analysis. The existing officially
scoped/gated fitting code is not replaced. Strictly 2005-2019, no OOS access,
no time folds or expanding/refit; all results are in-sample, development-biased.

The raw reader retains date-bounded/quarter-aware transformations: this is
necessary to avoid inadvertently treating later observations as earlier data.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rev363 import data as DATA, labels as LABELS, estimators as EST, metrics as MET
from rev363 import w01_native as NATIVE, w01_intramonth as INTRA
from rev363 import w01_native_joint_full as JOINT_FULL
from rev363 import w01_dfc_release_joint as DFC_JOIN
from rev363 import w01_vix_terminal_confirm as VIX_CONFIRM
from rev363.w01_ea7_research import dev_labels, EA7, DEV_START, DEV_END, file_sha
from rev363.w01_ea7_challengers import block_columns, _binary_search, NAMES, CLASSES
from rev363.w01_contract import assert_ea7_only, enforce_execution

FREQUENCY = {'CISS':'weekly','CLIFS':'monthly','DFC':'quarterly','GFC':'monthly','VIX':'daily'}
UNITS = {'CISS':'CISS index points','CLIFS':'CLIFS index points',
         'DFC':'standardized DFC composite units','GFC':'prior-z GFC units',
         'VIX':'VIX index points'}
# Direction is a research assumption, not a learned source classification.
SIGN = {'CISS':1,'CLIFS':1,'DFC':1,'GFC':-1,'VIX':1}
WINDOW = {'CISS':5,'CLIFS':3,'DFC':4,'GFC':3,'VIX':20}
LAG_DAYS = 31  # Monthly panel diagnostic except the CISS|VIX exact-day pair.
STATUS = 'EXECUTED_DEV_IN_SAMPLE_EXPLORATORY_NOT_ISSUED'


def _axis(df:pd.DataFrame, source:str, *, full=False):
    vals = pd.to_numeric(df['terminal_abs'],errors='coerce').abs().to_numpy(float)
    vals = vals[np.isfinite(vals)&(vals>0)]
    if len(vals)<5: raise ValueError('INSUFFICIENT_NATIVE_MAGNITUDES:'+source)
    lower,upper=np.quantile(vals,[.25,.75]);lower=float(lower);upper=float(upper)
    if not (0<lower<=upper):raise ValueError('INVALID_MAGNITUDE_BOUNDS:'+source)
    return {'source':source,'native_frequency':FREQUENCY[source],
            'polarity':SIGN[source], 'source_unit':UNITS[source],
            'L':lower,'U':upper,'STEP':(upper-lower)/4 if upper>lower else None,
            'K_count':2,'window_native_moves':WINDOW[source],
            'selection':'UNLABELED_DEV_QUARTILES_NOT_OPTIMIZED',
            'reference_n':len(vals), 'native_full':bool(full)}


def _encode(frame, prefix, votes):
    """Two-layer encoding: `_state` PASS/FAIL or empty, `_vote_status` EXECUTED/NOT_EVALUATED.

    GLOBAL NO-U: the REV373 `_unavailable` unknown-vote dummy regressor is retired; `_not_evaluated` is an
    eligibility indicator kept OUT of every model block.  A non-executed key is never read as FAIL."""
    t=votes[['country','issue','vote']].rename(columns={'vote':'state'}).copy()
    t['_present']=True
    if t[['country','issue']].duplicated().any():raise ValueError('DUPLICATE_VOTE:'+prefix)
    frame=frame.merge(t,on=['country','issue'],how='left',validate='one_to_one')
    if frame['_present'].isna().any():raise ValueError('MISSING_FEATURE_ROWS:'+prefix)
    frame=frame.drop(columns='_present')
    if (frame.state.eq('U')).any():raise ValueError('LEGACY_U_VOTE_REJECTED:'+prefix)
    if not (frame.state.isin(['PASS','FAIL'])|frame.state.isna()).all():raise ValueError('INVALID_BINARY_VOTE')
    frame[prefix+'_pass']=frame.state.eq('PASS').astype(int)
    frame[prefix+'_not_evaluated']=frame.state.isna().astype(int)
    frame[prefix+'_vote_status']=np.where(frame.state.isna(),'NOT_EVALUATED','EXECUTED')
    frame[prefix+'_state']=frame.pop('state')
    return frame


def _pair_monthly_vote(aa,bb,source_a,source_b,lo_a,lo_b):
    keys=['country','issue']
    a=aa[keys+['native_fresh','native_observed_at','terminal_abs']].rename(
        columns={'native_fresh':'fresh_a','native_observed_at':'stamp_a','terminal_abs':'delta_a'})
    b=bb[keys+['native_fresh','native_observed_at','terminal_abs']].rename(
        columns={'native_fresh':'fresh_b','native_observed_at':'stamp_b','terminal_abs':'delta_b'})
    z=a.merge(b,on=keys,how='inner',validate='one_to_one')
    lag=(pd.to_datetime(z.stamp_a)-pd.to_datetime(z.stamp_b)).abs().dt.days
    valid=(z.fresh_a & z.fresh_b & z.delta_a.notna() & z.delta_b.notna() & lag.le(LAG_DAYS))
    passed=(valid & z.delta_a.mul(SIGN[source_a]).ge(lo_a) &
            z.delta_b.mul(SIGN[source_b]).ge(lo_b))
    state=np.where(~valid,None,np.where(passed,'PASS','FAIL'))       # no fresh matched pair: no vote
    return pd.DataFrame({'country':z.country,'issue':z.issue,'vote':state})


def _full_ciss_vix_votes(pairs, lo_a,lo_b):
    # Native dates align one-to-one; require ++ directions and both magnitudes.
    hits=(pairs.signed_mode.eq('++') & pairs.delta_CISS.ge(lo_a) & pairs.delta_VIX.ge(lo_b))
    t=pairs.assign(hit=hits).groupby('issue',as_index=False).agg(
        paired_native_count=('hit','size'),any_hit=('hit','any'))
    issues=DATA.month_range(DEV_START,DEV_END)
    t=pd.DataFrame({'issue':issues}).merge(t,on='issue',how='left',validate='one_to_one')
    flags=t.any_hit.map(lambda x: bool(x) if pd.notna(x) else False)
    states=np.where(t.paired_native_count.isna(),None,np.where(flags,'PASS','FAIL'))
    country_frame=pd.MultiIndex.from_product([EA7,issues],names=['country','issue']).to_frame(index=False)
    by_m=pd.DataFrame({'issue':t.issue,'vote':states})
    return country_frame.merge(by_m,on='issue',validate='many_to_one')


def build_design(root, *, dfc_pair_mode="RELEASE_COINCIDENCE", vix_individual_mode="LEGACY_ANY20"):
    """Use only full DEV labels and source rows, NEVER future targets as features."""
    if dfc_pair_mode not in ("RELEASE_COINCIDENCE", "LEGACY_OBSERVATION_LAG"):
        raise ValueError("INVALID_DFC_PAIR_MODE")
    if vix_individual_mode not in ("LEGACY_ANY20", "TERMINAL_CONFIRM_5"):
        raise ValueError("INVALID_VIX_INDIVIDUAL_MODE")
    enforce_execution(research_only=True,oos_requested=False,supervised_refit=False)
    root=DATA.find_dataset_root(root)
    raw=DATA.build_monthly_raw(root,row_start=DEV_START,row_end=DEV_END,
                               data_end=DEV_END,countries=assert_ea7_only(EA7))
    labels=dev_labels(LABELS.load_episodes(root))
    if len(raw)!=1260 or len(labels)!=1260 or raw.month.max()!=DEV_END:
        raise AssertionError('EA7_DEV_1260_CONTRACT')
    df=labels[['country','issue','gold_label']].merge(
        raw,left_on=['country','issue'],right_on=['country','month'],
        how='inner',validate='one_to_one')
    if len(df)!=1260 or not df.gold_label.isin(CLASSES).all():
        raise ValueError('INVALID_FULL_DEV_LABEL_COVERAGE')
    # Explicitly no gold_label is used below as a feature or threshold input.
    native={s:NATIVE.native_motion_table(raw,s) for s in NATIVE.SOURCES5}
    full={s:INTRA.intramonth_native_dev(root,s,dev_start=DEV_START,dev_end=DEV_END)
          for s in INTRA.SOURCES}
    for s in INTRA.SOURCES:
        result=INTRA.verify_snapshots(full[s]['monthly'],raw,s)
        if result['match']!='PASS':raise ValueError('DATASET_INTERNAL_SNAPSHOT_MISMATCH:'+s)
    axes={s:_axis(full[s]['events'] if s in full else native[s].loc[native[s].native_fresh],
                  s,full=s in full) for s in NATIVE.SOURCES5}
    individual=[];joint=[];stats=[]
    for s in NATIVE.SOURCES5:
        from rev363.w01_native import NativeThreshold
        ax=axes[s]
        rule=NativeThreshold(s,FREQUENCY[s],ax['source_unit'],strong_abs=ax['U'],
               weak_abs=ax['L'],min_count=2,window=WINDOW[s],
               direction='POSITIVE' if SIGN[s]>0 else 'NEGATIVE')
        if s == "VIX" and vix_individual_mode == "TERMINAL_CONFIRM_5":
            v=VIX_CONFIRM.terminal_confirm_vix(
                full[s]["monthly"],full[s]["events"],full[s]["warmup_events"],
                L=ax["L"], U=ax["U"], window=5, min_count=2)
            v=pd.MultiIndex.from_product([EA7,v.issue],names=["country","issue"]).to_frame(index=False).merge(
                v[["issue","vote"]],on="issue",validate="many_to_one")
        elif s in full:
            v=INTRA.full_native_individual_vote(full[s]['monthly'],full[s]['events'],
                            full[s]['warmup_events'],rule).rename(columns={'individual_vote':'vote'})
            v=pd.MultiIndex.from_product([EA7,v.issue],names=['country','issue']).to_frame(index=False).merge(
                v[['issue','vote']],on='issue',validate='many_to_one')
        else:
            v=NATIVE.individual_threshold_vote(native[s],rule)
        key='i_'+s.lower()
        df=_encode(df,key,v)
        individual.append(key+'_pass')
        stats.append({'factor':'INDIVIDUAL','source_or_pair':s,
            'pass':int(df[key+'_state'].eq('PASS').sum()),
            'fail':int(df[key+'_state'].eq('FAIL').sum()),
            'not_evaluated':int(df[key+'_state'].isna().sum())})
    pairs=JOINT_FULL.aligned_terminal_pairs(full['CISS']['events'],full['VIX']['events'])
    for a,b in NATIVE.PAIRS10:
        if (a,b)==('CISS','VIX'):
            v=_full_ciss_vix_votes(pairs,axes[a]['L'],axes[b]['L'])
        elif "DFC" in (a,b) and dfc_pair_mode == "RELEASE_COINCIDENCE":
            other = b if a == "DFC" else a
            matched = DFC_JOIN.dfc_pair_issue_vote(
                native["DFC"], native[other], partner_name=other,
                dfc_min=axes["DFC"]["L"], partner_min=axes[other]["L"],
                partner_sign=SIGN[other])
            v = matched[["country", "issue", "vote"]]
        else:
            v=_pair_monthly_vote(native[a],native[b],a,b,axes[a]['L'],axes[b]['L'])
        key='j_'+a.lower()+'_'+b.lower()
        df=_encode(df,key,v)
        # The pure component does not fire when one of its individual sources
        # already supports the same observation. Descriptive, not causal.
        pure=(df[key+'_state'].eq('PASS') &
              df['i_'+a.lower()+'_state'].ne('PASS') &
              df['i_'+b.lower()+'_state'].ne('PASS'))
        df[key+'_pure']=pure.astype(int)
        joint.extend((key+'_pass',key+'_pure'))
        stats.append({'factor':'JOINT','source_or_pair':a+'|'+b,
            'pass':int(df[key+'_state'].eq('PASS').sum()),
            'fail':int(df[key+'_state'].eq('FAIL').sum()),
            'not_evaluated':int(df[key+'_state'].isna().sum()),
            'pure_pass':int(pure.sum())})
    # Baseline uses calendar/country only, not ESRB labels and not look-ahead.
    month=df.issue.str[-2:].astype(int)
    df['base_month_sin']=np.sin(2*np.pi*month/12)
    df['base_month_cos']=np.cos(2*np.pi*month/12)
    base=['base_month_sin','base_month_cos']
    for c in EA7[1:]:
        col='base_country_'+c.lower()
        df[col]=df.country.eq(c).astype(int)
        base.append(col)
    blocks=block_columns(base=base,individual=individual,joint=joint)
    assert list(blocks)==list(NAMES)
    for name,cols in blocks.items():
        if any(k.endswith('_state') or 'gold' in k for k in cols):
            raise AssertionError('INVALID_LABEL_OR_STATE_FEATURE')
        if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
            raise ValueError('NONFINITE_DESIGN:'+name)
    return df,blocks,axes,pd.DataFrame(stats),pairs


def _full_threshold_sweep(p4,y4,family,model):
    """Enumerate all distinct binary final-policy masks on matched DEV keys.

    Return the complete grid and the feasible frontier diagnostics. This is
    re-evaluated after every model/feature change, NOT published thresholds.
    """
    stress=1.0-np.asarray(p4,float)[:,0]
    y=(np.asarray(y4,int)!=0).astype(int)
    n_p=int(y.sum());n_n=int((y==0).sum())
    if n_p==0 or n_n==0:raise ValueError('NEEDS_BOTH_BINARY_CLASSES')
    # Thresholding rule is >= cut; unique values give unique masks; +inf is
    # the always-N end. Tied score values are handled as one decision block.
    cutpoints=np.r_[np.inf,np.unique(stress)[::-1]]
    rows=[]
    for cut in cutpoints:
        pred=stress>=cut
        fn=int(((y==1)&~pred).sum());fp=int(((y==0)&pred).sum())
        tp=n_p-fn;tn=n_n-fp
        fnr=fn/n_p;fpr=fp/n_n
        rows.append({'family':family,'model':model,'threshold':cut,
            'TP':tp,'FN':fn,'FP':fp,'TN':tn,'n_stress':n_p,'n_normal':n_n,
            'FNR':fnr,'FPR':fpr,'objective_FNR_plus_FPR':fnr+fpr,
            'dev_gate':bool(fnr<.10 and fpr<.20),
            'status':'APPARENT_DEV_THRESHOLD_SWEEP_NOT_VALIDATED'})
    grid=pd.DataFrame(rows)
    feasible=grid.loc[grid.dev_gate]
    under_fpr=grid.loc[grid.FPR<=.20]
    under_fnr=grid.loc[grid.FNR<=.10]
    def best(tab,keys):
        if tab.empty:return None
        z=tab.sort_values(keys,kind='stable').iloc[0]
        return {'threshold':float(z.threshold),'FNR':float(z.FNR),
                'FPR':float(z.FPR),'TP':int(z.TP),'FN':int(z.FN),
                'FP':int(z.FP),'TN':int(z.TN)}
    summary={'family':family,'model':model,'N_executed':len(grid),
      'N_unique_complete_decision_masks':len(grid),
      'N_feasible_STRICT_DEV_gate':len(feasible),
      'balanced_min_FNR_plus_FPR':best(grid,['objective_FNR_plus_FPR','FNR','FPR']),
      'lowest_FNR_given_FPR_le_20pct':best(under_fpr,['FNR','FPR']),
      'lowest_FPR_given_FNR_le_10pct':best(under_fnr,['FPR','FNR']),
      'selection_note':'APPARENT_DEV_NOT_PERMITTED_FOR_FROZEN_OOS_SELECTION'}
    return grid,summary

def fit_descriptive_dev(df,blocks,out,*,families=('LOGISTIC','XGBOOST')):
    """Perform matched full-DEV apparent fit; bypass NO official model gate.

    This is separate from w01_ea7_challengers.fit_dev_four_m which retains
    its formal gate. No independent validation or published issue time is
    claimed. Do not call this function for OOS, production, or prediction.
    """
    if len(df)!=1260 or set(df.country)!=set(EA7) or not df.issue.between(DEV_START,DEV_END).all():
        raise ValueError('DESCRIPTIVE_DEV_EA7_FULL_BLOCK_ONLY')
    if df[['country','issue']].duplicated().any():raise ValueError('DUPLICATE_DEV_KEYS')
    if list(blocks)!=list(NAMES):raise ValueError('FOUR_M_REQUIRED')
    y=df.gold_label.map({s:i for i,s in enumerate(CLASSES)}).to_numpy(int)
    if set(y)!=set(range(4)):raise ValueError('FOUR_GOLD_CLASSES_REQUIRED')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scores=[];preds=[];hashes=[];threshold_grids=[];threshold_summaries=[];confusions=[];country_errors=[]
    for family in families:
        for name in NAMES:
            cols=blocks[name];X=df[cols].to_numpy(float)
            model=EST.fit_two_family(X,cols,y,family=family,name=name)
            p=model.p4(X)
            if p.shape!=(len(df),4) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1):
                raise AssertionError('MODEL_SIMPLEX_INVALID')
            search=_binary_search(p,y)
            complete_grid,sensitivity=_full_threshold_sweep(p,y,family,name)
            threshold_grids.append(complete_grid)
            threshold_summaries.append(sensitivity)
            th=search['threshold_selected']
            if th is None:
                # This threshold is a transparent diagnostic only, not tuned
                # to yield an apparently favorable gate when no gate is feasible.
                th=.5
            yh=(1-p[:,0]>=th).astype(int)
            cm=MET.confusion_binary((y!=0).astype(int),yh)
            prediction=pd.DataFrame({'country':df.country,'issue':df.issue,
                       'gold_label':df.gold_label,'p_N':p[:,0],'p_D':p[:,1],
                       'p_G':p[:,2],'p_BOTH':p[:,3],'stress_score':1-p[:,0],
                       'stress_pred_fit':yh,'regime_pred_fit':np.asarray(CLASSES,dtype=object)[p.argmax(axis=1)],
                       'model':name,'family':family})
            preds.append(prediction)
            ctab=pd.crosstab(prediction.gold_label,prediction.regime_pred_fit)
            for true_class in CLASSES:
                for predicted_class in CLASSES:
                    confusions.append({'family':family,'model':name,'true_class':true_class,
                        'predicted_class':predicted_class,
                        'n':int(ctab.loc[true_class,predicted_class])
                        if true_class in ctab.index and predicted_class in ctab.columns else 0})
            for country,group in prediction.groupby('country',sort=True):
                truth=(group.gold_label!='N').to_numpy(dtype=int)
                predicted=group.stress_pred_fit.to_numpy(dtype=int)
                count=MET.confusion_binary(truth,predicted)
                country_errors.append({'family':family,'model':name,'country':country,
                    'n':len(group),'n_stress':int(truth.sum()),'n_normal':int((truth==0).sum()),
                    'FNR_fit':count['FNR'],'FPR_fit':count['FPR'],
                    'TP':count['TP'],'TN':count['TN'],'FP':count['FP'],'FN':count['FN'],
                    'status':STATUS})
            signature=model.signature
            if family=='LOGISTIC':
                artifact={'signature':signature,'columns':cols,'median':model.median.tolist(),
                        'scale':model.scale.tolist(),'coef':model.coef.tolist(),
                        'intercept':model.intercept.tolist()}
                path=out/f'{family}_{name}_fitted_DEV.json'
                path.write_text(json.dumps(artifact,indent=2,sort_keys=True)+'\n')
            else:
                path=out/f'{family}_{name}_fitted_DEV.json'
                path.write_bytes(base64.b64decode(model.booster_b64))
            hashes.append({'family':family,'model':name,'model_sha256':file_sha(path),
                           'model_filename':path.name})
            scores.append({'family':family,'model':name,'n_rows':len(y),
                 'country_count':len(EA7),'features':len(cols),
                 'p4_logloss_fit':MET.multiclass_logloss(y,p),
                 'p4_brier_fit':MET.multiclass_brier(y,p),
                 'FNR_fit':cm['FNR'],'FPR_fit':cm['FPR'],
                 'TP_fit':cm['TP'],'TN_fit':cm['TN'],
                 'FP_fit':cm['FP'],'FN_fit':cm['FN'],
                 'decision_threshold_in_sample':float(th),
                 'threshold_status':search['status'],
                 'n_nominal_thresholds':search['N_nominal'],
                 'n_unique_masks':search['N_unique_complete_decision_masks'],
                 'evidence_status':STATUS,
                 'matched_key_count':len(df),
                 'out_of_sample_evaluation':'NOT_RUN'})
    pd.DataFrame(scores).to_csv(out/'M00_M01_M10_M11_APPARENT_DEV_SCORES.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(out/'M00_M01_M10_M11_APPARENT_DEV_P4.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'MODEL_ARTIFACT_SHA256.csv',index=False)
    pd.DataFrame(confusions).to_csv(out/'M00_M01_M10_M11_4CLASS_CONFUSION.csv',index=False)
    pd.DataFrame(country_errors).to_csv(out/'M00_M01_M10_M11_COUNTRY_BINARY_ERRORS.csv',index=False)
    pd.concat(threshold_grids,ignore_index=True).to_csv(out/'APPARENT_DEV_COMPLETE_THRESHOLD_SWEEP.csv',index=False)
    (out/'APPARENT_DEV_THRESHOLD_DIAGNOSTICS.json').write_text(
        json.dumps(threshold_summaries,indent=2,sort_keys=True)+'\n')
    return scores


def run(dataset_root, out,*,families=('LOGISTIC','XGBOOST')):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df,blocks,axes,stats,pairs=build_design(dataset_root)
    # Tiny transfer table; full raw/source file stays in locked dataset.
    df[['country','issue','gold_label']+[s for s in df if s.endswith('_state')]].to_csv(
        out/'EA7_DEV_NATIVE_FACTOR_STATES.csv',index=False)
    stats.to_csv(out/'INDIVIDUAL_JOINT_SUPPORT.csv',index=False)
    (out/'THRESHOLDS_FROM_NATIVE_MOVEMENTS.json').write_text(json.dumps(
        axes,indent=2,sort_keys=True)+'\n')
    (out/'FEATURE_BLOCKS.json').write_text(json.dumps(blocks,indent=2)+'\n')
    scores=fit_descriptive_dev(df,blocks,out,families=families)
    result={'run_id':'W01_REV369_DATASET_DIRECT_DEV_01','status':STATUS,
            'data_mode':'DATASET_DIRECT_TRUSTED_AS_INPUT_NO_EXTERNAL_ISSUER_CHECK',
            'issue_rule':'DEV_MONTH_END_WITH_EXISTING_SOURCE_LAG_CONFIG',
            'retrospective_gold':'ESRB_LABEL_ONLY_NOT_MODEL_FEATURE',
            'full_dev':'2005-01..2019-12','ea7_countries':list(EA7),
            'n_matched':len(df),'class_counts':df.gold_label.value_counts().to_dict(),
            'joint_pair_count':len(NATIVE.PAIRS10),'native_ciss_vix_global_pair_count':len(pairs),
            'families':list(families),'model_count':len(scores),
            'model_comparisons':'APPARENT_IN_SAMPLE_ONLY_ADAPTIVELY_BIASED',
            'OOS':'NOT_USED_OR_SCORED_NO_SELECTION','issued_interface':'NOT_ISSUED',
            'certification':'NOT_CERTIFIED_NO_INDEPENDENT_OR_REAL_ISSUE_PROOF',
            'files_sha256':{p.name:file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'W01_REV369_RUN.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='W01 REV369 EA7 raw dataset direct DEV retrospective only')
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--family',choices=['LOGISTIC','XGBOOST','BOTH'],default='BOTH')
    a=p.parse_args(argv)
    fams=('LOGISTIC','XGBOOST') if a.family=='BOTH' else (a.family,)
    result=run(a.dataset_root,a.out,families=fams)
    print(json.dumps({k:result[k] for k in ('run_id','status','n_matched','model_count','OOS')},indent=2))

if __name__=='__main__':main()
