"""W01 REV366: EA7 retrospective DEV source research, no legacy owner dependency.

This is an additive, explicitly non-certified research branch; it does not
modify the binding EA8 owner/production contract, original RAW ZIP or OOS data.
No Rev55/Rev64 code, prediction or provenance is required for these diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rev363 import data as DATA
from rev363 import labels as LABELS
from rev363 import labels_next as LABELS_NEXT
from rev363 import w01_native as NATIVE
from rev363 import w01_threshold_pool as THRESH
from rev363 import w01_intramonth as FULL_NATIVE
from rev363 import w01_native_joint_full as FULL_JOINT
from rev363.w01_contract import (DATASET_FIT_COUNTRIES, SCIENTIFIC_COUNTRIES, assert_ea7_only,
                                   diagnose_scope, enforce_execution)

EA7 = DATASET_FIT_COUNTRIES
DEV_START, DEV_END = '2005-01', '2019-12'
UNITS = {'CISS': 'CISS index', 'CLIFS': 'CLIFS index',
         'DFC': 'signed standardized composite', 'GFC': 'negative-prior-z price',
         'VIX': 'index points'}
FREQUENCIES = {'CISS': 'weekly', 'CLIFS': 'monthly', 'DFC': 'quarterly',
               'GFC': 'monthly', 'VIX': 'daily'}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def dev_labels(episodes):
    """Retrospective 7*180 target grid; NEVER interpreted as issued information."""
    rows = []
    for country in EA7:
        for m in DATA.month_range(DEV_START, DEV_END):
            label = LABELS_NEXT.eligible_gold(episodes, country, m)
            rows.append({'country': country, 'issue': m, 'target_date': m,
                         'gold_label': label,
                         'gold_status': 'ESRB_EX_POST_DEV' if label in ('N','D','G','BOTH') else 'UNAVAILABLE',
                         'label_released_at': None, 'label_release_verified': False,
                         'label_is_feature': False})
    frame = pd.DataFrame(rows)
    assert len(frame) == 7 * 180
    assert not frame[['country','issue']].duplicated().any()
    assert set(frame.country.unique()) == set(EA7)
    return frame


def _threshold_reference(native, source):
    """Observed, unlabeled scale ONLY: not a selected classifier threshold.

    Use native rows rather than carried monthly snapshots, so positive and
    negative movements have proper source frequencies.  Five grid points are
    descriptive bounds for magnitude; they are not a combined policy search.
    """
    new = native.loc[native.native_fresh & native.terminal_abs.notna(), 'terminal_abs'].to_numpy(float)
    magnitude = np.abs(new)
    positive = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    if len(positive) < 5:
        return {'source':source, 'unit':UNITS[source], 'n_new_moves':len(new),
                'L': None, 'U':None, 'STEP':None, 'nominal_candidates':0,
                'status':'INSUFFICIENT_NATIVE_POSITIVE_MAGNITUDE'}
    lo,hi = (float(x) for x in np.quantile(positive,[.25,.75]))
    if hi <= lo:
        axes=[lo]
        step=None
    else:
        step=(hi-lo)/4
        axes=[lo+step*i for i in range(5)]
    decisions = [((native['native_fresh'] & native['terminal_abs'].abs().ge(v))
                  .to_numpy(dtype=bool).tobytes()) for v in axes]
    unique_masks = len({hashlib.sha256(mask).hexdigest() for mask in decisions})
    return {'source':source,'unit':UNITS[source], 'n_new_moves':len(new),
            'positive_magnitude_support':len(positive),'L':lo,'U':hi,
            'STEP':step,'nominal_candidates':len(axes),
            'unique_native_component_masks':unique_masks,
            'status':'UNSUPERVISED_REFERENCE_ONLY_NOT_THRESHOLD_SELECTED',
            'selection_penalty':None, 'direction':'ABSOLUTE_MOVEMENT',
            'downstream_full_pipeline_masks':'NOT_EVALUATED'}


def execute(dataset_root, output_dir):
    """Execute an EA7-only, full-DEV, source/target audit, never OOS or fit."""
    enforce_execution(research_only=True, oos_requested=False, supervised_refit=False)
    root = DATA.find_dataset_root(dataset_root)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    episodes=LABELS.load_episodes(root)
    label_frame=dev_labels(episodes)
    label_path=output_dir/'RQ01_EA7_DEV_RETROSPECTIVE_GOLD.csv'
    label_frame.to_csv(label_path,index=False)
    counts=label_frame.groupby(['country','gold_label']).size().unstack(fill_value=0).reindex(
        index=list(EA7),columns=['N','D','G','BOTH'],fill_value=0)
    counts['n_total']=counts.sum(axis=1)
    counts.to_csv(output_dir/'RQ01_EA7_DEV_GOLD_COVERAGE.csv')
    raw=DATA.build_monthly_raw(root,row_start=DEV_START,row_end=DEV_END,
                               data_end=DEV_END,countries=assert_ea7_only(EA7))
    assert len(raw)==1260 and set(raw.country)==set(EA7)
    assert raw.month.min()==DEV_START and raw.month.max()==DEV_END
    sources={}
    nrows=[];register=[]
    for source in NATIVE.SOURCES5:
        t=NATIVE.native_motion_table(raw,source)
        assert len(t)==len(raw) and not t[['country','issue']].duplicated().any()
        t.to_csv(output_dir/f'RQ01_EA7_NATIVE_{source}.csv',index=False)
        sources[source]=t
        nrows.append({'source':source,'frequency':FREQUENCIES[source],
                      'unit':UNITS[source], 'rows':len(t),
                      'new_source_stamps_in_monthly_panel':int(t.native_fresh.sum()),
                      'all_native_periods_enumerated':bool(source in ('CLIFS','DFC','GFC')),
                      'monthly_snapshot_limitation':('DAILY_WEEKLY_INTRAMONTH_PATH_NOT_COUNTED'
                                                     if source in ('CISS','VIX') else 'NONE_DECLARED'),
                      'terminal_defined':int(t.terminal_abs.notna().sum()),
                      'up':int((t.movement=='UP').sum()),
                      'down':int((t.movement=='DOWN').sum()),
                      'sideway':int((t.movement=='SIDEWAY').sum()),
                      'carried_or_unavailable':int((t.movement=='UNAVAILABLE').sum()),
                      'PIT_status':'ISSUER_VINTAGES_NOT_VERIFIED'})
        register.append(_threshold_reference(t,source))
    pd.DataFrame(nrows).to_csv(output_dir/'RQ01_EA7_NATIVE_COUNTS.csv',index=False)
    pd.DataFrame(register).to_csv(output_dir/'RQ01_EA7_THRESHOLD_UNSUPERVISED_REGISTRY.csv',index=False)

    # REV367 closes the weekly CISS/daily VIX intramonth-count gap.
    # Global source tapes are NOT independent country-source observations.
    full_native=[];crosschecks=[];intramonth_axes=[];native_full_events={};native_votes=[]
    for source in FULL_NATIVE.SOURCES:
        native=FULL_NATIVE.intramonth_native_dev(root,source,dev_start=DEV_START,dev_end=DEV_END)
        native['monthly'].to_csv(output_dir/f'RQ01_EA7_FULL_NATIVE_{source}_MONTHLY.csv',index=False)
        native['events'].to_csv(output_dir/f'RQ01_EA7_FULL_NATIVE_{source}_EVENTS.csv',index=False)
        native_full_events[source]=native['events']
        check=FULL_NATIVE.verify_snapshots(native['monthly'],raw,source)
        if check['match']!='PASS':
            raise AssertionError(f'FULL_NATIVE_SNAPSHOT_DISAGREEMENT:{source}:{check}')
        crosschecks.append({'source':source,**check})
        full_native.append({'source':source,'native_frequency':FREQUENCIES[source],
            'global_unique_native_points':int(native['monthly'].native_points_new.sum()),
            'global_unique_native_moves_valid':int(native['monthly'].native_moves_valid.sum()),
            'global_up':int(native['monthly'].up_count.sum()),
            'global_down':int(native['monthly'].down_count.sum()),
            'global_sideway':int(native['monthly'].sideway_count.sum()),
            'global_months_with_points':int((native['monthly'].native_points_new>0).sum()),
            'countries_reusing_global_source':len(EA7),
            'source_sha256':native['source_sha256'],'assumed_release_lag_days':native['lag_days'],
            'PIT_status':'RESEARCH_PROXY_NOT_OFFICIAL_ISSUER_VERIFIED'})
        axis=FULL_NATIVE.intramonth_magnitude_reference(native['events'],source)
        intramonth_axes.append(axis)
        if axis['status']=='UNSUPERVISED_REFERENCE_ONLY':
            from rev363.w01_native import NativeThreshold
            rule=NativeThreshold(source,FREQUENCIES[source],UNITS[source],
                  strong_abs=float(axis['U']),weak_abs=float(axis['L']),min_count=2,
                  window=5 if source=='CISS' else 20,direction='POSITIVE')
            votes=FULL_NATIVE.full_native_individual_vote(native['monthly'],native['events'],
                  native['warmup_events'],rule)
            votes.to_csv(output_dir/f'RQ01_EA7_FULL_NATIVE_{source}_EXAMPLE_I_VOTES.csv',index=False)
            native_votes.append({'source':source,'window_native_moves':rule.window,
               'min_count':rule.min_count,'weak_abs':rule.weak_abs,
               'strong_abs':rule.strong_abs,'pass':int(votes.individual_vote.eq('PASS').sum()),
               'fail':int(votes.individual_vote.eq('FAIL').sum()),
               'not_evaluated':int(votes.individual_vote.isna().sum()),
               'vote_status':'UNSUPERVISED_EXAMPLE_ONLY_NOT_LABEL_FITTED'})
    pd.DataFrame(full_native).to_csv(output_dir/'RQ01_EA7_INTRAMONTH_NATIVE_SUMMARY.csv',index=False)
    pd.DataFrame(crosschecks).to_csv(output_dir/'RQ01_EA7_INTRAMONTH_SNAPSHOT_RECONCILIATION.csv',index=False)
    pd.DataFrame(intramonth_axes).to_csv(output_dir/'RQ01_EA7_INTRAMONTH_THRESHOLD_REFERENCE.csv',index=False)
    pd.DataFrame(native_votes).to_csv(output_dir/'RQ01_EA7_FULL_NATIVE_I_DEMONSTRATIONS.csv',index=False)

    # Alignment audit measures opportunities, NOT registered/selected pair votes.
    joined=[]
    for a,b in NATIVE.PAIRS10:
        x=sources[a][['country','issue','native_fresh','terminal_abs','native_observed_at']].rename(
            columns={'native_fresh':'fresh_a','terminal_abs':'delta_a','native_observed_at':'stamp_a'})
        y=sources[b][['country','issue','native_fresh','terminal_abs','native_observed_at']].rename(
            columns={'native_fresh':'fresh_b','terminal_abs':'delta_b','native_observed_at':'stamp_b'})
        d=x.merge(y,on=['country','issue'],validate='one_to_one')
        valid=d.fresh_a & d.fresh_b & d.delta_a.notna() & d.delta_b.notna()
        lag=(pd.to_datetime(d.stamp_a)-pd.to_datetime(d.stamp_b)).abs().dt.days
        joined.append({'pair':a+'|'+b,'EA7_dev_month_rows':len(d),
                       'both_native_fresh_and_terminal':int(valid.sum()),
                       'native_exact_day_zero':int((valid & lag.eq(0)).sum()),
                       'native_lag_upto_31d':int((valid & lag.le(31)).sum()),
                       'selected_pair':'NOT_EVALUATED',
                       'release_PIT':'NOT_VERIFIED',
                       'two_source_magnitude_gate':'NOT_EVALUATED'})
    pd.DataFrame(joined).to_csv(output_dir/'RQ01_EA7_JOINT10_NATIVE_OPPORTUNITIES.csv',index=False)
    # REV368: new true same-observation-day CISS/VIX paired native experiment.
    # Exactly one global pair tape: never count its 7 country copies as independent.
    pair_tape=FULL_JOINT.aligned_terminal_pairs(native_full_events['CISS'], native_full_events['VIX'])
    pair_months=FULL_JOINT.monthly_pair_coverage(pair_tape)
    ref={r['source']:r for r in intramonth_axes}
    pair_grid=FULL_JOINT.pair_magnitude_grid(pair_tape,ref['CISS'],ref['VIX'])
    pair_tape.to_csv(output_dir/'RQ01_EA7_FULL_NATIVE_CISS_VIX_PAIRED_EVENTS.csv',index=False)
    pair_months.to_csv(output_dir/'RQ01_EA7_FULL_NATIVE_CISS_VIX_MONTHLY.csv',index=False)
    pair_grid.to_csv(output_dir/'RQ01_EA7_FULL_NATIVE_CISS_VIX_TWO_MAG_GRID.csv',index=False)
    old_month_exact = next(r for r in joined if r['pair']=='CISS|VIX')['native_exact_day_zero']
    if len(pair_tape)<=old_month_exact:
        # Reject pre-registered H1 with evidence; the experiment remains valid.
        pair_hypothesis='H1_FALSIFIED_NO_ADDITIONAL_NATIVE_ALIGNMENTS'
    else:
        pair_hypothesis='H0_FALSIFIED_ADDITIONAL_NATIVE_ALIGNMENTS'
    pair_summary={'status':'EXECUTED_DEV_PROXY_RESEARCH_ONLY',
        'n_full_native_exact_day_pairs':len(pair_tape),
        'n_monthly_end_snapshot_exact_day_pairs':old_month_exact,
        'new_pairs_beyond_monthly_snapshot':len(pair_tape)-old_month_exact,
        'n_issue_months_with_pair':int(pair_months.has_new_native_pair.sum()),
        'n_months_without_pair':int((~pair_months.has_new_native_pair).sum()),
        'signed_mode_counts':{k:int(v) for k,v in pair_tape.signed_mode.value_counts().items()},
        'n_grid_executed':len(pair_grid),
        'n_grid_unique_complete_pair_masks':int(pair_grid.decision_mask_sha256.nunique()),
        'hypothesis_result':pair_hypothesis,
        'source_pair_global_not_country_multiplied':True,
        'PIT_status':'RESEARCH_PROXY_ISSUER_PIT_UNVERIFIED',
        'threshold_grid_status':'UNSUPERVISED_COMPONENT_NO_LABEL_NO_SELECTION',
        'M00_M01_M10_M11_losses':'NOT_EVALUATED',
        'OOS':'NOT_RUN'}

    # GLOBAL NO-U: binary truth table over eligible votes (2^3) + separate partial-input determination table
    truths=[{'individual':r['Individual'],'joint':r['Joint'],'complement':r['Complement'],
             **{k:v for k,v in r.items() if k not in ('Individual','Joint','Complement')}}
            for r in NATIVE.binary_pool_truth_table()]
    determination=NATIVE.pool_determination_table()
    pd.DataFrame(truths).to_csv(output_dir/'RQ01_EA7_POOL6_TRUTH_TABLE.csv',index=False)
    pd.DataFrame(determination).to_csv(output_dir/'RQ01_EA7_POOL6_DETERMINATION_TABLE.csv',index=False)

    report={'run_id':'W01_EA7_JOINT_CISS_VIX_NATIVE_REV368_01',
            'status':'EXECUTED_DEV_RAW_RESEARCH_ONLY',
            'research_scope':'EA7_USER_SELECTED_20261023',
            'canonical_scientific_contract_scope':'EA8_UNCHANGED_UNLESS_AMENDED',
            'scoping_conflict':diagnose_scope().__dict__,
            'incumbents_Rev55_Rev64_required_for_EA7_research':False,
            'incumbents_modified_or_deleted':False,
            'country_list':list(EA7), 'dev_calendar':'2005-01..2019-12_FULL_SINGLE_BLOCK',
            'label_counts':{k:int(v) for k,v in label_frame.gold_label.value_counts().items()},
            'gold_n':len(label_frame),'raw_n':len(raw),
            'native_counts':nrows,
            'intramonth_native_counts':full_native,
            'intramonth_snapshot_reconciliation':crosschecks,
            'rev368_full_native_joint_CISS_VIX':pair_summary,
            'intramonth_individual_demonstrations':native_votes,
            'threshold_axis_description':'UNSUPERVISED magnitude reference, not full downstream grid',
            'source_release_verified':False,'PIT_Gate_A':'DORMANT',
            'daily_weekly_native_path':'EXECUTED_WEEKLY_CISS_DAILY_VIX_NATIVE_SEQUENCE_DEV_RESEARCH_PROXY_PIT_ONLY',
            'eligible_logistic_xgboost_fit':'NOT_RUN_PIT_UNVERIFIED',
            'OOS_access':'NOT_RUN', 'OOS_refit':'BLOCKED_BY_CONTRACT',
            'owner_frozen_benchmark_comparison':'NOT_APPLICABLE_TO_STANDALONE_EA7_CHALLENGER_DIAGNOSTIC',
            'FNR_FPR':'NOT_EVALUATED_NO_VALID_ISSUED_PREDICTIONS',
            'six_pool_count':len(truths),'joint_pair_count':len(joined),
            'issued_signals':'NOT_ISSUED', 'model_promoted':False}
    report['evidence_files_sha256']={p.name:file_sha(p) for p in sorted(output_dir.glob('*.csv'))}
    (output_dir/'W01_EA7_RESEARCH_RUN.json').write_text(
         json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True,default=str)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description='W01 REV366 EA7 DEV research ONLY (no fitted models/OOS)')
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    report=execute(a.dataset_root,a.out)
    print(json.dumps({k:report[k] for k in ('run_id','status','gold_n','raw_n','PIT_Gate_A','OOS_access')},indent=2))


if __name__=='__main__':
    main()
