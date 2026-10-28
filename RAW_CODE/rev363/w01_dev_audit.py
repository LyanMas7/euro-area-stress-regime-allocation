"""Executed W01 source/DEV audits under hard EA8-vs-EA7 and OOS gates.

Run:
 python -m rev363.w01_dev_audit --dataset-root <ASSET03_ROOT> --out <dir>

EA7 ONLY BY DEFAULT (BE, DE, ES, FI, FR, IT, NL): neither IE nor PT is read.  The former EA8 raw-descriptive path,
which reads IE, is OPT-IN (`--include-ie-descriptive-audit`), is written to its own `EA8_DESCRIPTIVE_ONLY_NOT_FOR_FIT`
sub-directory, and nothing from it can feed an EA7 result or a fitted model.  The EA7-vs-EA8 owner conflict stays
recorded as DATA_CONTRACT_CONFLICT(EA7_RESEARCH_SCOPE, EA8_OFFICIAL_OWNER).

This is NOT supervised estimation, cannot create M00-M11 probabilities or an
issued signal without real issuer PIT + resolving the binding EA8 owner scope.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rev363 import labels as LABELS
from rev363 import labels_next as LN
from rev363 import data as DATA
from rev363 import w01_native as NATIVE
from rev363 import w01_native as NATIVE_TABLES
from rev363.config import EA7
from rev363.w01_contract import diagnose_scope,enforce_execution,SCIENTIFIC_COUNTRIES


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def write_json(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,default=str)+'\n')


def dev_label_register(episodes,countries=EA7):
    """All contracted country-month targets of the requested scope (EA7: 1260); no OOS label read/evaluation."""
    countries=tuple(countries)
    eps=[e for e in episodes if e['country'] in countries and
         e['start'].strftime('%Y-%m') <= '2019-12']
    rows=[]
    for c in countries:
        for m in DATA.month_range('2005-01','2019-12'):
            g=LN.eligible_gold(eps,c,m)
            rows.append(dict(country=c,issue=m,target_date=m,gold_label=g,
                             gold_status=('ESRB_EX_POST_DEV' if g is not None else 'UNAVAILABLE'),
                             released_at=None,release_verified=False,
                             used_as_issue_feature=False))
    f=pd.DataFrame(rows)
    n=len(countries)*180
    assert len(f)==n and f[['country','issue']].drop_duplicates().shape[0]==n
    assert f.gold_label.isin(('N','D','G','BOTH')).all()
    return f


def native_fresh_joint_coverage(native_tables):
    results=[]
    for a,b in NATIVE.PAIRS10:
        A=native_tables[a][['country','issue','native_fresh','terminal_abs','native_observed_at']].rename(
            columns={'native_fresh':'fresh_a','terminal_abs':'term_a','native_observed_at':'stamp_a'})
        B=native_tables[b][['country','issue','native_fresh','terminal_abs','native_observed_at']].rename(
            columns={'native_fresh':'fresh_b','terminal_abs':'term_b','native_observed_at':'stamp_b'})
        f=A.merge(B,on=['country','issue'],validate='one_to_one')
        both=f.fresh_a & f.fresh_b & f.term_a.notna() & f.term_b.notna()
        lags=(pd.to_datetime(f.stamp_a)-pd.to_datetime(f.stamp_b)).abs().dt.days
        results.append(dict(pair=a+'|'+b,panel_rows=len(f),cofresh=int(both.sum()),
                            exact_native_lag0=int((both & lags.eq(0)).sum()),
                            native_lag_le_31=int((both & lags.le(31)).sum()),
                            selected_by_owner='NOT_VERIFIED',
                            both_gate_magnitudes='NOT_SELECTED',
                            issuer_release_status='NOT_VERIFIED'))
    return pd.DataFrame(results)


def execute(dataset_root,out,*,include_ie_descriptive_audit=False):
    """EA7 by default.  The EA8 (IE) raw-descriptive audit is OPT-IN, isolated in its own sub-directory and
    tagged not-for-fit: it cannot feed any EA7 result or fitted model."""
    enforce_execution(research_only=True)
    countries=SCIENTIFIC_COUNTRIES if include_ie_descriptive_audit else EA7
    tag='EA8' if include_ie_descriptive_audit else 'EA7'
    scope='EA8_RAW_DESCRIPTIVE_ONLY_NOT_FOR_FIT' if include_ie_descriptive_audit else 'EA7_RESEARCH_SCOPE'
    out=Path(out)
    if include_ie_descriptive_audit:
        out=out/'EA8_DESCRIPTIVE_ONLY_NOT_FOR_FIT'
    out.mkdir(parents=True,exist_ok=True)
    dataset_root=DATA.find_dataset_root(dataset_root)
    bench=DATA._exact(dataset_root,'ESRB_RQ01_CANONICAL_BENCHMARK.xlsx')
    episodes=LABELS.load_episodes(dataset_root)
    labs=dev_label_register(episodes,countries)
    labs.to_csv(out/f'RQ01_01_DEV_LABEL_LEDGER_{tag}.csv',index=False)
    classes=labs.groupby(['country','gold_label']).size().unstack(fill_value=0).reindex(columns=['N','D','G','BOTH'],fill_value=0).reset_index()
    classes['n']=classes[['N','D','G','BOTH']].sum(axis=1)
    classes.to_csv(out/f'RQ01_01_{tag}_COUNTRY_COVERAGE.csv',index=False)

    # Execute source-native research input audit ONLY over the unchanged DEV block, for the requested scope.
    raw=DATA.build_monthly_raw(dataset_root,row_start='2005-01',row_end='2019-12',
                               data_end='2019-12',countries=countries,
                               descriptive_ie_audit=include_ie_descriptive_audit)
    assert raw['month'].min()=='2005-01' and raw['month'].max()=='2019-12'
    assert raw['country'].nunique()==len(countries) and len(raw)==len(countries)*180
    if not include_ie_descriptive_audit:
        assert set(raw['country'])==set(EA7) and not ({'IE','PT'} & set(raw['country']))
    source_tables={}
    source_rows=[]
    unit={'CISS':'CISS INDEX LEVEL','CLIFS':'CLIFS INDEX LEVEL','DFC':'SIGNED STANDARDISED COMPOSITE',
          'GFC':'NEGATIVE PRIOR Z OF PRICE','VIX':'INDEX POINTS'}
    frequency={'CISS':'WEEKLY','CLIFS':'MONTHLY','DFC':'QUARTERLY','GFC':'MONTHLY','VIX':'DAILY'}
    for s in NATIVE.SOURCES5:
        t=NATIVE.native_motion_table(raw,s)
        source_tables[s]=t
        t.to_csv(out/('RQ01_02_NATIVE_'+s+'.csv'),index=False)
        source_rows.append(dict(source=s,native_frequency=frequency[s],unit=unit[s],
                                country_scope=scope,rows=len(t),
                                native_new=int(t.native_fresh.sum()),
                                terminal_defined=int(t.terminal_abs.notna().sum()),
                                native_up=int((t.movement=='UP').sum()),
                                native_down=int((t.movement=='DOWN').sum()),
                                native_sideway=int((t.movement=='SIDEWAY').sum()),
                                unavailable_or_carried=int((t.movement=='UNAVAILABLE').sum()),
                                issuer_vintage_proved=False,model_fit_eligible=False))
    sources=pd.DataFrame(source_rows)
    sources.to_csv(out/'RQ01_02_NATIVE_SOURCE_COUNTS.csv',index=False)
    joint=native_fresh_joint_coverage(source_tables)
    joint.to_csv(out/'RQ01_03_10_PAIR_NATIVE_ALIGNMENTS.csv',index=False)

    # Executed six-pool tables, no gold or fitted thresholds (GLOBAL NO-U): the 2^3 BINARY truth table over
    # eligible votes, and the separate partial-input determination table (NE = component not evaluated;
    # an undetermined output is DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS, a pre-vote status).
    pool_rows=NATIVE_TABLES.binary_pool_truth_table()
    pd.DataFrame(pool_rows).to_csv(out/'RQ01_04_SIX_POOL_BINARY_TRUTH_TABLE.csv',index=False)
    pd.DataFrame(NATIVE_TABLES.pool_determination_table()).to_csv(
        out/'RQ01_04_SIX_POOL_DETERMINATION_TABLE.csv',index=False)

    # Required gates are exact; avoid claiming 5th model, fake OOS metrics, or fake approvals.
    ticket_states={
      'RQ01-01':('PARTIAL',f'ESRB DEV chronology verified for {tag}; official released_at and owner issuance unverified'),
      'RQ01-02':('PARTIAL',f'{tag} native source clocks/moves executed; issuer vintages and owner theta missing'),
      'RQ01-03':('PARTIAL','Ten native cofresh alignment tables executed; true Gate1/Gate2 theta and selected pairs unverified'),
      'RQ01-04':('PARTIAL','All six Kleene pool operators truth-tested; fitted interactions and matched loss blocked'),
      'RQ01-05':('BLOCKED','DATA_CONTRACT_CONFLICT+issuer PIT; no certified Logistic/XGBoost fit'),
      'RQ01-06':('NOT_EVALUATED','Six-phase empirical transitions need eligible source thresholds and owner rules'),
      'RQ01-07':('NOT_EVALUATED','Phase-conditional 1-to-5 probabilities require valid issued phases'),
      'RQ01-08':('BLOCKED','Complete score threshold search requires eligible matched model/owner prediction'),
      'RQ01-09':('NOT_DEFINED','Official F1-F5 definitions not source-authenticated'),
      'RQ01-10':('BLOCKED','Cannot issue actual p4 without eligible fit and verified issuer vintage'),
      'RQ01-11':('BLOCKED','Rev55/Rev64 authentic issued comparator inputs not verified (EA8 owner scope)'),
      'RQ01-12':('BLOCKED_BY_CONTRACT','OOS sealed and already historically exposed; supervised expanding prohibited')
    }
    report_rows=[dict(ticket=k,status=v[0],first_invalid_dependency=v[1],
                      run_id='W01_SOURCE_DEV_20261023_01',official_owner_gate='NOT_CERTIFIED')
                 for k,v in ticket_states.items()]
    pd.DataFrame(report_rows).to_csv(out/'RQ01_12_TICKET_GATES.csv',index=False)
    counts={k:int(v) for k,v in labs.gold_label.value_counts().to_dict().items()}
    science=diagnose_scope()
    summary={
      'run_id':'W01_SOURCE_DEV_20261023_01',
      'dataset_root_local_only':str(dataset_root),'benchmark_sha256':sha(bench),
      'research_step':'RAW_TO_LABEL_AND_NATIVE_EVIDENCE_ONLY',
      'scope_authority':'EA8_OFFICIAL_OWNER_CONFLICT_RECORDED' if not include_ie_descriptive_audit else 'EA8_DESCRIPTIVE_AUDIT_ONLY',
      'scope_used':scope,'countries':list(countries),'ie_included':bool(include_ie_descriptive_audit),
      'usable_for_ea7_claims':not include_ie_descriptive_audit,
      'can_feed_any_fit':False,
      'ie_pt_in_any_fit':False,
      'contract_conflict':'DATA_CONTRACT_CONFLICT(EA7_RESEARCH_SCOPE, EA8_OFFICIAL_OWNER)',
      'science_vs_dataset_scope':science.__dict__,
      'dev_target_country_months':len(labs),
      'dev_label_counts':counts,
      'native_source_panel_rows':len(raw),
      'native_sources':source_rows,'pair_counts':joint[['pair','cofresh','exact_native_lag0']].to_dict('records'),
      'n_pool_truth_rows':len(pool_rows),
      'issuer_pit':'UNVERIFIED','authentic_owner_A_B_C':'UNVERIFIED',
      'oos_rows_used_in_estimation_or_scoring':False,
      'oos_source_files_contain_future_rows':'YES; readers load files but only DEV rows are selected; not a physically air-gapped OOS',
      'oos_fitting':'BLOCKED_BY_CONTRACT','oos_scoring':'NOT_RUN',
      'supervised_dev_fit':'NOT_RUN_DATA_CONTRACT_CONFLICT',
      'hypothesis':{
        'H0':'Out-of-coverage labels do not leak into scored confusion because status masks are enforced.',
        'H1':'Out-of-coverage surrogate N can contaminate all-row classification unless masked.',
        'falsifier':'Append or replace 308 NOT_SCORABLE rows: matched scored confusion must stay bit-identical.',
        'experiment_kind':'UNIT_ADVERSARIAL_MUTATION_NOT_OOS_EXECUTION',
        'status':'TESTED_BY_UNIT_TEST_NO_LIVE_OOS_TAPE'},
      'threshold_contract':'NATIVE_UNITS_REGISTERED_ONLY; L/U/STEP/DECISION_MASK_UNEVALUATED',
      'four_estimator_model_comparison':'NOT_EVALUATED',
      'issued_signal_interface':'SCHEMA_ONLY_NOT_ISSUED',
      'model_promotion':'BLOCKED_NO_OWNER_APPROVAL',
      'ticket_gates':report_rows
    }
    write_json(out/'W01_DEV_EXECUTION_EVIDENCE.json',summary)
    # Empty schema is intentional. Never mint signals without a fitted, issued model.
    pd.DataFrame(columns=['country','issue_at','target_at','estimator','model','update_policy',
                          'p_N','p_D','p_G','p_BOTH','Stress_N','phase','source_hash',
                          'model_hash','threshold_hash','issuer','available_at','provenance_status']).to_csv(
                          out/'RQ01_ISSUED_P4_SCHEMA_NOT_ISSUED.csv',index=False)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--include-ie-descriptive-audit',action='store_true',
                   help='opt-in EA8 raw-descriptive audit that reads IE; isolated, never feeds a fit or an EA7 result')
    a=p.parse_args()
    s=execute(a.dataset_root,a.out,include_ie_descriptive_audit=a.include_ie_descriptive_audit)
    print(json.dumps({'run_id':s['run_id'],'scope':s['scope_used'],'DEV_country_months':s['dev_target_country_months'],
                      'classes':s['dev_label_counts'],'raw_native_rows':s['native_source_panel_rows'],
                      'OOS':'SEALED_NOT_RUN','model':'BLOCKED_DATA_CONTRACT_CONFLICT'},indent=2))

if __name__=='__main__':main()
