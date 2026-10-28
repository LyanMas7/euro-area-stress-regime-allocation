
### Appendix A - RQ01 Financial stress identification thresholds
Identification converts five native stress indicators into one binary country-month decision. The registered rule is a country threshold on a boosted-tree probability, one joint confirmation that can only add stress and one that can only remove it.

#### A.1 Individual signal thresholds

Confirmation geometry for the five native indicators. The magnitude cuts are percentiles of each country's own prior deltas, so they carry no common unit across indicators.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Observation window | 12 | months | every issue | One latest as-of native level per month; the vote is formed at the issue month |
| Native frequency - CISS / CLIFS / DFC / GFC price / VIX | W / M / Q / M / D | frequency | sampling | Weekly and daily series are sampled; quarterly series are de-duplicated so a quarter is never counted twice |
| Direction of movement | CISS +1, CLIFS +1, DFC -1, GFC price +1, VIX +1 | sign | every delta | D = sign x (level_end - level_base); DFC enters with a reversed sign, so a wider credit spread still means more stress |
| Prior-delta history | deltas before issue minus 11 months; at least 4 | months / count | every issue | Reference distribution for the magnitude cuts; below four prior deltas the source is NOT_EVALUATED |
| Strong magnitude cut (theta_hi) | 75th percentile of prior absolute adjacent deltas | native units | per country x source x issue | An endpoint difference above this cut counts as a strong confirmation |
| Weak magnitude cut (theta_lo) | 35th percentile of prior absolute adjacent deltas | native units | clamped to 0 <= lo <= hi | Lower cut that feeds the count leg of the vote |
| Minimum native levels | 2 | count | every issue | Fewer than two levels cannot produce a delta; the source is flagged N_LT_2_NATIVE_LEVELS |
| Minimum confirming bases | min(2, n-1) | count | strong and weak legs | A leg confirms only when at least this many base pairs pass the magnitude test |
| Confirmation count threshold | max(1, ceil(0.5 x M)), M = n(n-1)/2 | count | count leg | Half of the available base pairs must confirm |
| Magnitude predicate | D > 0 AND D >= cut | rule | each base pair | Equality confirms; a zero difference never confirms |
| Count predicate | signed_count >= count_k | rule | count leg | Counts signed confirming bases only |
| Individual vote | Strong OR (Weak AND Count) | rule | every issue | Single-issue decision with no persistence requirement |

#### A.2 Joint pair selection thresholds

Gates used to decide whether a pair may enter the rule. These are not the thresholds that activate the pair afterwards; activation is in A.3.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Pair roster and motif lag | 10 pairs, 9 ordered motifs, EXACT L = 0, delta = 0 | count / months | pair registry | Only the EXACT(0,0) lag is registered, so no pair may use a future publication |
| Selection gate G1 - matched native pairs | >= 2 | count | pair eligibility | Below two matched native pairs the pair is INELIGIBLE |
| Selection gate G3 - domain check | 1 <= min_bases <= n-1; 0 <= count_k <= M; lo <= hi | rule | pair eligibility | A pair whose count geometry is impossible is BLOCKED |
| Selection gate G4 - false-positive budget | new_fp <= round(0.15 x n_neg); source cap 2; <= 1024 subsets | count | pair selection | Budget that stops a pair from buying true positives with false alarms |
| Per-pair configuration choice | tp_final down, fp_final up, lost_tp up, new_fp up, motif up | lexicographic | one configuration per pair | Deterministic tie-break |
| Selected pairs in the registered rule | OR pkt_joint_CLIFS_VIX; NOT pkt_joint_DFC_GFC | packet flag | registered rule | Two pairs survive selection: one can only add stress, one can only remove it |

#### A.3 Joint activation thresholds

Conditions applied to a selected pair at each issue. Magnitudes are member-level percentiles; counts are base-pair counts.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Activation magnitude cuts | high = Q75, low = Q35 per member | native units | selected pair, every issue | The cuts used after selection are the member percentiles, not the selection gates |
| Gate 1 - separate-base magnitude test | PASS if a base has D_a >= gate1_a AND a base has D_b >= gate1_b | rule | member magnitudes | The two members may confirm on different base pairs |
| Gate 2 - paired count test | strong >= min_bases OR (weak >= min_bases AND paired_count >= count_k) | rule | paired bases | A pair needs either strong pairs or a majority of paired confirmations |
| Joint activation | Gate 1 PASS AND Gate 2 PASS | rule | every issue | Either gate failing yields FAIL; insufficient data yields NOT_EVALUATED, never zero |
| Synchronisation | monthly intersection with a single quarterly anchor; >= 2 matched anchors | count | every issue | A carried quarter cannot be double-counted |

#### A.4 Pool integration thresholds

How individual and joint layers are combined and the country cuts applied to the pooled score.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Pooled score model | XGBOOST_M01 | model id | every country-month | Boosted-tree probability of a stressed month; the only input to the country threshold |
| Threshold comparison operator | >= | operator | every country-month | Stress is raised when the score is at or above the country threshold |
| Additive joint filter | pkt_joint_CLIFS_VIX | packet flag | both members published before the decision month | Adds stress even when the country threshold is not crossed |
| Exclusion joint filter | pkt_joint_DFC_GFC | packet flag | both members published before the decision month | Removes stress raised by the other two layers (domestic-versus-global divergence) |
| Pool operator | selected pool, else COMPLEMENT_AND_JOINT; IND_ONLY fallback | rule | every issue | Combination of the individual and joint layers |
| Insufficient or conflicting signals | NOT_EVALUATED, no vote; empty complement = FAIL | rule | every issue | Missing evidence is never imputed as calm |
| Country decision threshold - BE | 0.0141416192054748 | probability | Ps_XGB_M01 >= 0.0141416192054748 | Country-specific cut on the pooled score; the only country-level parameter |
| Country decision threshold - DE | 0.013771653175354 | probability | Ps_XGB_M01 >= 0.013771653175354 | Country-specific cut on the pooled score; the only country-level parameter |
| Country decision threshold - ES | 0.0361300110816955 | probability | Ps_XGB_M01 >= 0.0361300110816955 | Country-specific cut on the pooled score; the only country-level parameter |
| Country decision threshold - FI | 0.0125994086265563 | probability | Ps_XGB_M01 >= 0.0125994086265563 | Country-specific cut on the pooled score; the only country-level parameter |
| Country decision threshold - FR | 0.0060475468635559 | probability | Ps_XGB_M01 >= 0.0060475468635559 | Country-specific cut on the pooled score; the only country-level parameter |
| Country decision threshold - IT | 0.0175201892852783 | probability | Ps_XGB_M01 >= 0.0175201892852783 | Country-specific cut on the pooled score; the only country-level parameter |
| Country decision threshold - NL | 0.9851701250299812 | probability | Ps_XGB_M01 >= 0.9851701250299812 | Country-specific cut on the pooled score; the only country-level parameter |

#### A.5 State and phase thresholds

Peak, trough and recovery are phase classifications in the executed code, not probabilities. Percentiles, quorums and persistence are reported as they are implemented.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Peak percentile p_E | 0.90 | quantile | phase classification | Peak branch requires a rise above this quantile of the normal-state reference |
| Peak percentile p_P | 0.75 | quantile | phase classification | Peak level test |
| De-escalation percentile p_D | 0.50 | quantile | phase classification | Falling below the median marks de-escalation |
| Recovery percentile p_L | 0.25 | quantile | phase classification | Lower reference of the recovery band |
| Tolerance percentiles p_H / p_X | 0.75 / 0.75 | quantile | phase classification | Exit tolerance band around the phase median |
| Phase quorum - ENTRY / PEAK / DE-ESCALATION / RECOVERY / EXIT | 3 / 2 / 2 / 3 / 3 | votes of 5 | every issue | K-of-5 bound rule: three bounds must agree for entry, recovery and exit, two for peak and de-escalation |
| Phase persistence | 2 | consecutive issues | every phase change | A phase is confirmed only when it repeats |
| Peak / Trough / Recovery as probabilities | not implemented | probability | requested probabilistic states | The executed path classifies phases by quantile, quorum and persistence. It emits no peak, trough or recovery probability and the three are not a distribution |

### Appendix B - RQ02 Transition forecasting thresholds
Forecasting asks three questions: whether stress starts, whether it ends and when the equity market reaches its bottom. Entry and exit are thresholded probabilities; the bottom is a zone rule.

#### B.1 Individual thresholds, long and short

Long-horizon features feed entry; short-horizon features feed exit and the bottom rule.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Long-horizon change window | 27 | months | entry features | Signed change over 27 months, z-scaled |
| Short-horizon change window | 6 | months | exit and bottom features | Signed change over 6 months, z-scaled |
| Long / short magnitude threshold | Q75 of strictly prior absolute values, at least 12 priors | z units | every issue | A movement activates only when it exceeds the prior 75th percentile and is positive |
| Long pair set | L_CG and L_DSR-P | pair id | entry features | The two long-horizon confirmation pairs |
| Short pair set | all 21 short pairs | count | exit and bottom features | Complete short-horizon pair set |

#### B.3 Pool and short-system thresholds

The five-vote short system and the last-short flag that feeds the allocation stage.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Short-system vote count | 5 | votes | every issue | Five independent votes must agree before the last-short flag is issued |
| Short-system proxy trigger | n_short_sources >= 4 AND n_short_pairs >= 5 | count | every issue | Transition-based trigger, not a level threshold |
| Last-short flag age | <= 18 | months | after the proxy trigger | The flag stays valid for eighteen months and is the input to the RQ03 lock |
| S5 vote universe | EW, IVOL, MVP, ERC, MaxDiv | portfolio | every issue | Five portfolio votes, three tail measures each, 30 series in total |
| S5 tail parameters | alpha 0.95 / 0.975 / 0.99; EWMA lambda 0.94; lookback 52 | level / weight / weeks | every issue | Peak detection uses the first argmax inside (H6, T] |

#### B.4 Entry and exit forecasting thresholds

Registered operating points are reported next to the selection rule the executed path actually uses.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Registered entry model | M11_LOGISTIC_L2_STRONG | model id | reported operating point | Logistic L2 model on the long-horizon confirmation set |
| Registered entry horizon | 18 | months | reported operating point | Forecast window of the entry question (H6 = 18 months) |
| Registered entry threshold | 0.111821 | probability | p_entry >= 0.111821 | Reported out-of-sample operating point |
| Registered exit model | LOGISTIC_L2_STRONG_PLUS_DURATION | model id | reported operating point | Logistic L2 model plus a duration term |
| Registered exit horizon | 3 | months | reported operating point | Forecast window of the exit question |
| Registered exit threshold | 0.027174 | probability | p_exit >= 0.027174 | Reported operating point, also used by the RQ03 phase overlay |
| Horizon-register threshold - H15 | 0.3196694994575123 | probability | p >= 0.3196694994575123 | Registered threshold for that horizon in the ladder; only H6 is used by the registered entry point |
| Horizon-register threshold - H12 | 0.2214215326280583 | probability | p >= 0.2214215326280583 | Registered threshold for that horizon in the ladder; only H6 is used by the registered entry point |
| Horizon-register threshold - H10 | 0.2166420964185038 | probability | p >= 0.2166420964185038 | Registered threshold for that horizon in the ladder; only H6 is used by the registered entry point |
| Horizon-register threshold - H8 | 0.1883593300131536 | probability | p >= 0.1883593300131536 | Registered threshold for that horizon in the ladder; only H6 is used by the registered entry point |
| Horizon-register threshold - H6 | 0.1594707667685234 | probability | p >= 0.1594707667685234 | Registered threshold for that horizon in the ladder; only H6 is used by the registered entry point |
| Executed entry horizon ladder | H15 / H12 / H10 / H8 / H6 = 45 / 36 / 30 / 24 / 18 | months | entry training | Four model families are fitted at each horizon |
| Executed entry selection rule | feasible if FNR < 0.10 and FPR < 0.20; then min FN, then min FP | rate | threshold search | Operating point chosen inside the feasible set |
| Executed exit horizons | 3 / 6 / 12 / 18 | months | exit training | Exit is fitted at four horizons; the phase overlay uses three months |
| Executed exit selection rule | feasible if FNR < 0.05 and FPR < 0.15; 0.5 confusion weight | rate | threshold search | Operating point chosen inside the feasible set |
| Exit-A and Exit-B | B = A + stacked RQ01 probability | model | exit models | Two exit models exist; the bridge that would score Exit-B is never called, so B is NOT_EVALUATED |

#### B.5 Asset-price bottom thresholds

Three independent factors, a two-of-three vote and an ex-post truth window.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| F1 - credit and stress level | ddCLIFS <= -0.15 AND zCISS >= 0.5, OR phase in {PEAK, DE-ESCALATION, RECOVERY} | level / z / phase | every issue | First of three independent bottom factors; the two branches are joined by OR |
| F2 - peer cooling | >= 2 of 6 peer countries in cooling phases | count | every issue | The own country is excluded from the peer set |
| F3 - volatility compression | sigma13 < sigma52 AND 4-week return > 0, at least 53 levels | vol / return / weeks | per asset E1, E2, E3, B1, B2 | Third factor, computed separately for each asset |
| Bottom vote and zone | F1 + F2 + F3 >= 2; zone L < U <= issue + 3 months | count / months | every issue | Two of three factors must agree before a bottom zone is issued |
| Bottom evaluation timing | issue = episode start + 3 months; truth = normal + 6 months, capped at the development end | months | evaluation only | The truth window is defined ex post, so the scored hit rate is not a trading signal |

### Appendix C - RQ03 Portfolio risk and restructuring thresholds
Allocation is a four-state policy. The Main Model is compared with five passive benchmarks built from the same six assets and with a fully invested equity portfolio.

#### C.1 Main model risk thresholds

Tail limits, feasibility and trading assumptions of the executed policy.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Allocation engine | four-state state-and-cycle allocator | — | always | State-and-cycle allocator: four states, each with its own weight rule |
| Normal-state equity target | 0.9 | share of NAV | state S1 (no lock, no bottom) | Sets the core equity weight; the rest of S1 is bonds |
| Normal-state bond target | 0.1 | share of NAV | state S1 | Bond sleeve B1/B2 split equally; cash is the residual |
| Lock-state bond share (base) | 0.65 | share of NAV | state S2 (short confirmed, bottom not yet confirmed) | Bond share of the locked portfolio at neutral probabilities |
| Lock-state stress tilt | 0.1 | share per unit of P_S | state S2 | Raises the bond share as the stress probability rises, inside a 50-80% band |
| State S4 high-volatility response | bonds absorb the shock | rule | state S4 with elevated z | S4 absorbs the shock through bonds rather than cash |
| Volatility-z direction | risk-off | sign | states S1 and S4 | Risk-off mapping: a higher volatility z-score lowers equity; no contrarian inversion is used |
| Post-bottom equity cap | 1 | share of NAV | states S3/S4 | Hard ceiling on equity once the bottom has been confirmed |
| Hard expected-shortfall cap | 0.02 | weekly return | every rebalance | Shrinks the target until the historical weekly ES is below the cap |
| Risk-tolerance scaling | off | flag | every rebalance | Disabled: the alpha/ES tolerance table is not applied to size |
| Bottom trigger rule | first confirmed bottom | rule | state S3 | ANY: the first confirmed bottom starts the bottom state |
| Bottom ES bypass | on | flag | state S3 | The hard ES cap is not applied inside the bottom state |
| Lock-state bond band | 50 - 80 | share of NAV | state S2 | Bond share is clamped to 50-80%; the complement is cash |
| Transaction cost | 10 | basis points | on every traded notional | One-way cost charged on realised turnover |

#### C.2 Benchmark construction

The five passive rules, their constraints and the relative band that was requested but not implemented.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| EW - base allocation rule | Equal weight | rule | every weekly rebalance | v_i = 1/5 inside the risky sleeve; long-only and fully invested across the five risky sleeves |
| IVOL - base allocation rule | Inverse volatility | rule | every weekly rebalance | v_i proportional to 1/sigma_i, SAMPLE_DEV estimator; long-only and fully invested across the five risky sleeves |
| MVP - base allocation rule | Minimum variance | rule | every weekly rebalance | argmin v'Av on the simplex, active-set and KKT; long-only and fully invested across the five risky sleeves |
| ERC - base allocation rule | Equal risk contribution | rule | every weekly rebalance | RC_i = 1/5 by Newton iteration; long-only and fully invested across the five risky sleeves |
| MaxDiv - base allocation rule | Maximum diversification | rule | every weekly rebalance | max sigma'v / sqrt(v'Av); long-only and fully invested across the five risky sleeves |
| Benchmark risk limits | none | limit | not part of the benchmark rule | The five benchmarks carry no VaR or ES cap and no state-dependent tolerance; the Main Model cap does not apply to them |
| Benchmark weight constraints | simplex | constraint | every rebalance | w >= 0 and sum(w) = 1; no per-asset cap and no relative band |
| Benchmark rebalancing and cost | weekly, 10 bp one-way | frequency / bps | weekly issue | Full re-solve to the exact target each week on the same cost schedule |
| Relative allocation band around base weights | not implemented | percentage points | requested +/-5 pp band | No such band exists in the executed or adapter code; the only 5 pp object is a research grid step |

#### C.3 Restructuring thresholds

State targets, gates, the profit ladder and the cycle reset.

| Parameter / Rule | Selected value | Unit | Activation condition | Interpretation |
|---|---|---|---|---|
| Pre-bottom residual split | pro-rata | rule | state S2 when the pre-bottom supplement is active | Residual after equity is split pro-rata between bonds and cash (65% to bonds) |
| Pre-bottom window | 26 | weeks | state S2 supplement active | Caps how long the early-equity supplement may stay on |
| Pre-bottom cooldown | 4 | weeks | after a pre-bottom exit | Blocks immediate re-entry into the supplement |
| Pre-bottom stop | 0.1 | return | supplement active | Cuts the supplement when its own equity basket falls by this much |
| Bottom confirmation: normal probability | 0.995 | probability | state S2 | One of the four gates that must all hold before bottom equity is deployed |
| Bottom confirmation: stress probability | 0.02 | probability | state S2 | Stress must have collapsed to this level |
| Bottom confirmation: volatility z-score | 2 | standard deviations | state S2 | Deep-drawdown regime required before full equity is bought |
| Bottom confirmation: drawdown tolerance | 0.05 | return | state S2 | Maximum remaining distance to the running peak accepted at confirmation |
| Bottom-state equity target | 1 | share of NAV | state S3 (bottom confirmed) | Moves the portfolio to a fully invested equity position |
| Recovery equity cap | 0.8 | share of NAV | post-bottom recovery path | Intermediate equity level when the cycle has not been reset |
| Recovery normal-probability gate | 0.98 | probability | recovery path | Normal probability required to leave the recovery allocation |
| Recovery stress-probability gate | 0.02 | probability | recovery path | Stress probability required to leave the recovery allocation |
| Post-exit equity floor | 0.5 | share of NAV | states S4 | Lowest equity weight the band can produce after a completed cycle |
| Post-exit bond ceiling | 0.25 | share of NAV | states S4 | Upper bound of the bond band in the post-exit state |
| Profit-taking ladder | +8% / 20%; +16% / 20%; +25% / 20%; +40% / 20% | gain / share | bottom position held | Four trim steps of 20% each at +8%, +16%, +25% and +40% |
| Tranche cooldown | 4 | weeks | between two trims | Prevents trimming twice in consecutive weeks |
| Minimum holding after the bottom | 52 | weeks | bottom position held | Keeps the bottom exposure for a full year before the ladder completes |
| Profit trail | 0.1 | drawdown from peak | after the ladder | Closes the residual position when it retraces this far |
| Runner activation | 2 | tranches completed | after the ladder | Number of completed trims after which a runner is kept |
| Runner trail | 0.15 | drawdown from peak | runner active | Wider trail applied only to the runner |
| Profit proceeds destination | on | rule | on every trim | Proceeds go to cash, not to bonds |
| Post-bottom stop | 0.1 | return | states S3/S4 | Stops the position if it falls this far from entry |
| Pre-bottom stop | 0.08 | return | states S1/S2 | General stop applied to the equity sleeve before the bottom |
| Stop reference | equal-weight equity basket | rule | whenever a stop applies | Stop is measured on the equal-weight equity basket, not on NAV |
| Pre-bottom ES overlay | — | rule | supplement active | NONE: the pre-bottom supplement is not additionally ES-constrained |
| Bottom supplement switch | on | flag | states S2/S4 | Enables the bottom-specific supplement that allows equity before confirmation |
| Forced bottom entry | off | flag | state S2 | Disabled; the bottom is only traded on confirmation |
| Cycle reset window | 26 | weeks | after a completed cycle | Consecutive normal-state weeks required to re-arm the bottom supplement |
| Cycle reset normal-probability | 1.01 | probability | after a completed cycle | Set above 1, so the probability branch never triggers and the reset is inert |
| Bottom lookback | 0 | weeks | state S3 | Zero: the tail trigger uses the whole available history |

### Threshold completeness audit
128 rows are published: 60 traced to a code line, 68 traced to a selected-configuration file, 0 unverified. Status counts: 122 active, 3 inert, 2 not implemented or not available.
| Id | Area | Severity | Finding | Action |
|---|---|---|---|---|
| D1 | RQ03 | high | The executed allocation engine (W16_S4BOT_Z2) is not part of this repository. The overlay module points to an owner hand-off tree, so the RQ03 numbers are traceable to the selected configuration dump and to the executed ledger columns, not to shipped source. | Ship the engine with the next release, or treat the parameter table as the authoritative record. |
| D2 | RQ03 | high | Two rows share the label 'Pre-bottom stop' with different values (0.10 and 0.08): one governs the pre-bottom supplement, the other the general equity sleeve. | Rename both rows so that each stop names the sleeve it protects. |
| D3 | RQ03 | medium | The adapter contract sets a risky-share envelope of 0.50-0.75, while the executed policy holds 0.90 equity in S1 and 1.00 in S3. The adapter is not the executed allocator. | Scope the adapter envelope to the research branch or remove it from the public tree. |
| D4 | RQ03 | medium | The requested +/-5 percentage-point band around each benchmark's base weights is not implemented. The only 5 pp object in the code is a capital-grid step used by a research audit. | State that benchmarks re-solve to their exact shape each week. |
| D5 | RQ03 | medium | The 2% weekly expected-shortfall cap cannot be reproduced from the ledger: the recording column is empty and three weeks are flagged as bottom-state bypass violations. | Recompute the cap from the engine and re-publish the ledger flag. |
| D6 | RQ03 | medium | VaR override triggers (95th and 99th percentile flags that push equity to 1.00) exist only as ledger outputs; their threshold values are not in the repository. | Publish the override thresholds or drop the claim that the rule is documented. |
| D7 | RQ03 | low | The cycle-reset probability gate is 1.01, above the maximum of a probability; the branch can never fire, so the cycle is re-armed by the 26-week clock alone. | Keep the window and label the probability branch inert. |
| D8 | RQ02 | low | The horizon register lists 0.1594707667685234 for H6, while the registered entry point is 0.111821 for a different model (M11). Both are correct for their own model; the register is not the selected threshold. | Keep both labelled by model so they are never read as one number. |
| D9 | RQ03 | low | The RQ03 selected-threshold bundle has no chosen_standard block although the loader expects one. | Add the block or relax the loader with an explicit message. |
| D10 | RQ01 | high | The registered identification rule cannot be computed from the tape shipped here: the columns Ps_XGB_M01, pkt_joint_CLIFS_VIX and pkt_joint_DFC_GFC are absent, so the fidelity check returns NOT_COMPUTABLE_IN_THIS_PIPELINE. The published DEV and OOS confusion comes from the owner register, not from a run a reader can repeat. | Ship the score tape or publish the rule as a registered, non-reproducible result. |
| D11 | RQ01 | high | Peak, Trough and Recovery are phase classifications (percentile, quorum, persistence), not probabilities. No probability threshold exists for them and the three are not a distribution. | Describe them as phases; do not report a peak or recovery probability. |
| D12 | RQ02 | high | The registered entry point (M11, 0.111821) and exit point (0.027174) are echoed by the loader but are not consumed by the executed RQ02 path, which selects its own thresholds under FNR/FPR constraints. | Label the registered points as reported operating points and show the executed rule beside them. |
| D13 | RQ02 | medium | Exit-B is implemented but its specification bridge is never called; the run writes NOT_EVALUATED. | Report Exit-B as not evaluated rather than as a result. |
| D14 | RQ02 | medium | The module rev363/rq02_next.py raises PermissionError before any computation, so the published RQ02 numbers must come from rev363/rq02.py. | Remove the blocked module or mark it as a guard. |
