# E25: observed-memory routing recovers access but does not restore lost models

All 64 query cases completed. Eight frozen E24 memory states are paired across four selectors and two fresh query laws. Stored experts, optimizers and experiences are unchanged. Fitting uses only observed reservoir inputs and noisy targets; final query outcomes, true region flags and E24 probe labels do not train or select a router. Three merged states have a single expert and explicitly bypass all routing work. The final non-merging state for seed 24104 also retains a temporary expert; it is included and charged.

| Source memory | Query law | Selector | Mean MSE | Rare MSE | Common MSE |
|---|---|---|---:|---:|---:|
| local_replay_isolation | original_support | active | 0.044054 | 0.063947 | 0.032216 |
| local_replay_isolation | original_support | nearest_anchor | 0.025773 | 0.034943 | 0.020269 |
| local_replay_isolation | original_support | learned_top1 | 0.024067 | 0.034667 | 0.017685 |
| local_replay_isolation | original_support | learned_mixture | 0.027526 | 0.034007 | 0.023670 |
| local_replay_isolation | shifted_support | active | 0.117991 | 0.199383 | 0.068864 |
| local_replay_isolation | shifted_support | nearest_anchor | 0.081380 | 0.104349 | 0.067607 |
| local_replay_isolation | shifted_support | learned_top1 | 0.068450 | 0.101760 | 0.048539 |
| local_replay_isolation | shifted_support | learned_mixture | 0.077486 | 0.100189 | 0.063875 |
| guarded_sharing | original_support | active | 0.036493 | 0.050049 | 0.028525 |
| guarded_sharing | original_support | nearest_anchor | 0.020246 | 0.052121 | 0.001044 |
| guarded_sharing | original_support | learned_top1 | 0.020510 | 0.052819 | 0.001041 |
| guarded_sharing | original_support | learned_mixture | 0.020425 | 0.052608 | 0.001034 |
| guarded_sharing | shifted_support | active | 0.101118 | 0.161766 | 0.064570 |
| guarded_sharing | shifted_support | nearest_anchor | 0.065425 | 0.160965 | 0.008095 |
| guarded_sharing | shifted_support | learned_top1 | 0.065616 | 0.161476 | 0.008095 |
| guarded_sharing | shifted_support | learned_mixture | 0.065538 | 0.161266 | 0.008096 |

## Access repair and its counterexamples

- local_replay_isolation, original_support: learned top-one changes mean error -45.4% versus active (4/4 strict wins) and -6.6% versus nearest-anchor.
- local_replay_isolation, shifted_support: learned top-one changes mean error -42.0% versus active (4/4 strict wins) and -15.9% versus nearest-anchor.
- guarded_sharing, original_support: learned top-one changes mean error -43.8% versus active (1/4 strict wins) and +1.3% versus nearest-anchor.
- guarded_sharing, shifted_support: learned top-one changes mean error -35.1% versus active (1/4 strict wins) and +0.3% versus nearest-anchor.

Non-merging memory benefits from input-dependent access on every world and both supports, without changing any expert. The simpler nearest-anchor selector recovers much of the gain, so neural fitting is not the sole explanation. Top-one routing improves aggregate MSE versus nearest-anchor, but nearest-anchor wins individual comparisons, including shifted seed 24101 and both supports of the only nontrivial merged source. The learned mixture does not uniformly improve on top-one selection.

The three one-expert merged states produce exactly the same predictions under every selector. Their missing accuracy cannot be recovered by selecting a different retained model. The remaining merged state has much lower whole-query error with routing but higher rare-query error on original support: top-one increases rare MSE from 0.049869 to 0.060950. Whole-query improvement therefore does not establish rare-query preservation.

Non-merging seed 24102 retains high rare error under all selectors. Access does not restore competence that is absent from the available experts. All arms also face higher errors after support shift; new query frequencies and shifted coordinates do not create newly learned expertise. Eight frozen source states and correlated query summaries do not establish broad transfer or statistical significance.

## Complete costs distinguish a learned router from cheap access

| Source | Selector | Fit + target construction seconds | Query seconds / 4096, original support | Additional persistent tensor bytes |
|---|---|---:|---:|---:|
| local_replay_isolation | active | 0.000000 | 0.016663 | 0 |
| local_replay_isolation | nearest_anchor | 0.000874 | 0.051204 | 11,584 |
| local_replay_isolation | learned_top1 | 0.140771 | 0.042762 | 2,267 |
| local_replay_isolation | learned_mixture | 0.177396 | 0.039426 | 2,267 |
| guarded_sharing | active | 0.000000 | 0.016665 | 0 |
| guarded_sharing | nearest_anchor | 0.000157 | 0.025382 | 2,576 |
| guarded_sharing | learned_top1 | 0.034912 | 0.022975 | 554 |
| guarded_sharing | learned_mixture | 0.044052 | 0.021909 | 554 |

For non-merging states, top-one query time is 2.57 times active selection despite the same selected-expert example count. It uses less routing-cache memory and query time than nearest-anchor on these means, but fitting plus both 4,096-query laws costs 2.19 times nearest-anchor execution. This finite assay does not repay the neural fitting cost relative to the simpler selector. The listed extra bytes are added to the full inherited E24 state; they are not total model memory.

Target construction evaluates all K experts on every retained observation. Nontrivial neural routers add 256 backward/optimizer steps and 16,384 router training examples. Top-one evaluates one selected expert per query; the mixture evaluates every expert. Nearest-anchor computes every query-to-memory distance pair. All counts, transient training-prediction cache bytes, router/Adam state, source-loading time and inherited E24 work are retained. Fitting is shared across the two query laws and must be charged once per source/selector, not twice and not zero times. Privileged diagnostic evaluations and their timings are separate from operational queries.

Full process memory, peak autograd workspace, complete FLOPs, energy and eventual deployment amortization are unmeasured. Millisecond CPU query loops provide local comparisons, not production latency guarantees. The original acquisition and E24 memory costs are inherited, not erased by starting this experiment from checkpoints.

## Audit and next requirement

With the frozen single-thread CPU setting, all 32 saved selector states reproduce both complete 4,096-query error sequences and operation counts exactly after restoration. An initial audit using the default thread setting failed exact floating equality; restoring the experiment's single-thread setting resolves it without changing any scored source or result. Source checkpoint hashes, unchanged expert tensors and source counters are checked. One-expert bypass predictions are exactly equal to active selection. Preflight exercises actual learned parameter changes, nearest-anchor recovery on stored dummy inputs, exact restored predictions and a double-precision finite-difference mixture gradient. The two fresh query laws never supply training labels.

This closes a scoped causal access gap. It does not demonstrate continual routing under changing experts, repair information loss, learn a new world, improve its own learning algorithm or establish ordered general reasoning. Mixture-of-experts and nearest-neighbor routing are established mechanisms, not novelty claims. The next integration must test this access mechanism as models and memory change, against the cheaper selector, and then move beyond supervised retrieval to the missing ordered reasoning and closed-loop components.
