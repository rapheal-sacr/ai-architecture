# E26: online routing preserves an access gain at substantial extra work

All 24 continuations completed from eight matched E24 sources. The entire 8,192-batch prefix was regenerated exactly, every historical prediction error reproduced, and model/optimizer/anchor/control state matched each saved source before its missing global reservoir RNG was carried forward. Each continuation then learns for another 8,192 updates (262,144 observations), reaching 16,384 persistent updates. Four historical worlds are shared across source policies and access arms; these are not 24 independent new worlds.

Predictions use only preceding routing state and current inputs. After real outputs arrive, the unchanged E6 learner updates, then observed records can train the router. All three selectors finish with exactly equal underlying model, optimizer, memory, control and global-RNG states within each source. The access interface changes the predictions while preserving the underlying learning policy.

| Source memory | Access | Mean stream MSE | Mean final rare-query MSE | Mean final common-query MSE | Mean seconds incl. bootstrap/audit |
|---|---|---:|---:|---:|---:|
| local_replay_isolation | active | 0.0017937 | 0.081998 | 0.084461 | 11.433 |
| local_replay_isolation | nearest_cached | 0.0020904 | 0.043245 | 0.004430 | 13.293 |
| local_replay_isolation | online_top1 | 0.0014366 | 0.043126 | 0.004267 | 20.873 |
| guarded_sharing | active | 0.0009977 | 0.059054 | 0.000850 | 10.802 |
| guarded_sharing | nearest_cached | 0.0009558 | 0.058891 | 0.000824 | 11.266 |
| guarded_sharing | online_top1 | 0.0009524 | 0.059329 | 0.000801 | 13.147 |

## What survives online use

- local_replay_isolation: online top-one changes mean stream error -19.9% (4/4 strict wins) and final rare error -47.4% versus active. Runtime is 1.83 times and model examples 2.03 times active. Versus nearest-cache, stream error changes -31.3% at 1.57 times runtime.
- guarded_sharing: online top-one changes mean stream error -4.5% (1/4 strict wins) and final rare error +0.5% versus active. Runtime is 1.22 times and model examples 1.26 times active. Versus nearest-cache, stream error changes -0.3% at 1.17 times runtime.

Non-merging memory retains a useful access benefit while its experts and reservoirs update. The learned router has lower stream error in all four such worlds than both alternatives. Refreshed nearest-anchor access lowers mean final rare error but raises mean stream error: a rare-retrieval gain is not automatically better performance under the actual observation frequencies. The learned router needs much more work and is not established as an efficient dominant solution.

Three merging sources remain one-expert cases throughout this continuation and produce exactly equal error sequences across access methods. Their online neural selector still retains a private RNG state even when no network is allocated; those bytes and execution overhead remain counted. In the remaining merging source, learned selection lowers stream error but slightly worsens final rare error versus active selection. Access still cannot restore information absent from every available expert.

Final rare competence remains poor in the second and fourth non-merging worlds despite routing. The continuation revisits the same fixed conditional functions and input means; it does not establish adaptation to new algorithms, new functions or a closed-loop environment. Region-balanced probes and the rare-event observation stream answer different questions. The best-region expert metric is a privileged single-expert diagnostic, not a bound on input-dependent combinations.

## Actual library changes

| Seed | Source | New events | New merges | Final persistent models | Live temporary model |
|---:|---|---:|---:|---:|---|
| 24101 | local_replay_isolation | 0 | 0 | 2 | False |
| 24101 | guarded_sharing | 0 | 0 | 1 | False |
| 24102 | local_replay_isolation | 74 | 0 | 2 | False |
| 24102 | guarded_sharing | 0 | 0 | 1 | False |
| 24103 | local_replay_isolation | 0 | 0 | 2 | False |
| 24103 | guarded_sharing | 0 | 0 | 1 | False |
| 24104 | local_replay_isolation | 46 | 0 | 2 | False |
| 24104 | guarded_sharing | 0 | 0 | 2 | False |

Event counts can include repeated checks rather than new knowledge. The actual changes above bound what this continuation tested. The dummy preflight additionally exercises temporary-expert and merge slot handling; passing that functional check is not broad evidence under repeated novel-task arrivals.

## All added work and retained state

| Source | Access | New operational + fitting expert examples | Router training examples | Router backward batches | Query distance pairs | Final counted core + routing bytes |
|---|---|---:|---:|---:|---:|---:|
| local_replay_isolation | active | 1,104,984 | 0 | 0 | 0 | 132,240 |
| local_replay_isolation | nearest_cached | 1,172,072 | 0 | 0 | 69,682,176 | 142,544 |
| local_replay_isolation | online_top1 | 2,239,784 | 540,672 | 8,448 | 0 | 139,716 |
| guarded_sharing | active | 1,048,688 | 0 | 0 | 0 | 82,650 |
| guarded_sharing | nearest_cached | 1,057,008 | 0 | 0 | 16,777,216 | 85,226 |
| guarded_sharing | online_top1 | 1,319,024 | 135,168 | 2,112 | 0 | 88,311 |

The eight required source reconstructions cost 89.55 seconds of replay, plus separately recorded checkpoint loading and prefix generation. Reconstruction is shared once per source, not charged three times or omitted. Original E24 work is inherited separately. Router bootstrap is included in the main time table, and actual loop time includes audit probes. Probe expert/router/distance work and times are separately retained; the model-example table excludes those audit examples. New core inference counters exclude predictions because each access method counts its actual prediction work separately.

Target construction recomputes expert errors on the observed router training batches, including every neural bootstrap batch. Nothing is treated as a free oracle label. Final state includes the core optimizers/anchors and routing parameters, Adam state, normalization/cache tensors and private RNG where retained. Full process and peak autograd RAM, complete FLOPs, energy and production latency remain unmeasured. Small CPU runtimes and tensor counts do not establish hardware-independent efficiency.

## Verification and architecture boundary

All 24 final checkpoints have 16,384 updates, finite persistent experts, matching source identity and exactly reproduced final probe values, core counters and router counts. All eight groups have exactly equal final underlying learning states and global RNG across selectors. The preflight verifies RNG isolation on each dummy step, valid slots, actual router training and exact saved/restored next-update behavior. Source reconstruction is a stronger check than assuming a serialized model contains its unstored RNG.

This integrates persistent online learned selection with real model updates and bounded replay. It is not recursive self-application, a new general learning algorithm, semantic compression or general reasoning. Gains coexist with cost and retention counterexamples. The next experiment should address the missing multi-step reasoning processor on externally specified problems, rather than continue tuning this supervised access assay.
