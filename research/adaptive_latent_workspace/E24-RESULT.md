# E24: no harmful accepted merge observed, but rare answers still deteriorate

All 12 cases completed: four fresh worlds paired across contextual replay, local replay with merging disabled, and unchanged E6 guarded merging, at a two-module cap. Each stream first teaches rare inputs frequently for 512 updates, then reduces their probability to 1/512 while common input means change. The conditional functions stay fixed. Every learner consumes 262,144 observations and persists for 8,192 updates; no oracle region or boundary enters training.

This experiment targets retained answers outside the samples used to approve compression. Its independent probes are never supplied to the learner. It does not establish general semantic memory or exact language-fact recall.

| Arm | Mean prequential MSE | Rare MSE after acquisition | Rare MSE at end | Final rare competent worlds | Module counts |
|---|---:|---:|---:|---:|---|
| context_replay | 0.008103 | 0.015201 | 0.109851 | 1/4 | [1, 1, 1, 1] |
| local_replay_isolation | 0.007749 | 0.012048 | 0.065703 | 0/4 | [2, 2, 2, 2] |
| guarded_sharing | 0.005747 | 0.012048 | 0.050673 | 1/4 | [1, 1, 1, 2] |

## The intended merge falsification did not occur

| Arm | Checks | Competent outgoing sources | Accepted merges | Material proposed rare losses | Accepted material losses |
|---|---:|---:|---:|---:|---:|
| context_replay | 0 | 0 | 0 | 0 | 0 |
| local_replay_isolation | 254 | 188 | 0 | 20 | 0 |
| guarded_sharing | 11 | 11 | 10 | 0 | 0 |

Guarded sharing accepts ten merges, all from sources competent on the held-out rare probes. None crosses the frozen material-loss rule: source rare MSE at most 0.05, replacement rare MSE greater than max(0.05, twice source error). This is a real negative result for the attempted falsification, not a lack of acquired competence or an inactive merge mechanism. It does not prove arbitrary future-query preservation. Smaller losses, accumulation and inputs outside these finite probes remain possible.

The non-merging arm proposes twenty replacements with material rare losses, all in one world, but never applies a merge by design. These are not observed erasures. They also do not show that the E6 anchor threshold would necessarily accept those proposals, since that arm deliberately disables acceptance. All checks remain in the complete record.

## Retained knowledge and operational access separate

| Seed | Arm | Final operational rare MSE | Best stored-module rare MSE (diagnostic) |
|---:|---|---:|---:|
| 24101 | context_replay | 0.052996 | — |
| 24101 | local_replay_isolation | 0.058118 | 0.009503 |
| 24101 | guarded_sharing | 0.052952 | 0.052952 |
| 24102 | context_replay | 0.031543 | — |
| 24102 | local_replay_isolation | 0.083279 | 0.076151 |
| 24102 | guarded_sharing | 0.047763 | 0.047763 |
| 24103 | context_replay | 0.222949 | — |
| 24103 | local_replay_isolation | 0.069953 | 0.015739 |
| 24103 | guarded_sharing | 0.050924 | 0.050924 |
| 24104 | context_replay | 0.131915 | — |
| 24104 | local_replay_isolation | 0.051459 | 0.036210 |
| 24104 | guarded_sharing | 0.051054 | 0.051054 |

Every arm first acquires rare-query competence in every world. Guarded sharing ends with 4.21 times its acquisition error, and three of four final errors narrowly exceed the frozen 0.05 competence threshold. Yet its mean final rare error is 53.9% lower than context replay and 22.9% lower than non-merging operational predictions. The record must preserve both relative benefit and absolute deterioration.

In three of four non-merging worlds, a persistent module remains rare-competent while the active operational choice is not. This directly establishes an access/selection gap on these queries. The best-module value uses audit targets to choose a model and is explicitly privileged; it is not reported as an agent capability. The existing router selects an active model using observed batch outcomes, while the rare region can be identified from the current input. A future input-dependent retrieval mechanism should be tested before attributing this gap to absent capacity.

Rare error can rise through repeated ordinary adaptation, subthreshold merge losses, or changing model selection. These endpoints and immediate merge checks do not isolate every later cause. The final checkpoint audit records how many rare examples remain in each module reservoir. Coverage correlations alone are not a causal test of a sampling repair.

## Resource accounting and validation

| Arm | Mean operational forward examples | Mean audit forward examples | Mean loop seconds, including audit | Mean audit seconds | Mean counted tensor bytes |
|---|---:|---:|---:|---:|---:|
| context_replay | 786,400 | 8,192 | 10.186 | 0.010 | 480,984 |
| local_replay_isolation | 1,196,728 | 89,088 | 11.960 | 0.083 | 148,770 |
| guarded_sharing | 1,052,984 | 21,120 | 10.781 | 0.020 | 82,650 |

The lower byte count versus contextual replay partly reflects its smaller configured replay budget. Guarded sharing and the non-merging control use the same per-module anchor budget and cap, but their learned trajectories and final module counts differ. Lower bytes alone do not certify preservation.

The audit transiently holds the outgoing model and 24,576 bytes of probe inputs/targets. Its model evaluations and timing are recorded separately and included in total study loop work. Parameter bytes of each outgoing model are recorded, but this is not complete peak RAM. Subtracting audit kernels does not remove every Python instrumentation cost. Checkpoint size/serialization time, data-generation time and inherited tensor/parameter peaks remain in the complete output. Full process RAM, energy, complete FLOPs and deployment costs are unmeasured.

Preflight exercises accepted merges and verifies that audited and unaudited predictions, final weights, anchors, counters and RNG are exactly equal. It checks outgoing models remain unchanged on merge steps, source/candidate mapping, prefix-stable data and exact read-only checkpoint restoration. All twelve scored final checkpoints match frozen source identity and counters, have 8,192 updates and finite weights, and reproduce every recorded final probe prediction exactly. The module/anchor bounds hold.

No new memory policy or reasoning component was introduced. E24 supplies a scoped pass for immediate accepted-merge safety on the registered probes, a failure of sustained rare-query competence, and a concrete retrieval gap. It neither certifies compression nor establishes efficient general continual learning. The next intervention should distinguish input-dependent access to already retained knowledge from actual information loss, using only observed memory records to train any router.
