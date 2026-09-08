# E36: acquisition improves; efficient action competence is not established

The source-frozen main run is complete and audited: 1,152 outer updates,
110,592 training transition uses, 73,728 real evaluation actions and 612
noninitial checkpoints. The nine models used 36,864 unique training records,
shared across three learning arms. Read FALSIFICATION.md before this report.
The full architecture goal remains active.

E2E training through future updates beats conventional static training followed
by the same test-time updates on goal completion in all 24 paired cases.
Full second-order training does not establish an advantage over first-order
meta-training. It also fails the declared efficiency comparison with cheap
observed-memory controls. These are predictive meta-training results, not
training on task return, learned goal invention or recursive self-improvement.

## Main results

Mean completed goals per 512 real actions:

| Controller | Stationary | Recurring | Drifting | Noisy |
|---|---:|---:|---:|---:|
| Static training + updates | 137.00 | 131.33 | 122.50 | 114.67 |
| First-order meta-training + updates | 185.33 | 164.00 | 166.83 | 145.83 |
| E2E meta-training + updates | 182.17 | 171.67 | 164.67 | 148.67 |
| E2E, updates disabled | 119.67 | 122.67 | 115.00 | 108.67 |
| Observed-edge BFS | 170.50 | 203.00 | 236.00 | 170.00 |
| Rolling counts, same soft planner | 180.00 | 153.50 | 169.50 | 154.50 |
| Constant action 0 | 170.50 | 175.50 | 168.00 | 134.50 |
| Constant action 1 | 169.00 | 172.00 | 168.00 | 136.50 |

Each neural cell contains three initialization seeds on two shared worlds;
each control cell has those same two worlds. These are not six independent
worlds. E2E beats first-order on 12 paired cases, ties one and loses 11.
Its recurring mean improves by 7.67 goals but stationary/drifting means fall.
There are too few worlds to infer a population-wide ordering. Different policies
produce different observations; equal training data do not imply equal
evaluation evidence. Full case tables and censored acquisition are in
E36-TABLES.md and results/e36_summary.json.

Actual transition accuracy supplies separate acquisition evidence: stationary
E2E reaches 97.66%, first-order 98.11%, static-plus-updates 78.87%. E2E reaches
80.05% recurring, 82.39% drifting and 81.45% noisy. With 20% uniform outcome
replacement, perfect-map expected actual accuracy is 83.33%; this is an
expectation, not an upper bound on each finite sample. Expected perfect-map
NLL is .718801. Private map queries cannot supply ground-truth labels to updates.

The task family has a major shortcut: each individual action is a Hamiltonian
cycle. Always taking either action reaches every goal, with expected stationary
completion 170.444 per 512 actions. Both constants were added before main neural
evaluation results were available and retained without choosing a winner per
world. Their 8,192 receipts and 312 cycle traversals are checked separately.
Goal count alone therefore cannot establish learning. See E36-CONSTANT-CONTROL.md.

## Memory and cost limits

The pre-action map criterion is at least 11 correct argmaxes among 12 pairs.
All six stationary E2E cases hit it, but all later fall below it. Under recurrence,
43/54 E2E segments hit it; all 43 later fail. Under drift, 46/54 hit it and 45
later fail. Never-hit segments remain censored. Mean restricted acquisition
fractions are .053 stationary, .664 recurring, .570 drifting and .066 noisy.
This measures what the current query interface expresses; it does not prove
irreversible erasure of information. In particular, shared KV context may affect
counterfactual map reads. No post-final-action probe or individual recovery wall
time was measured. First hits must not be described as stable retained mastery.

The model has 48,032 base parameters (192,128 float32 bytes) plus 28,672 persistent
tensor bytes: 24,576 fast-weight and 4,096 KV bytes. Counts use 288 tensor bytes
and at most 96 logical history bytes. BFS uses at most 288 logical record bytes.
These exclude Python and allocator overhead. Every neural decision queries all
12 state/action pairs, then performs a separate real forecast/update. Planning
charges six Bellman iterations, 432 transition terms per decision.

E2E observed acting time is about 500–516 times BFS and 25–27 times counts across
regimes. Mean per-model training times are approximately 117.84s static, 264.22s
first-order and 142.60s E2E in this implementation. Full second-order training
was not slower here; operation counters do not measure all autograd FLOPs, and
these timings do not establish a general complexity ordering. Training finished
before audit overlap; later action timing overlapped a separate one-thread audit.
Acting timers include environment/control/update overhead but exclude private
diagnostics and serialization. Main scorer total was 2,276.959s. Audit stage
times sum to 1,885.443s, not an elapsed time including overlaps or an energy cost.

## Post hoc planner falsifier

Giving the existing planner exact CURRENT transition probabilities, including
the true noise rate, does not repair action selection. All eight frozen worlds
improve when soft selection is replaced by choosing maximum-value actions with
the same 10% exploration. These are privileged diagnostics, not learned results:

| Exact-model controller | Stationary | Recurring | Drifting | Noisy |
|---|---:|---:|---:|---:|
| Original soft selection | 181.5 | 186.0 | 170.5 | 159.5 |
| Greedy, same exploration | 256.0 | 264.0 | 263.5 | 219.0 |
| Greedy, exploration disabled | 268.5 | 278.0 | 281.0 | 238.0 |

The original planner uses discounted goal-hit values with gamma .95 and
temperature .1. A one-step versus two-step deterministic goal can differ in
value by only .05, so soft selection substantially randomizes useful choices.
This is a repairable action-use cost under R3, independent of absent model
knowledge. It rejects a memory-only explanation of this gap. It does not show
that greedy choices improve a learned, imperfect model: that is the separately
registered frozen-checkpoint intervention in E36-GREEDY-PROTOCOL.md.

All 12,288 oracle diagnostic receipts were independently physically replayed;
the original soft probabilities agree with explicit NumPy Bellman calculations
within 2.77e-7. Greedy variants reuse the tested Bellman values. Only the current
map/noise rate is privileged; future goals and realized noise are unavailable.
Results and source hash are in results/e36_oracle_planner.json. This was designed
after inspecting main results and is not an untouched holdout evaluation.

## Evidence and decision

Main frozen source: 2d3eff32a79a1561ba76d98a81b6a7f10bc92c04.
Main complete SHA256: 0474bcdf5db7ddd06a7d0180d1c64e3d7d75fd2a7f4179eab64bfe49ea9a1a47.
Data SHA256: 3b13d16a22d0b80d4c52aeb332a75be958e7b94818434070262465c2d35f4d59.

Training replay reuses the training routine and matches all updates/logits and
optimizer checkpoints. Evaluation updates bypass the action wrapper. Physical
world/noise/goal receipts, NumPy policy calculations and count histories are
separately reconstructed. 5,760 serial map queries agree within 3.10e-6; policy
error is at most 8.76e-7. A separate metadata audit checks training/evaluation
totals, work and storage. Compact audit artifacts are in results/e36_*audit.json.
Main scorer and training-audit process handles were lost after an environment
reset; complete artifacts and independent audits establish completion. Their
original process exit codes are unavailable. Evaluation/metadata audits were
observed at exit0. No stale handle should be restarted.

Settled within this assay: meta-training improves actual acquisition over the
static-plus-update control; constant actions exploit the family; the soft
planner limits performance even with exact dynamics. Open: first-order versus
full E2E general superiority, useful compressed memory under changing queries,
general reasoning and repeated autonomous procedure improvement. Unmeasured:
large pretrained-model behavior, long-horizon transfer, full energy/amortization
and a universal improvement-rate bound. Do not promote the present integration
as an efficient or general self-improving architecture.
