# Research state — goal still active

Read `FALSIFICATION.md` first. No result establishes a scientifically novel,
general continually improving agent or efficient very long-horizon reasoning.
E1–E31 are complete and audited. E32 is running from frozen source7d4edc9.
E33 now runs from63baff8 on CPU; read E33-INTERIM.md and E2E-REVISION.md.
Read E32-PROTOCOL.md, E32-INTERIM.md and RUNS/ACTIVE-STATE.json for status.
E30 implements a bounded integrated language/memory/feedback pilot and fails
reliable acquisition and efficient improvement; read E30-RESULT.md.

## What is locally established

- E2/E5 support protected temporary adaptation for reuse of four conflicting
  hidden functions. The overwrite ablation loses that benefit, and wider replay
  controls do not eliminate it. Active-first routing saves substantial scoring
  work on the tested family.
- E3/E4 reject general transfer of that allocation rule: drift wastes work,
  capacity is exhausted, and a fixed-function input-shift problem favors sharing.
- E13 extends the exact E6 prefix to five million observations. Guarded sharing
  eventually improves mean cumulative error by 21%, but uses 2.42 times the
  forward examples, 1.64 times the time and 1.45 times stored tensor bytes.
  Efficient compression and rare retention remain unproved.
- E7 feature renewal improves whole-stream prediction at extra cost. E8 shows
  that it does not prevent closed-loop forgetting during failed adaptation.
- E10 repairs one PPO checkpoint setup by removing entropy pressure. E12 rejects
  that as a sufficient general repair: both fresh zero-entropy seeds collapse
  after harder-task training; only one reacquires the original task. Late
  collapse can happen within a stationary stage. Its exact cause is unisolated.
- E14 implements a learned neural dynamics/reward model with bounded replay and
  planning. Its protection/merge mechanism never activates. E15/E16 show that
  high-gravity search stays poor even with accurate dynamics and simple temporal
  proposal structure. Model accuracy and search are separate limits.
- E9/E11 implement bounded self-application on reset students, with mixed results
  against conventional extra meta-training. E17 carries those frozen procedures
  into persistent students: accuracy gains are modest, execution costs about
  2.27 times fixed Adam, and self-application comparisons remain mixed.
- E18 actually changes the procedure online while student weights, optimizer,
  context and bounded experience persist. Self-application beats equal-trial
  conventional meta-updates in all four input-shift cases but only one of four
  conflicting-function cases. It also loses to freezing in three conflicting
  cases. Accepted updates do not establish reliable future improvement.

These are scoped results on the recorded seeds and tasks. Read individual
`E*-RESULT.md` reports for controls, full costs and failures. E12's interim record
is historical and is superseded by its complete report.

## The current candidate and its boundary

`DESIGN-REVISION.md` proposes an ordered recurrent workspace, protected adaptive
state, reusable modules, query-sensitive episodic compression, learned dynamics
and persistent procedure state. Only subsets are implemented and tested.
`ONLINE-PROCEDURE-DESIGN.md` gives the implemented E18 procedure loop: propose
from already-observed update traces, test a temporary branch on subsequent
observations, retain incumbent operational predictions until admission, and
charge every trial. The comparator uses ordinary meta-updates at the same trial
budget. Neither mechanism invents arbitrary architectures or changes its own
evaluator.

E18's online self-application takes about 1.73 times frozen-procedure time and
3.82 times fixed-Adam time, plus inherited bootstrap. Its input-shift gain is
real within this experiment but is not cost-matched efficiency. Both branches
receive full supervised feedback; this does not provide their counterfactual
action outcomes in a closed-loop environment. Memory policy, model family and
admission rule remain fixed.

E19 now holds student/optimizer/context/replay/RNG state fixed and changes only
the procedure. The updater matters, but its benefit depends on world and horizon:
input-shift keeping wins 4/4 original-world continuations and 0/4 new-world
transfers on whole-stream error. New-world late error is better after a worse
start, without repaying that deficit at this horizon. This supports a changed
learning tradeoff, not uniform general improvement.

E20 finds a much larger conflicting-function gain from privileged context than
from width: 82.6% versus 9.5% less mean error. Correct context raises recovery
from 234/407 to 396/407 segments; width alone reaches 260/407. The oracle changes
information and encoding and also repairs replay assignment, so the result
locates a limit in the inference/assignment package rather than proving an
irreducible identification bound.

E21 completed those non-oracle changes. Faster decay plus post-outcome assignment
reduces conflicting-function error 63.5% and raises recovery from 195/408 to
385/408 at the same model-example count and stored learner state, with about
4% extra runtime. Input-shift error rises 21.8%. E22 now implements the learned
causal gate described in `INFERENCE-REVISION.md`. It changes online but loses to
fast/prior in every conflicting, independent and alternating world, and to
slow/prior in every input-shift world. Coupled changes give a small scoped gain.
At 1.95 times fast/prior runtime, it is rejected as a sufficient repair.

E23 implements regularized coefficient descriptors with the same student and
post-outcome assignment. They improve 38/40 matched raw-descriptor pairs and
reduce fast independent/alternating error 13.2%/9.9%. Runtime rises 9.0%, storage
396 tensor bytes, and each run adds 4,096 nine-by-nine solves. Counterexamples
remain; coefficients of a nonlinear function are still distribution-weighted
projections. This is established regression machinery and a stronger fixed
baseline, not autonomous self-improvement or a novelty claim.

E24 completed the actual E6 merge audit. All arms acquire rare competence.
Guarded sharing accepts ten merges from competent sources; none crosses the
frozen material rare-loss threshold on independent probes. That attempted
falsification did not occur. Its final rare error still rises 4.21 times from
acquisition and narrowly fails the 0.05 threshold in three of four worlds, while
beating both controls on mean final rare error. This is neither universal
preservation nor evidence that an accepted merge caused each later loss.

Three of four non-merging worlds retain a rare-competent persistent module while
the operational active prediction is not competent. E25 now tests input-dependent access trained only from stored observations.
Learned top-one routing reduces non-merging mean query error 45.4%/42.0% on
original/shifted support, winning every world versus active selection. Nearest
anchors recover much of the gain at less fitting cost. Learned fitting plus
8,192 queries costs 2.19 times nearest-anchor execution. Three one-expert merged
states cannot improve, and the remaining merged state trades rare accuracy for
better whole-query error. The next test must let the experts and memory change
while keeping the access comparison causal. The privileged best-module audit is not
an existing agent capability. No wider memory bank is justified before this
selection gap is addressed.

## Binding constraint and unresolved measurement

The operational candidate is **recovery cost relative to useful context life**,
charging identification, adaptation, replay, routing, search, consolidation and
procedure trials. E17 supplies a censored assay: self-applied procedures recover
in 115/228 conflicting segments versus 227/228 input-shift segments. About
82% versus 11% of segment life is consumed before recovery or censoring. E18
further shows that mean error and recovery frequency can favor different arms.

No universal single numerical rate bound has been measured. Identifying evidence,
retained distinctions and useful computation remain separable roots. A universal
claim without specifying tasks, queries, resources and competence would hide
that missing measurement. `CONSTRAINTS.md` gives the limited analytical bounds.

## What must change before confidence

1. Separate context-identification delay from representational capacity and
   destructive sharing. E17/E18's conflicting-function failures cannot be fixed
   merely by citing their easier input-shift gains.
2. Demonstrate memory compression with lower total cost and preserved rare/old
   queries outside the compression and admission samples.
3. Extend the censored recovery assay to closed-loop tasks and explicit resource
   budgets; supervised stream length is not autonomous planning horizon.
4. Show repeated procedure improvement against ordinary meta-training without
   losing retention, and repay bootstrap, rejected trials and duplicate work.
5. Establish general reasoning and a defensible novelty distinction against
   existing modular, recurrent, learned-optimizer and policy-proposal systems.

The broad goal remains unachieved. Current evidence supports narrow mechanisms
and provides concrete counterexamples to stronger interpretations.

## E26: online access gains survive continued learning, with substantial cost

All 24 continuations reconstructed their E24 prefixes exactly and advanced to
16,384 updates. Core model, optimizer, anchors and global RNG finish identical
across selectors in all eight source groups. Routing changes predictions only.
On non-merging sources, incremental top-one routing reduces mean stream error
19.9% and final rare-query error 47.4% against active selection, at 2.03 times
model examples and 1.83 times runtime. The simpler nearest cache helps rare
queries but worsens whole-stream error 16.5%. Against guarded sources, online
routing gives only 4.5% mean stream improvement (one strict seed win), worsens
mean final rare error 0.5%, and costs 1.22 times time. The continuation revisits
fixed laws; no new merge was accepted during scored continuation. It establishes
neither novel-task transfer nor efficient autonomous memory. See `E26-RESULT.md`.

The initial `reasoning_processor.py` foundation was untrained; E27 has now trained
and falsified its size-transfer claim. Its source adapter and functional
checks pass, including 96 independently validated graph references. A two-step
local processor cannot distinguish two distant sources requiring different
answers on a chain. That structural counterexample motivates a trained public-
size recurrence test; it does not demonstrate learned reasoning. Read
`REASONING-REVISION.md`. Full integration and architectural novelty remain open.

## E27: trained recurrence succeeds locally and fails size transfer

All twelve registered cases completed from `505da5d`; all 36 stage checkpoints
and three paired stream/replay histories pass audit. The shared 15,177-parameter
processor learns Dijkstra, Prim, then Dijkstra, with explicit requested-task IDs
and final parent targets only. No gold hint trajectories or target-derived
stopping times enter the candidate. Initial weights and current observations
match across two/public-N recurrence and no-replay/128-graph-replay arms.

No arm or seed meets the declared 80% random-N16 acquisition criterion after its
first Dijkstra or Prim stage. Final random-N16 Dijkstra correctness averages
27.8% short/no-replay, 19.4% short/replay, 59.7% size/no-replay and 63.9%
size/replay. Prim remains poor. Size/replay nevertheless solves all 72 tested
final N16 Dijkstra chains; size/no-replay solves 70/72. At N64, every final
arm/task/family cell has zero functionally correct graphs. High pointer accuracy
can conceal incorrect solutions: short/no-replay's final N64 random Dijkstra
parent accuracy is 83.4%, with zero optimal whole graphs.

Replay preserves or improves some earlier accuracy during the task switch, but
successful retention of the registered random task cannot be claimed where
acquisition failed. Public-N recurrence costs eight times forward training
message candidates and about 3.6–3.7 times training time at the same replay
choice. Size/replay uses about sixteen times short/no-replay message candidates
and 1.80 times persistent tensor bytes. More computation has not established
size-general algorithmic reasoning or an efficient autonomous agent.

This falsifies the current component's sufficiency, not recurrence in general.
Fixed training depth, optimization, representation and supervision remain
confounded explanations of transfer failure. `E27-RESULT.md` and its preserved
predictions include every failure. The next bounded diagnostic should change
inference depth on frozen checkpoints before changing training. No E28 protocol
or run is yet registered. Full integrated learning, compression, reliable RSI
and architectural novelty remain unproven.

## E28: extra recurrence directly destroys some learned answers

All twelve E27 final checkpoints completed the frozen-depth grid from `0105e35`.
All original predictions reproduce exactly; loaded checkpoint states, model
weights, memory, optimizer, RNG and source files remain unchanged. The grid
executes 20,736 graph queries and 1,548,288,000 forward message candidates.

Doubling public-N depth improves whole-graph accuracy in zero of 144 source/
task/size/family cells and worsens it in 24; the others tie. These are correlated
cells, not 144 independent experiments. Size/replay's N16 chain Dijkstra
accuracy falls from 100% at 16 steps to 27.8% at 32, 9.7% at 64 and 5.6% at
128. Size/no-replay falls from 97.2% to 18.1%, 15.3% and 12.5%. No frozen depth
restores N64 whole-graph performance in the reported primary comparisons.

Thus this failure is not solely a shortage of permitted inference steps: extra
steps can damage an answer on the same input with identical weights. Fixed
training-depth specialization, optimization and learned transition stability
remain unresolved causes. Randomized-depth training is a source-motivated next
hypothesis, not an established repair. Compare it with an equal-message-work
fixed-depth control before attributing any gain to the distribution of depths.
Read `E28-RESULT.md`. No result establishes general reasoning, efficient continual
learning or reliable recursive improvement.

E29 now compares fixed16, fixed24 and balanced variable16–32 recurrence on three
fresh seeds, with exactly matched message work for fixed24/variable. Its separate
preflight reproduces E27 baseline losses and state exactly. The complete E29 result and state audit now supersede the running-status record.

## E29: depth robustness improves, general reasoning still fails

All nine cases completed from frozen source `c8a0884`; all 27 stage checkpoints,
reconstructed replay histories and nine equal-work stage pairs pass audit.
Variable16–32 and fixed24 each use 150,945,792 forward training message
candidates, identical observations, parameters and memory. At 32 inference
steps, variable training gives 95.8% final N16 Dijkstra chain correctness versus
63.9% for fixed24. Two seeds improve; the first loses 4.2 percentage points.
This is a scoped stability gain, not a uniform one.

With public-N inference, final random-N16 Dijkstra correctness is 50.0% variable
versus 29.2% fixed24, but the cheaper fixed16 reaches 54.2%. Variable training
costs 1.36 times fixed16 training time and 1.50 times its message candidates.
No seed/arm meets the registered initial acquisition criteria; all final N64
cells are zero whole-graph correctness for every arm, task, family and policy.
Thus learning a range of depths does not establish a transferable algorithm.
Read `E29-RESULT.md`; it preserves all failures and all inference policies.

Further graph-depth variants cannot by themselves meet the full goal. A pinned
pretrained language core now passes CPU execution and a GPU low-rank-adapter
preflight. Base weights remain unchanged and saved adapters restore exact logits.
These are feasibility checks, not learned memory, reliable procedure improvement
or architectural novelty. The next work should define an integrated, genuinely
closed-loop memory-and-adaptation assay, retaining the source/falsification and
cost obligations in `GOAL-EVIDENCE.md`.

## E30: integration executes, but competence is not reliable

The observed-map baseline completes24/24 missions; full/bounded frozen language
completes6/24 and8/24; adaptive bounded completes9/24. Adaptation costs1.84 times
acting/update time and9.6% more actions than its frozen control. It succeeds on
both relocations while losing one larger-world success. Initial acquisition
fails on one of two seeds, and no neural case meets larger-world competence.
All96 mission states reconstruct; all192 online updates replay exactly.

Visible wrong-object and target-discarding errors show a use-of-present-evidence
limit. This is not solely unavailable memory. Frozen checkpoints should next
separate tutorial fitting, symbol/address binding and exploration before a new
adaptation mechanism is trained. No E31 protocol is frozen. Full architecture,
learned compression, reliable RSI, broad reasoning and long-horizon efficiency
remain unproven.

## E31: diagnose acquisition before redesigning retention

All18 frozen cells and4,293 queries are audited. Bootstrap original tutorial
rule fidelity reaches58.2%/76.5%, failing the mastery gate even on exposed
queries. Training damages one source's immediate-home choice16/32→2/32.
Pure renaming changes correct decisions, but compact IDs have counterexamples
and are not a sufficient repair. E30 tutorials share7/8 worlds across learning
seeds, and E31 fresh sets share2/4 worlds; report their dependence.

The next study should establish a competent learned execution baseline using
matched-example conventional optimization controls and disjoint fresh data.
That is necessary groundwork, not a replacement for the full architecture.
Read E31-RESULT.md. No E32 protocol is frozen or running.

## E32 interim — two of twelve cases audited, run continues

Four snapshots are independently audited. On32101 at1024 examples, batch1
with rate0.0001 reaches140/142 tutorial correctness and3/4 initial deliveries,
but0/4 larger-world deliveries. Its rate0.001 control deteriorates from129/142
to93/142 tutorial decisions and4/12 to0/12 deliveries with more fitting.
The other ten cases remain running; this is not the completed E32 result.
The imitation source audit separately establishes the need to preserve executed
actions versus advisory labels when sharing experience with a world model.

## E2E mechanism priority

The user emphasized E2E. E2E-REVISION.md maps its source mechanism to the
persistent candidate and preserves exact-recall/persistence limitations. A small
functional core passes gradient, causality and save/resume checks. E33-PROTOCOL.md
freezes ordinary/reset/carry meta-training controls before scoring; six
development updates and21 sessions replay exactly. Follow RUNS/ACTIVE-STATE.json
for live execution. E32 continues unchanged. The full goal remains unachieved.
