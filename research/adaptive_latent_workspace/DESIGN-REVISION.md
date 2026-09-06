# Architecture revision after the mechanism tests

Read `FALSIFICATION.md` first. This is a proposed integrated system, not a claim
that the components below already form a validated or scientifically novel
agent. E1–E29 have implemented and tested only subsets. The original supplied
papers and model gallery were reviewed through the preserved WD corpus/clones.

## The change that the failures require

A module library is insufficient. E4 shows that surprise can reflect learnable
input shift rather than a new incompatible function. E8 shows that a learning
objective can destroy useful behavior even without acquiring anything new. E11
shows that a learned updater changing its own parameters does not reliably beat
ordinary extra meta-training. Capacity, more updates and self-application are
therefore not improvement criteria.

The proposed architecture must learn allocation of **evidence, retained state
and computation** against subsequent behavior. Its procedure state must govern
what signals change the learner, as well as step size. It must be possible for
an update to be rejected, shared or isolated on the basis of measured downstream
consequences. That decision is learned and fallible; it is not a certificate.

E10 completed. Removing entropy with shared critic gradients succeeds on both old and harder tasks in both development seeds. Removing entropy and detaching the critic succeeds in only one seed; keeping entropy fails in both. This does not justify a universal rule to delete entropy or block critic gradients. E12 supplies fresh counterexamples: both entropy-off seeds collapse by the end of four-room training and only one reacquires the old task. Neither fixed entropy choice reliably learns the harder sequence. Collapse can occur late within a stationary stage; its cause remains unisolated.

## Proposed states and computations

**Ordered recurrent workspace.** An encoder preserves token/observation order.
A shared recurrent core refines the current belief with input reinjection and a
bounded number of iterations. Recent observations, actions and outcomes enter
this state; no oracle regime label is available. The core produces predictions,
action proposals and estimates of which additional computation might change a
decision. Recurrence and adaptive depth come from TTC-LR/MoR; their mere presence
is not new and does not establish reasoning quality.

**Fast associative state and protected reusable modules.** A temporary neural
state adapts to the current trajectory. Slow modules retain transformations that
have proved reusable. Routing tries the active module first; alternatives are
considered when its predictions and task outcomes justify their cost. The learner
can share an update, keep it temporary, or recruit bounded capacity. E2/E5
support protection and cheaper routing in one family; E4/E6 prohibit making high
prediction error an automatic recruitment or consolidation criterion. TTT-E2E,
Titans and Macaron motivate the separation, not its correctness.

**An episodic memory with query-sensitive compression.** Keep observed
trajectories and rare outcome examples while learning compressed neural state.
A compression candidate is trained to preserve future decisions and multi-step
predictions, not merely average reconstruction or next-token loss. Evaluation
must include old, rare and changed query distributions outside the compression
sample. Weight, optimizer, index, anchor and archive storage all count. At a
fixed memory budget, arbitrary independent new facts cannot all remain exactly
recoverable; capacity allocation must expose which capability it gives up.
AdaMM/OaK suggest explicit operations where needed, while the supplied TTT
needle-retrieval failure prevents equating better language loss with recall.

**A learned dynamics model and bounded planner.** Predict outcomes of candidate
actions from observed history. Search only the learned model; the agent cannot
call a hidden transition function. Allocate recurrent depth and rollout horizon
where extra prediction can change the chosen action. Validate predictions against
subsequent real outcomes and charge mistaken plans, replanning and model updates.
NeSyFS motivates belief-conditioned foresight, but internally consistent model
rollouts cannot validate their own accuracy. E14 now implements a small neural dynamics/reward model with bounded replay and learned-only planning. It does not implement the full recurrent belief architecture. E15 shows that an accurate model alone does not repair the current high-gravity search failure.

**Persistent procedure state.** A learned controller proposes learning rates,
relative objectives, replay/admission allocations and computation budgets from
observed learning histories. Train it on subsequent acquisition *and* retention
under charged work. Preserve predecessor versions when testing self-applied
changes. The controller may update itself, but changing the evaluator or its
held-out data is outside that boundary. E9 implemented a narrow per-tensor
learning-rate controller, not this full allocation procedure; E11 gives no
reliable superiority over equal-budget conventional meta-updates. DGM and
RecHarness motivate an archive and explicit trial accounting; neither supplies
an independent source of truth.

## Which root each addition must pay for

| Mechanism | Root attacked | Required evidence against its strongest alternative |
|---|---|---|
| Belief inference and action-dependent information gathering | Identifying evidence | Better future decisions per observation than passive collection or action entropy; uncertainty reduction alone is insufficient |
| Temporary/slow separation and conditional sharing | Retention/interference | Retained old competence and new acquisition against shared replay under equal total memory; no oracle routing |
| Query-sensitive neural compression | Retention and useful computation | Smaller complete retained state with bounded losses on rare/old queries and lower amortized access cost than the uncompressed baseline |
| Active-first routing, adaptive recurrence and planning | Useful computation | Better task success under fixed total work, counting failed plans and routing; compare with a strong reactive policy and explicit-memory planner |
| Learned persistent procedure and self-application | All three through allocation | Better fresh sequential learning than equal extra conventional meta-training; count bootstrap, rejected trials, archive growth and controller inference |

## Binding quantity and measurements still missing

For a declared competence criterion, use the amortized cost of recovering useful
behavior divided by the useful lifetime of a context. Include evidence gathering,
updates, routing, replay, planning, consolidation and a declared amortization of
procedure training. Estimate a censored recovery distribution; an arm that never
recovers has an observed lower bound on its cost, not an omitted value. Measure
retention on recurrence separately from reacquisition.

This is a candidate operational bottleneck, not a proven universal rate law.
The tests currently expose different limiting roots in different environments.
No honest single numerical bound follows until the task distribution, future
queries, resource costs and competence threshold are specified and measured.
Extra entropy can even increase work while reducing competence: counting updates
as improvement would give the wrong sign.

## The falsification sequence before an architecture claim

1. E10's factor comparison and E12's fresh new-action/retention test are complete.
   A useful objective correction becomes a strong baseline, not an architectural
   achievement attributed to unused memory mechanisms.
2. E14 adds learned dynamics and planning; E15 shows a search limitation even
   with an accurate model. Test temporally structured action proposals at the
   same number of model evaluations before adding model capacity or depth.
   A future learned proposal memory should retain useful computations, with
   real-outcome checks against self-reinforcing model errors.
3. Add bounded episodic memory and neural compression. Distinguish retention,
   retrieval, routing and readout failures. Test rare-query loss under exhausted
   capacity and whether consolidation repays its complete cost.
4. Train the persistent procedure over sequential tasks without resetting skill
   state between tasks. A meta-learner evaluated only on reset students cannot
   satisfy this step. Test sharing versus protection across both incompatible
   functions and input-only changes.
5. Test several self-application generations with fixed external measurement,
   fresh final distributions and equal-budget ordinary meta-training controls.
   A flat or worsening rate is a result; it must not be relabeled as successful
   recursive improvement because some proposals were accepted.

The integrated algorithm, broad reasoning, reliable recursive improvement,
long-horizon efficiency and research novelty remain open. The evidence supports
specific mechanisms and rejects several broad interpretations; confidence in
the full design has not been earned.

## The next architectural distinction

Persistent memory must retain both learned information and useful ways to search. E15 makes the second role concrete: the planner can possess accurate transition knowledge yet fail to generate a useful action sequence at its search budget. A candidate procedure memory would learn compact temporal action proposals from prior planning/real outcomes, while preserving an exploration route outside its learned proposal family. This is proposed, not implemented or established as novel. A fixed temporal action-blocking control should first test whether search representation, rather than extra model calls, repairs the failure. Any learned version must then beat that simple control after its training, storage and proposal costs.

E16 is complete: simple temporal blocks do not repair high-gravity control at equal model evaluations. The exact-model improvements elsewhere do not reliably carry to learned models. The next design must retain uncertainty about both its prediction and its proposed computation. Learning policy proposals is already implemented in POPLIN; it is a baseline to credit, not a sufficient novelty claim.

## Online procedure result

E18 implements persistent student/optimizer/context/replay state together with
online procedure proposals, actual self-application and temporary trials on
subsequent observations. It improves input-shift learning against freezing and
ordinary meta-updates in all four cases, but loses to both in three of four
conflicting-function cases. It costs 1.73 times frozen-procedure execution and
3.82 times fixed Adam. The complete design must explain this split; accepted
self-updates alone do not provide the missing mechanism. Memory admission,
compression, belief inference and procedure family remain fixed in this subset.

## Inference and memory assignment before more capacity

E19 isolates the procedure intervention with identical student starting states.
Its mixed continuation and new-world results show that the procedure changes a
plasticity/retention tradeoff, not a universally improving algorithm. E20 then
finds that correctly identifying and encoding context helps conflicting tasks
far more than increasing width. The next implementation target is causal belief
formation and assigning experience to the appropriate retained conditional
model. E21 tests simple recency and post-outcome assignment controls first.
A prediction must use only past evidence; an observation can be written after
its new evidence arrives. That distinction is established inference practice,
not a novelty claim. Its actual benefit must survive fresh prequential tests.

Future latent-code or generated-weight memories must also be compared with LEO
and HyperCL, which already implement latent adaptation and task-conditioned
weight generation/retention. Those mechanisms do not independently solve
unannounced context inference or certify rare-query compression.

E21 completed: faster evidence and post-outcome assignment produce a large cheap
conflicting-function gain and a clear input-shift loss. `INFERENCE-REVISION.md`
therefore proposes learned causal allocation across evidence timescales, with
controls that break the original family/input-shift correlation. E22 implemented that gate and rejected it as a sufficient repair: it loses to
strong fixed controls at almost twice their runtime. E23 then obtains a modest
repair with regularized coefficient descriptors, at 9% extra time, while retaining
counterexamples. Neither result justifies attributing fixed filtering changes
to recursive self-learning. E24 returns to actual merge decisions and rare-query
retention; its probes are research instrumentation, not a new agent capability.

## Rare-memory audit changes the next intervention

E24 supplies no material rare-query loss in ten actual accepted E6 merges.
The finite checks do not certify preservation, but the attempted immediate
counterexample did not occur. Later rare answers deteriorate anyway. In three
non-merging worlds, rare competence remains in a persistent module that the
operational choice fails to select. The next addition should therefore learn
query-dependent access to retained models before expanding their number.

Fit any routing policy using observed memory records and later real errors,
not private audit targets or region flags. Count target construction, fitting,
stored routing state and query execution. Compare a simple distance-based
memory selector. First hold experts fixed to isolate access from information
loss, then test online continuation before claiming improved continual memory.
This is established mixture/retrieval machinery and is not itself a novelty
claim. Neither an input router nor a successful compression sample establishes
ordered reasoning or reliable recursive improvement.

E25 converts part of the privileged best-module gap into a learned input router
using only actual stored observations. It improves all frozen non-merging
worlds on two supports, but nearest anchors supply most of the access benefit
without neural fitting. Learned fitting has not amortized at 8,192 queries, and
routing does not restore a missing expert. The next integration must retain
query-dependent selection through actual model and memory updates. Model-slot
changes, stale competence labels, normalization drift and fitting cost are
explicit new failure modes. This remains a component of the proposed system;
ordered general reasoning and closed-loop efficient learning still require
separate implementation and falsification.

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

The new `reasoning_processor.py` is untrained. Its source adapter and functional
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

## E30 execution-core limitation

The first integrated language/memory/actual-feedback loop executes and its
updates replay exactly, but it does not support confidence in this design.
A conventional observed-map planner completes24/24 missions; the adaptive
language pilot completes9/24 at much higher cost. One seed fails even short
acquisition with full memory. Visible wrong-object and target-discarding errors
show that useful computation over present facts is a current limitation.

This pilot implements only explicit observed state, a conventional pretrained
interpreter and fixed adapter learning. It does not validate the proposed
latent workspace, learned compression, world model or procedure-admission loop.
Increasing memory capacity or invoking recursive improvement cannot stand in
for an execution core that reliably uses known facts. The next diagnostic must
separate tutorial fitting, symbol/address binding and planning before a revised
architecture is claimed. See E30-RESULT.md and NEXT-TEST.md. No novelty claim is
established by replacing one conventional component with another.

## E31 strengthens the execution-core objection

Frozen diagnostic queries show that tutorial mastery was never established.
Original bootstrap rule fidelity is58.2%/76.5%, with one seed's immediate-home
decisions damaged by training. Name changes alone affect answers; compact IDs
are not a sufficient repair and can make them worse. Retention proposals cannot
explain away failures on unchanged, exposed tutorial states. Establish a
competent learned execution baseline under matched-example controls before
attributing gains to new memory or procedure mechanisms. This does not validate
the full architecture or establish novelty. Read E31-RESULT.md.

## Preserve causal and advisory roles in integrated experience

The imitation source audit gives an executable counterexample to naively
reusing supervision trajectories as dynamics data: an expert label can differ
from the executed action that produced the saved next observation. A shared
episodic representation must keep observed state, executed action, observed
consequence, optional advisory target and provenance separate. Compression
that conflates those roles cannot safely support both world-model fitting and
policy supervision. This is a necessary semantic constraint, not architectural
novelty or an implemented learned-compression result. See
IMITATION-SOURCE-AUDIT.md. The frozen E32 experiment remains unchanged.

## E2E priority: train the update before claiming persistent competence

E2E-REVISION.md proposes an execution core trained through future fast-weight
updates, extended to persistent contexts and returns. E30–E32's LoRA is not
that objective. The small core passes causal, numerical-gradient and save/resume
checks. E33 compares ordinary, reset-E2E and carry-E2E training with acquisition
gates and strong task controls. This is not the full architecture or a paper
replication. Residual evidence, learned selection, planning and procedure
self-application still need integration/falsification; novelty is unestablished.

## E33 evidence supports update-trained state, with explicit limits

The small E2E core now acquires every tested context and uses carried state to
improve early returns. This is stronger evidence for the future-update objective
than E30–E32's conventional LoRA fitting. Training across persistent contexts
adds gains under longer lifetimes/interference but is not uniformly better;
the specialized transition-table control remains stronger overall. E34's frozen
return/archive interventions will distinguish retained function from adaptation
needed to use it. Oracle restored snapshots are diagnostics, not a deployed
memory router. Integration with evidence, actions and actual procedure changes
remains necessary; E33 is not completion of the candidate or novelty proof.

## E34 selects the next architectural question

The current E2E state retains useful function without additional gradients, but
interfering updates cause large functional loss. A preserved first-A snapshot
recovers much of it while holding slow weights/inputs fixed. This motivates
protected fast-state storage and observed-evidence selection. Oracle restoration
in E34 is not that selector. The next integrated mechanism must pay bounded
archive, routing, admission and update costs under unpredictable recurrence and
new contexts. Cold-context errors persist even with the correct snapshot. Read
E34-RESULT.md; a bank of snapshots alone is not a complete architecture or novelty.
