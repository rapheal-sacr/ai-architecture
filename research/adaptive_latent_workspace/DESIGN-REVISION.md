# Architecture revision after the mechanism tests

Read `FALSIFICATION.md` first. This is a proposed integrated system, not a claim
that the components below already form a validated or scientifically novel
agent. E1–E18 have implemented and tested only subsets. The original supplied
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
