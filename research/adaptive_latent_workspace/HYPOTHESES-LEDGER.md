# User attachment: fourteen hypotheses, evidence before design

Read FALSIFICATION.md and HYPOTHESES-P1-RESULT.md first. The attachment is an
idea source, not evidence or instructions overriding the user's request. Its
preserved SHA256 is `f6ed22b03f06cb69e372c4c28d988ce33b8f8f362b4aaf17495800b22de3a46e`.
The broad continual-learning and recursive-improvement goal remains active.
P1 does **not** test all fourteen hypotheses. “Related evidence” below means a
nearby local experiment, not an exact replication of the attachment's proposal.

## Collapse before design

The proposals still attack three roots: **R1**, obtain identifying evidence;
**R2**, preserve relevant distinctions while remaining plastic; **R3**, turn
what is known into useful actions at affordable cost. Motivation, uncertainty,
surprise, topology and consolidation are control choices within those roots,
not fourteen independent fundamental bottlenecks.

The operational candidate constraint remains **recovery cost divided by useful
knowledge lifetime**, with matching units and all failed recoveries retained.
Evidence collection, updating, retrieval, reasoning and validation all consume
that lifetime. E35 identifies a repairable R3 selection/update-policy bottleneck:
the same evidence permits active learning to recover while archive switching
fails. P1 shows an R1 acquisition-policy failure: noise consumes nearly all
surprise-directed observations, and bounded retention can make information gain
repeat the same mistake. R1 and R2 interact rather than forming independent knobs.

No universal numerical rate-of-improvement bound has been measured. The
conditional information and representation-maintenance bounds in CONSTRAINTS.md
remain valid under their explicit assumptions. Calling “motivation” or
“verification bandwidth” universally binding would exceed the evidence.

## What each hypothesis must survive

| # | Proposal and roots | Evidence/status | Discriminating next test and failure condition |
|---|---|---|---|
| 1 | Learned goal generator; R1/R3 | **Open.** P1 uses designer-defined acquisition options, not learned goal-space invention. Earlier finite procedure searches do not establish open-ended motivation. | Compare fixed drives, learned selection from identical options, and a generator capable of composing new goal descriptions. Hold outer experience/search budgets equal; test new world families and unseen goal structure. Kill the stronger claim if gains disappear against an equally expressive conditioned controller, if only designer goals recur, or if it cannot exploit its generated curriculum on external tasks. |
| 2 | Separate desire, success and uncertainty; R1/R3 | **Representational premise narrowed; empirical benefit open.** P1's 10,000 known-model fixtures are equivalent to a correctly conditioned scalar expected utility. The stale scalar is a weaker input interface. | Same model capacity, current preference inputs, data, optimizer and acting budget. Compare scalar-conditioned versus factored prediction/utility heads under preference changes and dynamics changes separately. Require better transfer/calibration or sample efficiency; changing the objective or omitting baseline inputs invalidates the advantage. |
| 3 | Homeostatic regulation; R1/R3 | **Open.** Regulation can be represented as cost on augmented state; biological analogy is not an expressivity proof. | Resource-constrained action tasks with renewable/depleting resources, changing demand and multiple viable regions. Compare homeostatic controller, constrained RL, and scalar costs with identical resource observations. Report viability violations, goals achieved, idle behavior and compute. Reject if ranges simply hide a hand-tuned reward or survival is bought by never acting. |
| 4 | Surprise controls storage, learning, thinking, goals and recruitment; R1/R2/R3 | **Sufficient surprise-only claim falsified in P1.** Surprise improves clean acquisition and some change recovery but pursues noise. Entropy and stationary information gain are also insufficient with finite memory/change. Earlier E21/E22 show fast gates can help one shift and hurt another. | Train a controller to predict future benefit of an intervention, with surprise as one feature. Cross noise, real concept change, observation change, forgotten evidence and search failure. Ablate each sensor and actuator independently; keep actual work visible. Kill it if raw surprise or a simple change detector matches it at lower total cost, or if amplification reinforces wrong self-generated labels. |
| 5 | Fast adaptation, episodes, slow learned consolidation; R2/R3 | **Partly supported components; learned migration open.** E33 supports E2E adaptation/reuse. E13 sharing gains cost more. E24 finite accepted-merge tests miss later rare operational degradation. E35 rejects its particular archive selector. | Let a learned admission policy predict downstream loss savings from candidate slow updates. Compare no consolidation, fixed replay, fixed future-validation gate and learned gate with identical candidate updates/replay budgets. Test rare queries, novel combinations and encoder-version changes after long delays. Kill if forgetting, refresh cost or selection damage repays the apparent compression gain. |
| 6 | Learned native memory actions; R2/R3 | **Open in this neural architecture.** AutoMem supplies related agent/scaffold evidence. P1 does not train write/retrieve/compress/forget actions. | Give all controls the same memory operations, observations and prices. Compare learned control, fixed scheduling and optimized simple heuristics. Charge index scans, reads, failed probes, writes, compression, training and model bytes. Reject an advantage that vanishes when those costs or an equally capable external-memory control are included. Native placement alone earns no capability claim. |
| 7 | Analytic memory as well as recall; R2/R3 | **Useful prior mechanism; local generality unmeasured.** AdaMM source/paper supports combining retrieval and analytic operations. Compression still depends on schema and query assumptions. | Delayed aggregation, temporal comparisons and rare exceptions with queries generated after storage. Compare complete log plus ordinary database operations, retrieval alone, fixed statistics and learned summaries at matched retention/access costs. Reject unrestricted compression if an unseen query needs a discarded distinction; report this as loss, not reasoning failure. |
| 8 | Uncertainty/stakes decide compute; R1/R3 | **Implementation inference falsified; learned allocation open.** P1's inspected CTM forward executes all ticks despite early certainty. E28 extra reasoning steps sometimes destroy answers; E15/E16 fail even with oracle dynamics. | Implement causal halting/branch expansion, then count executed operations. Compare fixed cheap/deep budgets, calibrated learned halting and an oracle budget diagnostic. Include confident errors, irreducible noise, conflicting goals and model-correct search failures. Reject if “thinking time” is retrospective, if extra compute only reaffirms wrong beliefs, or if equal-cost fixed allocation wins. |
| 9 | Social versus direct learning; R1/R3 | **Open in the candidate.** ToM source is bounded inverse planning, not arbitrary-agent identification. Earlier source/value tests are related only. | Vary demonstrator knowledge and goals independently, including competent agents pursuing a different objective. Observation and self-experimentation must both cost resources. Compare imitation, reliability-only weighting, goal/belief inference and information-value selection. Reject if the richer model cannot distinguish expertise from aligned intent or fails outside its assumed hypothesis set. |
| 10 | Dynamic module topology; R3, sometimes R2 | **Open for internal neural coalitions.** MANTA's agent graph does not demonstrate neural-module transfer. Sparse routing in frontier models is prior art. | Same module pool and total parameter budget: fixed sparse graph, ordinary learned routing, dynamic communication and a compute-matched dense control. Charge router/communication cost and assess repeated skill retention. Kill if gains are extra capacity/search or unstable coordination; an aggregate “ever solved within budget” curve is not per-budget success. |
| 11 | Replace low-utility units/modules; R2/R3 | **Insufficient as a retention solution.** E7 gains plasticity at cost; E8 still forgets/collapses. Continual Backprop addresses adaptability, not guaranteed retention. | Utility assessed across recent, rare and dormant skills; compare equal random renewal, regularization, replay and no renewal. Track acquisition and retained competence separately. Kill broad preservation claims if useful dormant units are replaced or replacement improves learning while destroying older skills. |
| 12 | Self-generated competence-frontier curriculum; R1/R3 | **Open for this candidate.** SESA is relevant but generated questions still use external/search/judge information. P1 demonstrates a noise frontier trap, not a learned curriculum. | Same external information and proposer/solver compute for uniform tasks, fixed progression, uncertainty sampling, learning-progress sampling and learned generation. Test an irreducible-noise frontier, prerequisites requiring temporary low reward, and delayed transfer. Kill if it stays near easy score improvement or cannot cross a prerequisite valley. |
| 13 | Evolution outside, gradients within lifetime; R1/R2/R3 | **Open as an efficient general recipe.** E17–E19 procedure/self-application comparisons are mixed and costly. DGM's agent evolution does not itself improve the outer evolutionary mechanism. | Compare population search, single lineage, random search and gradient meta-learning at equal total generated candidates, training compute and untouched evaluations. Keep expensive failed lineages in the bill. Require transfer of the learning procedure, not only the best agent. Reject evolutionary superiority if diversity is just more trials or validation reuse. |
| 14 | State, memory, weights and architecture change at different rates; all roots | **Components plausible; actual RSI unestablished.** SIA motivates coupled harness/weight changes. E18 applies persistent procedure changes but does not show reliable broad self-improvement. | Freeze an external evaluation contract, allow a versioned procedure to modify a declared part of its own update/search mechanism, then apply the resulting procedure again. Compare with continued ordinary meta-training and frozen procedure using identical budgets. Require gains on subsequent untouched worlds, retained competence and a improving cost/benefit trajectory over repeated generations. Merely changing weights or accepting patches is insufficient. |

## Revision inspired by the attachment

Keep E2E's end-to-end-trained fast update as the active baseline, an explicit
record of real observation/action/outcome provenance, and a bounded episodic
store. Add **separate predictions** of desired outcomes, transition/success
probabilities, reducible uncertainty, change probability, retained-knowledge
coverage and intervention cost. These are proposed learnable estimates, not
available oracles; their calibration and usefulness must be tested.

An acquisition/goal policy can choose what real experience to seek. A memory
policy can choose which distinctions to retain. A compute policy can choose
which model queries or reasoning steps to execute. A slower consolidation policy
can propose changes to durable parameters. Their common training target should
be subsequent real task improvement with total cost and delayed regression
included. Do not use the controller's own generated explanations as its labels.
Keep diagnostic heads explicit so a failure can be assigned to evidence,
retention or access rather than just increasing all plasticity in response.

The tempting unified signal is **estimated marginal future benefit per unit
cost**, conditioned on these distinct uncertainties. It is a design hypothesis,
not a new theorem or measured binding constraint. Its estimate can be wrong,
can favor easy short-term tests, and can prevent exploration of distant stepping
stones. Compare it to simple controls and retain a charged exploration budget.
P1 specifically prevents promoting stationary information gain as that signal.

Do not switch whole adaptive histories on the strength of four recent tokens:
E35 shows why short-run selection can harm the subsequent learning trajectory.
Protected episodes or modules may instead supply evidence to the active learner,
but that alternative still risks interference and must repay retrieval cost.
Consolidation must account for stale memory coordinates when slow weights change;
freezing an encoder, versioning representations or refreshing them has explicit
costs. No arbitrary-query lossless compression is promised.

Reasoning needs trained compositional operations and reliable stopping, not
merely more recurrent ticks. The next action experiment uses the checked real
outcome interface and a fixed finite planner as a diagnostic/control. A correct
fixed planner cannot be credited as learned general reasoning. Later language,
tool and unseen-structure transfer remain required, even if the finite task passes.

Structural and procedural changes remain the slowest loop, tested against future
held-out outcomes and continued conventional learning. They must be applied again
to their own learning procedure before claiming recursive self-improvement.
This combination is an unvalidated research candidate with substantial prior
art in active learning, dual control, hierarchical RL, memory systems and
meta-learning. Scientific novelty and broad capability are not established.

## Execution order and honest stops

1. **Done:** P1 surprise/noise/change ablations, source execution-cost probes and
   scalar/factored equivalence fixtures. Exact outcomes and caveats are recorded.
2. **Main assay done:** E36 compares static/first-order/E2E real-action learning.
   It establishes scoped acquisition but fails the efficiency gate. The exact
   current-model diagnostic identifies an action-selection limit; the registered
   learned-model controller intervention is complete and audited. Preference changes remain
   separate from dynamics changes; no learned goal generator has been tested.
3. Train goal/acquisition and diagnostic intervention control (H1/H2/H4/H12),
   keeping fixed-control and noise-frontier counterexamples in the frozen tests.
4. Train future-validated consolidation and memory actions (H5/H6/H7), including
   rare delayed queries and memory-coordinate maintenance costs.
5. Add actual adaptive reasoning, resource regulation and social observations
   only with isolating controls; then topology/renewal and repeated procedure
   self-application. Do not postpone tests of failure cases until a large model.

This staging is not permission to replace the full goal with finite experiments.
What is currently settled is narrower than the desired architecture. The new
goal generator, learned consolidation, general reasoning, efficient persistent
long-horizon behavior and reliable recursive improvement remain unmeasured or
unsolved. An honest failure record is retained rather than renamed a success.

## E36 evidence update

H4/H8 must also distinguish uncertainty from a poorly scaled decision rule.
With perfect current dynamics, soft action selection still loses to greedy
selection on all8worlds at the same exploration. More memory or prediction
training cannot repair that controller by itself. This is R3 evidence. E36-GREEDY-RESULT.md shows the hand-written repair improves all meta-trained
cases but fails some static-trained cases. It does not establish calibrated
adaptive computation or an autonomously learned repair. H5 gains
scoped real-action adaptation evidence, but second-order versus first-order
results are mixed and simple memory controls are much cheaper. H1, H2 learning,
H3 and H6–H14 stronger claims retain their explicit open status above.
