# Falsification record — active

## Prior constraints, not inherited architecture

The previous research turn made progress by auditing the supplied experiments,
building a different finite-program prototype and identifying its failures. It
did not achieve the present goal of an efficient continuously learning
architecture. The present goal therefore remains active.

The earlier audit established that the historical recurrent input encoding loses
token order, two task families include gold intermediate targets, and the native
memory assay's readout loses non-null answers despite correct upstream selection.
These are reasons to build a clean neural assay, not reasons to reject recurrence
or learned memory in general.

The finite-program witness approach also failed as a general solution. Its
certificates did not transfer to a different weight model, its useful-information
assumption failed, and exhaustive query selection was too expensive. This new
architecture does not import that implementation or its finite hypothesis class.

## Claims under test

| ID | Hypothesis | Status | Evidence required to change status |
|---|---|---|---|
| H1 | Context can be inferred well enough to reuse learned modules without task IDs | SCOPED_SUPPORT | E1 failed; E2 improves returning-context error on four fresh seeds; no broad transfer claim |
| H2 | Temporary adaptation plus stable modules improves retention/adaptation tradeoff | SCOPED_ABLATION_SUPPORT | E5 protection ablation loses reuse; wider replay controls do not eliminate gain; E4 general transfer still fails |
| H3 | Consolidation lowers long-horizon resource cost without hiding rare failures | ACCURACY_AMORTIZES_COST_UNPROVEN | E13 five-million continuation improves cumulative error on both seeds but uses 2.42× forward examples, 1.64× runtime and 1.45× stored tensor bytes; rare retention remains untested |
| H4 | Learned dynamics support efficient long-horizon planning | SCOPED_CONTROL_GAIN_SEARCH_LIMIT | E14 twenty-step search improves 7/8 return settings at 19.63× forward examples; E15 accurate-model control still fails at high gravity, including horizon sixty |
| H5 | Plasticity can be sustained over many task changes | LIMITED_DURATION_AND_RENEWAL_SUPPORT | E7 renewal lowers whole-stream error with extra runtime; replay-only wins the final window; indefinite plasticity is not established |
| H6 | The system improves its own learning or memory procedure, and that improvement transfers | ONLINE_SELF_APPLICATION_TRANSFER_FAILURE | E18 implements online self-application with persistent students; it beats ordinary meta-updates in 4/4 input-shift cases but only 1/4 conflicting-function cases and costs 1.73× frozen-procedure execution; general and efficient recursive improvement remain unestablished |

## New failures and boundaries

- E1's unprotected module bank fails returning-context reuse against context-conditioned replay. Preserve [E1-RESULT.md](E1-RESULT.md).
- E2's isolated adaptation package reduces early returning-context MSE 63.4% against context replay across four fresh teacher seeds. It uses more model parameters, fewer stored replay bytes, and more model evaluations. This is evidence for the package on one synthetic family, not proof of the architecture. See [E2-RESULT.md](E2-RESULT.md).
- E3 gradual drift produces exactly the online learner's errors while costing roughly 23% more runtime. No context-history retention benefit is established there.
- E3 exceeds persistent capacity: thousands of repeated commit requests cannot receive a slot. Whole-stream accuracy remains better on the screen, but the final eighth is worse than context replay and routing/temporary-learning cost grows substantially. See [E3-RESULT.md](E3-RESULT.md).
- E3's million-observation case is supervised prediction on four recurring functions. It does not satisfy the user's requirement for efficient autonomous long-horizon tasks.
- E4 rejects efficient general transfer for the current rule: on both seeds of an independent fixed-function, changing-input generator, context replay has lower error and runtime. Mean candidate MSE is 0.11336 versus 0.08128; time is 63.53 versus 41.13 seconds. Eight persistent slots are exhausted and approximately 27,919 subsequent updates satisfy an unavailable commit. Read [E4-RESULT.md](E4-RESULT.md) before proposing a larger module bank.
- E5 supports protection within the abrupt-context family against wider replay controls. Copying temporary updates into the persistent source removes most of the benefit. Active-first routing preserves all recorded prediction metrics on four seeds while reducing runtime by 21.1% and forward examples by 47.5%. See [E5-RESULT.md](E5-RESULT.md). This does not repair E4 or establish calibrated novelty.
- E6's local replay plus anchor-tested merging costs 69.4% more time and incurs 17.6% more whole-stream error than context replay on two fresh external seeds. It makes 77/80 merges. Its final-ten-block mean error is 8.5% lower, so a late benefit is possible but cost amortization is unproven. It still ends with eight persistent modules. Rare retention and true semantic compression remain unmeasured. See [E6-RESULT.md](E6-RESULT.md).
- Architectural novelty is not established. Prior task-free replay, model-copy and gating implementations exist; read [NOVELTY-AUDIT.md](NOVELTY-AUDIT.md). The proposal must not be presented as novel merely because this repository's earlier designs differed.
- E7 imported feature renewal lowers mean error 38.4% without replay and 6.1% with replay versus the same optimizer without renewal, with 34.2%/28.8% more time. Both seeds agree on whole-stream direction, but replay without renewal has lower final-window error. This supports a mechanism in a limited port; it is not a published CBP replication or proof of sustained superiority. See [E7-RESULT.md](E7-RESULT.md).
- E8 completed the fixed-procedure closed-loop pilot. Both arms acquired two-room success (50/50 per seed), then four-room training lost nearly all old-task success (baseline 2/50 and 0/50; renewal 2/50 and 3/50) while achieving 0/50 on the harder task. Both reacquired the old task. After that return, harder-task transfer was 9/50 and 13/50 for baseline versus 22/50 and 29/50 for renewal. Thus renewal has a possible transfer benefit but does not solve retention during unsuccessful adaptation. Near-maximal action entropy suggests an objective/interference mechanism; E10 is a completed causal test showing that removing entropy with shared critic gradients repairs this failure in both development seeds. See [E8-RESULT.md](E8-RESULT.md).
- E9 completes a bounded learned-updater/self-application test with a persistent 65-parameter procedure. The two replicates admit 1/8 and 7/8 self-updates. Against their frozen predecessors, final query error improves in 2/9 and 9/9 distribution/horizon settings respectively. Those settings are correlated summaries, not independent replications. Inner learned-update runs cost roughly 3.1× fixed-optimizer runs, plus about 65 seconds bootstrap and 10 seconds self-trial work per replicate. A control with equal additional conventional meta-training is missing. This is mixed evidence for bounded self-application, not robust or open-ended recursive improvement. See [E9-RESULT.md](E9-RESULT.md).

The broad goal remains active. No measured result currently establishes a universal binding constraint. In the tested streams, identification delay, destructive adaptation, and per-module scoring cost are separable limits. Their practical importance depends on context lifetime and capacity. E4 further shows why novelty in observations must not be equated with a need for another conditional model. Representational underfit, distribution change and interference are candidate explanations to distinguish experimentally.

## Updated explicit requirement: recursive self-improvement

The user clarified that recursive self-improvement and persistent memory are
explicit goals. E1–E7 are fixed learning procedures tested by a human-directed
research process. E8 is also a fixed procedure. They do not demonstrate a system
improving its own learning algorithm, and this agent's manual experiment edits
must not be counted as the proposed architecture's autonomous improvement.

An outer learning process is therefore required. Its learned update or memory
policy must change persistently from experience, improve acquisition/retention on
fresh tasks after all meta-training and selection costs, and survive adversarial
distribution changes. Self-application of the learned updater is a further test,
not something established by ordinary meta-learning. A fixed candidate menu or
repeated tuning on the final evaluation set would not establish the full claim.

No improvement is counted from oracle-selected heads, hindsight task assignment,
or metrics computed only after training on the same target. These may be labeled
diagnostics. A failing pilot must remain in the record even if the next version
fixes it.

- E11 supplies the previously missing conventional meta-update control. It reuses the frozen E9 bootstrap states and proposal/admission budget but sets the learned multiplier to one. On new final tasks self-application beats this control in only 4/9 and 5/9 distribution/horizon means; it beats freezing in 4/9 and 8/9. There are no divergent final tasks. These correlated setting summaries do not establish reliable self-application superiority. See [E11-RESULT.md](E11-RESULT.md).

- E10 completed: restoring the same E8 checkpoints and removing entropy with shared critic gradients changes four-room success from 0/50 to 50/50 in both seeds while retaining 50/50 on the old two-room task. With entropy retained, critic detachment still fails; without entropy, detachment fails in one seed and succeeds in the other. Thus entropy pressure causes a failure in this setup, but critic protection is not a reliable general repair. All baseline evaluation episodes reproduce E8; floating training metrics are close but not exactly reproduced. E12 challenges the promising fixed objective from random initialization on fresh seeds and a new action chain. See [E10-RESULT.md](E10-RESULT.md).

- E13 completed the exact-prefix five-million continuation. Guarded sharing has lower cumulative MSE in both seeds (0.05894 versus 0.07864; 0.06976 versus 0.08417), reversing its E6 first-million deficit. Mean cumulative error is 21.0% lower, but forward examples are about 2.42×, elapsed time 1.64×, and stored tensor bytes 1.45× context replay. It ends with eight modules and 465/470 merges. Thus accuracy amortization is observed at this horizon; efficient total-resource superiority and rare retention are not. All four original prefix error/counter checks reproduce exactly. See [E13-RESULT.md](E13-RESULT.md).

- E14 completed the first neural world-model/planning/memory pilot. Twenty-step replay planning beats one-step replay in 7/8 paired stage means, with 19.63× model examples and about 7.59× elapsed time. High gravity remains poor, and initial acquisition varies substantially by seed. Guarded sharing never enters temporary adaptation, grows modules or merges; final checkpoint event lists are empty. Its lower memory is a smaller anchor budget, not demonstrated semantic compression. Independent random-action probes expose compounding prediction error, but those actions differ from the MPC control distribution. See [E14-RESULT.md](E14-RESULT.md).

- E15 supplies an explicitly privileged accurate-model reference to the same planner. High-gravity return remains poor at both horizons twenty and sixty, and sixty improves only 1/8 paired setting means at three times the model calls. Learned model error therefore cannot be the sole cause; the current search/proposal budget remains limiting. The privileged predictor is never supplied to E14 and does not count as learned reasoning. See [E15-RESULT.md](E15-RESULT.md).

- E12 interim: completed fresh seed 13101 is a counterexample to a universal entropy-off repair. It acquires two-room success (50/50) and initial four-room transfer (45/50), then collapses during four-room training to 0/50 on all three tasks and remains at zero through key-door and returning two-room training. Standard entropy also fails harder-task acquisition but reacquires the original task. The second seed remains running. See [E12-INTERIM.md](E12-INTERIM.md); no full two-seed conclusion is asserted yet.

- E12 completed: the interim above is historical. Both zero-entropy seeds finish four-room and key-door training with zero success on all three tasks; only one reacquires the initial task on return. Standard entropy reacquires the initial task in both seeds but fails harder-task acquisition. Both fixed entropy choices therefore fail as reliable continual-learning procedures. Late collapse can occur within a stationary stage; its exact cause is not isolated. See [E12-RESULT.md](E12-RESULT.md).

- E16 completed: at matched model-example counts, temporal blocks of five/ten improve 7/8 and 5/8 accurate-model setting means but only 4/8 and 3/8 learned-model means. High-gravity control remains poor; both blocked learned versions are worse there in both seeds. Simple temporal blocking does not repair the search limitation. Prior POPLIN already learns policies for action/parameter search, so learned proposals alone cannot establish novelty. See [E16-RESULT.md](E16-RESULT.md).

- E17 completed: frozen learned procedures retain modest prediction advantages over the three fixed-rate controls on persistent students, but cost about 2.27× the local execution time before inherited meta-training. Self-application improves only 2/4 conflicting-function streams and 3/4 input-shift streams versus conventional extra meta-training in each bootstrap replicate. Both self-applied procedures recover in 115/228 conflicting segments; the second frozen bootstrap reaches 121/228. No tasks diverge, but stronger universal retention, efficient learning and self-application superiority are not established. See [E17-RESULT.md](E17-RESULT.md).

- E18 completed: the procedure now changes online while student weights, optimizer, context and replay persist. Self-application beats both freezing and equal-trial conventional meta-updates in 4/4 input-shift cases, but only 1/4 conflicting-function cases. Its early mean advantage in conflicting functions reverses later; pooled recovery is 99/250 segments versus fixed Adam's 138/250 despite better mean MSE. Costs are about 1.73× frozen-procedure time and 3.82× fixed Adam, plus inherited bootstrap. All 28 new final checkpoints verify 2,304 persistent steps, bounded 512-example replay, unchanged frozen and changed online procedures. This closes an implementation gap while supplying a fresh failure of broad transfer and reliable improvement. See [E18-RESULT.md](E18-RESULT.md).

- E19 completed: keep/revert procedures start from identical E18 self-applied student/optimizer/context/replay/RNG states and run 4,096 further updates. Input-shift keeping wins 4/4 original-world continuations but 0/4 new-world transfers on whole-stream error. On new worlds it has worse first-quarter but better last-quarter error, without amortizing its initial deficit at this horizon. Conflicting-function keeping wins 1/4 original-world and 4/4 new-world cases, with recovery still poor. The procedure intervention matters, but it changes specialization/plasticity tradeoffs rather than establishing uniform improvement. All regenerated original prefixes match exactly. See [E19-RESULT.md](E19-RESULT.md).

- E20 completed: privileged correct-context encoding cuts conflicting-function mean error 82.6% at unchanged small-model capacity, versus 9.5% from widening the inferred model from 738 to 6,018 parameters. Recovery rises from 234/407 to 396/407 segments with oracle context, versus 260/407 with width alone. This identifies a major limit in the current inference/assignment/interface package; oracle information and encoding change together, and the experiment does not distinguish prediction-time inference from replay assignment. It is diagnostic access, not an agent capability. See [E20-RESULT.md](E20-RESULT.md).

- E21 completed: faster evidence decay plus post-outcome replay assignment cuts conflicting-function mean error 63.5% and increases recovery from 195/408 to 385/408 segments with the same 738 student parameters, 71,328 stored tensor bytes and model-example count, at about 4% additional runtime. It increases input-shift error 21.8% and loses on all four such worlds. This is a cheap scoped repair with negative transfer, not a universal forgetting rule or learned allocation. The next gate test must also break the current correlation between input-distribution shift and task family. See [E21-RESULT.md](E21-RESULT.md) and [INFERENCE-REVISION.md](INFERENCE-REVISION.md).

- E22 completed: the causal gate learns online but does not repair the tradeoff. It loses to fast/prior on all four conflicting, independent and alternating worlds, increasing mean error 6.6%, 12.3% and 22.8%, respectively; it loses to slow/prior on all four input-shift worlds. Coupled changes give a small 3.2% mean gain versus fast/prior (3/4 wins). Learned-gate execution takes 1.95× fast/prior time, including an additional backward pass per update. Gate movement is real but does not establish improved learning. All 100 final checkpoints and 80 paired raw reservoir/RNG comparisons pass audit. See [E22-RESULT.md](E22-RESULT.md).

- E23 completed: regularized linear coefficient descriptors improve 38/40 same-decay raw-descriptor pairs. At fast decay they reduce independent and alternating mean error 13.2% and 9.9%, respectively, winning all four seeds in each family; the fast input-shift and slow conflicting-function cells each contain one loss. Added cost is 9.0% local runtime, 4,096 nine-by-nine solves and 396 persistent tensor bytes. This supports a modest fixed representation repair, not nonlinear invariance, novelty or autonomous self-improvement. All 80 final states and 40 paired reservoir/RNG checks pass. See [E23-RESULT.md](E23-RESULT.md).

- E24 completed: all arms acquire rare-query competence, then operational rare error worsens. Guarded sharing accepts ten merges from rare-competent sources; none causes the frozen material rare-loss event on independent probes. The attempted immediate-merge falsification therefore did not occur. Its final rare error still rises 4.21× from acquisition and exceeds the competence threshold in three of four worlds, while remaining better than both controls on mean rare error. In three of four non-merging worlds, a stored module remains rare-competent but the active prediction is not, exposing a separate access/selection gap. All twelve final checkpoint/cost/probe audits pass. See [E24-RESULT.md](E24-RESULT.md).

- E25 completed: routing trained only on stored observations improves frozen non-merging memory on all four worlds under both query laws. Learned top-one cuts mean error 45.4% on original support and 42.0% after support shift versus active selection. Nearest-anchor recovers much of the same gain; learned fitting plus 8,192 queries costs 2.19× nearest-anchor execution. Three single-expert merged states cannot benefit, and the fourth improves total error but worsens original-support rare error. All 32 saved routers reproduce 64 complete query sequences with the frozen single-thread setting. This establishes a scoped access repair, not information recovery or continual routing. See [E25-RESULT.md](E25-RESULT.md).

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

## E30 preflight: integrated learning is not yet established

The delivery-world planner initially failed after an unannounced shelf move;
its observed-only reinspection fallback was corrected before scoring. It then
completed54/54 development missions. The reduced neural functional preflight
completed0/2 missions in every neural arm, versus2/2 for the planner. Exact
state/logit restoration, immutable base identity, paired-world and memory-boundary
checks pass; online adapter parameters change. None proves learned competence.
The new four-room tutorial design and its256 updates are declared external
scaffolding. E30-PROTOCOL.md records every pre-score correction and limitation.
The scored source is being frozen; no scored results exist at this point.

## E30: the first integrated pilot fails reliable acquisition and improvement

All eight cases completed from028bffe. All96 mission checkpoints reconstruct
exactly; all192 online updates replay to exact adapters, optimizers, memory,
learning RNG and losses. The observed-map planner completes24/24 missions in119
actions. Frozen full/bounded language completes6/24 and8/24; adaptive bounded
completes9/24 in641 actions versus585 for its frozen control, at1.84 times acting
plus update time. Common bootstrap adds306 teacher actions and512 gradient
updates. Adaptation solves both relocations but loses a larger-world success.
Registered step-penalized return improves2.15→2.59 overall, with one seed better
and one worse. Runtime has no registered utility conversion: this is a tradeoff,
not an established net-utility loss. The observed planner dominates both.

All neural arms fail initial acquisition on seed30101 despite full memory in
one control. Seed30102 acquires3/4 short missions but retains only1/2 unchanged
return successes. No neural seed/arm reaches3/4 larger-world success. Frozen
weights also fail returns, so not every failure is parameter forgetting.
Local diagnostics reveal wrong-object pickups and discarded targets while the
relevant facts are visible. Memory eviction is not a sufficient explanation.

Read E30-RESULT.md for all cells, costs, post hoc diagnostics and limits. This
fixed LoRA/LRU/PPO-style loop is not reliable RSI, learned compression or the
complete candidate architecture. Tutorial fitting, symbol binding, exploration
and credit assignment remain confounded; diagnose frozen checkpoints before
claiming successful learning followed by a transfer failure. The broad goal
remains unachieved.

## E31 preflight: frozen binding diagnostic, no new competence result

E30 tutorial mastery remains unverified. E31 will compare frozen base/bootstrap/
final adapters on identical observed facts with original, fresh random and compact
labels. Structural tie and irrelevant-world checks pass; nine neural preflight
cells preserve weights and RNG. The initial command omitted --preflight and was
rejected before queries; corrected execution passed. No scored E31 logits exist
at this source freeze. Read E31-PROTOCOL.md; it is not an autonomous-task result.

## Cross-seed dependence found during E31 reconstruction

E30's tutorial generator uses consecutive seed offsets: learning seeds30101 and
30102 share7/8 tutorial worlds. Operational test worlds differ. E31's fresh
world generator also shares2/4 exact (seed,size) world specifications between
these sources, while remaining disjoint from tutorials. These are correlated
corpora, not independent environment replications. Preserve the frozen runs
and report the overlap; use disjoint streams for future changed designs.

## E31: failed tutorial fitting and a counterexample to compact naming

All18 cells complete fromff78ad7. All4,293 queries, references, metrics, source
adapters and work reconstruct;648 identical-prompt repeats have exact logits.
Original bootstrap tutorial correctness is58.2%/76.5%; exposed-query accuracy
is59.9%/76.7%. Every cell fails the mastery gate. We cannot call fresh failure
transfer after demonstrated acquisition. On30101, training improves delivery
but damages immediate-home choice16/32→2/32 and worsens tutorial CE1.095→1.572.

Pure renaming changes correct decisions with unchanged facts/weights. Compact
IDs sometimes help, but hurt the first final adapter's fresh correctness
58.0%→40.7%; no compact cell passes mastery. This rejects compact renaming as
a sufficient repair. Some multi-edge categories contain only tied valid routes
and cannot establish difficult reasoning. E31-RESULT.md preserves all categories,
exposure counts, overlap, flips, costs and limits. Tutorial fitting, interference
and representation must be separated before adding another architecture module.
The full goal, novel design, learned compression and reliable RSI remain open.
