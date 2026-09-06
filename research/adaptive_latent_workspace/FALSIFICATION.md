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
| H6 | The system improves its own learning or memory procedure, and that improvement transfers | MIXED_BOUNDED_SELF_APPLICATION | E9 uses its own learned updater on its procedure parameters; E9 transfer is mixed; E11 self-application beats equal-budget conventional meta-updates in 4/9 and 5/9 fresh settings; reliable superiority and cost-matched efficiency remain unestablished |

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
