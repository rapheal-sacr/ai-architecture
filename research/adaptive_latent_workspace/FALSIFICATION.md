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
| H3 | Consolidation lowers long-horizon resource cost without hiding rare failures | NOT_ESTABLISHED | E6 sample-tested merging costs more and worsens whole-stream error versus context replay; final-window error improves, so amortization remains open; rare retention untested |
| H4 | Learned dynamics support efficient long-horizon planning | NOT_RUN | Closed-loop task success with no privileged transition access |
| H5 | Plasticity can be sustained over many task changes | LIMITED_DURATION_AND_RENEWAL_SUPPORT | E7 renewal lowers whole-stream error with extra runtime; replay-only wins the final window; indefinite plasticity is not established |

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

The broad goal remains active. No measured result currently establishes a universal binding constraint. In the tested streams, identification delay, destructive adaptation, and per-module scoring cost are separable limits. Their practical importance depends on context lifetime and capacity. E4 further shows why novelty in observations must not be equated with a need for another conditional model. Representational underfit, distribution change and interference are candidate explanations to distinguish experimentally.

No improvement is counted from oracle-selected heads, hindsight task assignment,
or metrics computed only after training on the same target. These may be labeled
diagnostics. A failing pilot must remain in the record even if the next version
fixes it.
