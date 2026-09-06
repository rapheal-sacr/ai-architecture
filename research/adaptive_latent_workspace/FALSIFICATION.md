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
| H2 | Temporary adaptation plus stable modules improves retention/adaptation tradeoff | PACKAGE_SUPPORT_ONLY | E2 bundles routing and isolation changes; matched-resource controls and component ablations remain |
| H3 | Consolidation lowers long-horizon resource cost without hiding rare failures | NOT_IMPLEMENTED | E3 exhausts slots; no compression or consolidation tested |
| H4 | Learned dynamics support efficient long-horizon planning | NOT_RUN | Closed-loop task success with no privileged transition access |
| H5 | Plasticity can be sustained over many task changes | LIMITED_DURATION_ONLY | E3 repeats four contexts over 1,048,576 observations; independent drift/plasticity and renewal tests remain |

## New failures and boundaries

- E1's unprotected module bank fails returning-context reuse against context-conditioned replay. Preserve [E1-RESULT.md](E1-RESULT.md).
- E2's isolated adaptation package reduces early returning-context MSE 63.4% against context replay across four fresh teacher seeds. It uses more model parameters, fewer stored replay bytes, and more model evaluations. This is evidence for the package on one synthetic family, not proof of the architecture. See [E2-RESULT.md](E2-RESULT.md).
- E3 gradual drift produces exactly the online learner's errors while costing roughly 23% more runtime. No context-history retention benefit is established there.
- E3 exceeds persistent capacity: thousands of repeated commit requests cannot receive a slot. Accuracy remains better on the screen, but routing/temporary-learning cost grows substantially. See [E3-RESULT.md](E3-RESULT.md).
- E3's million-observation case is supervised prediction on four recurring functions. It does not satisfy the user's requirement for efficient autonomous long-horizon tasks.

The broad goal remains active. No measured result currently establishes a universal binding constraint. In the tested streams, identification delay, destructive adaptation, and per-module scoring cost are separable limits. Their practical importance depends on context lifetime and capacity.

No improvement is counted from oracle-selected heads, hindsight task assignment,
or metrics computed only after training on the same target. These may be labeled
diagnostics. A failing pilot must remain in the record even if the next version
fixes it.
