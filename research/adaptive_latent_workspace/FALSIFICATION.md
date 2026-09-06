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
| H1 | Context can be inferred well enough to reuse learned modules without task IDs | NOT_RUN | Prequential performance and return-task gain against context-conditioned replay |
| H2 | Temporary adaptation plus stable modules improves retention/adaptation tradeoff | NOT_RUN | Actual weight-learning ablation with comparable compute and memory |
| H3 | Consolidation lowers long-horizon resource cost without hiding rare failures | NOT_RUN | Total bytes, runtime, updates, rare metrics and capacity-exhaustion results |
| H4 | Learned dynamics support efficient long-horizon planning | NOT_RUN | Closed-loop task success with no privileged transition access |
| H5 | Plasticity can be sustained over many task changes | NOT_RUN | Long-duration runs, feature-renewal ablation and recurring-task retention |

No improvement is counted from oracle-selected heads, hindsight task assignment,
or metrics computed only after training on the same target. These may be labeled
diagnostics. A failing pilot must remain in the record even if the next version
fixes it.
