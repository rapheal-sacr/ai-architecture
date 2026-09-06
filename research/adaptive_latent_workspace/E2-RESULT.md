# E2: temporary adaptation improves inferred-context reuse

E1 failed to preserve returning contexts. E2 isolates adaptation from established modules while checking whether an existing model explains the newly observed outcomes. After a new temporary model consistently improves observed prediction, it can occupy a persistent slot.

The implementation and protocol were held fixed for four fresh teacher seeds after four development seeds. These are nonlinear networks trained through actual gradient updates; the correct function is not selected from a supplied finite library. Each arm/seed receives 65,536 observations.

| Arm | Stream MSE | Returning-context early MSE | Seconds | Parameters | Tensor bytes |
|---|---:|---:|---:|---:|---:|
| context_replay | 0.04263 | 0.09703 | 2.885 | 7300 | 480984 |
| module_bank | 0.02820 | 0.09776 | 2.414 | 13739 | 164934 |
| isolated_adaptation | 0.02288 | 0.03553 | 2.921 | 19984 | 239904 |
| oracle_modules | 0.01770 | 0.01096 | 1.926 | 19984 | 239904 |

The returning-context reduction relative to context-conditioned replay is 63.4%, with the same direction on all four fresh seeds. This supports H1/H2 for the tested abrupt recurring-regime problem. It does not establish a general continual-learning solution.

Equal observations and optimizer-update count are not equal compute or parameters. The candidate uses more model parameters than the replay baseline but less tensor state because it does not retain that baseline's replay buffer. It evaluates multiple models for routing. Local timings are indicative and not a controlled isolated-device benchmark; the exact operation/example and storage counts remain available.

The oracle still performs considerably better on returning contexts. No task ID or future target is given to the candidate. It must incur identification delay after a change; oracle-selected predictions are never substituted into its score.

Temporary isolation and altered routing/admission rules are bundled in this intervention. The comparison supports the package; further ablations are required to isolate individual causal contributions.

The next failure search is E3: more noise, short contexts, more contexts than module capacity, gradual drift and one-million-observation streams. Beyond E3, closed-loop planning, episodic memory compression, learned routing cost reduction, shared representation learning and an independent external benchmark remain required.

A train-only preflight found and repaired a nonexistent PyTorch API call before any E2 registered outcomes were produced. No protocol thresholds or completed E1 results were changed.
