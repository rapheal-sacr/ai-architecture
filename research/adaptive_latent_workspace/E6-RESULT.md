# E6: local replay and guarded sharing

The frozen screen adds local observed-example replay to E5 active-first isolation, then ablates a sample-based merge rule. All arms receive the same stream per seed; the external case uses the pinned official generator.

| Case | Arm | MSE | Late MSE | Seconds | Modules | Merges | Tensor bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| abrupt | context_replay | 0.024411 | 0.016980 | 10.75 | 1.0 | 0.0 | 480984 |
| abrupt | active_first_isolation | 0.009556 | 0.004935 | 9.42 | 4.0 | 0.0 | 239904 |
| abrupt | local_replay_isolation | 0.010391 | 0.004793 | 12.90 | 4.0 | 0.0 | 264480 |
| abrupt | guarded_sharing | 0.010391 | 0.004793 | 12.65 | 4.0 | 0.0 | 264480 |
| external | context_replay | 0.094595 | 0.056616 | 40.06 | 1.0 | 0.0 | 427128 |
| external | active_first_isolation | 0.126366 | 0.099558 | 60.39 | 8.0 | 0.0 | 601668 |
| external | local_replay_isolation | 0.103906 | 0.070136 | 78.10 | 8.0 | 0.0 | 698436 |
| external | guarded_sharing | 0.111212 | 0.051814 | 67.84 | 8.0 | 78.5 | 620832 |

Late MSE is the final quarter of abrupt-context segments or the final ten external blocks. These are different declared windows; compare arms within a case.

For abrupt, guarded sharing changes error by -57.4% and runtime by +17.7% versus context replay. Relative to local replay without merging, its error changes +0.0%.
For external, guarded sharing changes error by +17.6% and runtime by +69.4% versus context replay. Relative to local replay without merging, its error changes +7.0%.

On abrupt contexts, the sample test admits no merges in these runs. Rehearsal therefore adds work without reducing module storage. The protected architecture still requires separate models for these conflicting conditional functions.

In the external stream, a successful anchor check is only evidence about the retained sample. It cannot establish preserved performance on omitted past inputs. Compare actual prequential and late performance above; the count of successful merges is not a count of useful improvements.

All replay observations, merge checks and temporary model state are charged. Capacity hits inherited from E2 count commit requests at a full bank; a later merge can recover capacity, so they are not a count of permanently lost tasks. Tensor storage excludes Python overhead, gradients and transient activations. Timing remains indicative local CPU time.

No rare-fact recovery or distribution-expansion assay has been run for these merges. This is not verified semantic compression or a complete memory solution. No general-reasoning or closed-loop agent result is supplied by E6.

A train-only preflight checked replay and a forced full-capacity merge with deliberately permissive test settings. The scored protocol retains its frozen, stricter thresholds. Implementation review added full-capacity merge checks before the protocol was committed and before any scored E6 run.
