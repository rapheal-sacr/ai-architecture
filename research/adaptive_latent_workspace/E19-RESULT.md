# E19: isolate the procedure from the selected student state

Each pair starts from exactly the same final E18 self-applied student, optimizer moments, context, replay and RNG. Only the procedure differs: keep the final updater or restore its E9 bootstrap predecessor. Both procedures are frozen while the student learns for another 4,096 batches (131,072 examples), reaching 6,400 global updates. Every original E18 self-applied case is included.

| Replicate | Source seed | Family | Continuation | Keep MSE | Revert MSE | Keep / revert recovered full segments |
|---|---|---|---|---:|---:|---|
| 12101 | 18101 | conflicting_functions | same_world | 0.186334 | 0.186947 | 15 / 17 of 110 |
| 12101 | 18102 | conflicting_functions | same_world | 0.214173 | 0.211374 | 23 / 31 of 104 |
| 12102 | 18101 | conflicting_functions | same_world | 0.190163 | 0.187324 | 7 / 13 of 110 |
| 12102 | 18102 | conflicting_functions | same_world | 0.216498 | 0.214446 | 16 / 25 of 104 |
| 12101 | 18101 | conflicting_functions | new_world | 0.239011 | 0.241536 | 5 / 6 of 97 |
| 12101 | 18102 | conflicting_functions | new_world | 0.227451 | 0.232983 | 23 / 26 of 108 |
| 12102 | 18101 | conflicting_functions | new_world | 0.235603 | 0.242440 | 2 / 5 of 97 |
| 12102 | 18102 | conflicting_functions | new_world | 0.236200 | 0.245851 | 20 / 22 of 108 |
| 12101 | 18101 | input_shift | same_world | 0.002740 | 0.005187 | 110 / 110 of 110 |
| 12101 | 18102 | input_shift | same_world | 0.003891 | 0.006631 | 104 / 104 of 104 |
| 12102 | 18101 | input_shift | same_world | 0.002434 | 0.004218 | 110 / 110 of 110 |
| 12102 | 18102 | input_shift | same_world | 0.003897 | 0.005989 | 104 / 104 of 104 |
| 12101 | 18101 | input_shift | new_world | 0.019491 | 0.018289 | 90 / 93 of 97 |
| 12101 | 18102 | input_shift | new_world | 0.028827 | 0.028771 | 107 / 108 of 108 |
| 12102 | 18101 | input_shift | new_world | 0.019273 | 0.018332 | 91 / 93 of 97 |
| 12102 | 18102 | input_shift | new_world | 0.031035 | 0.025913 | 107 / 108 of 108 |

## What keeping the learned procedure changes

conflicting_functions, same_world: keeping the final procedure wins 1/4 paired cases; mean MSE is 0.201792 versus 0.200023 after reverting.
conflicting_functions, new_world: keeping the final procedure wins 4/4 paired cases; mean MSE is 0.234566 versus 0.240702 after reverting.
input_shift, same_world: keeping the final procedure wins 4/4 paired cases; mean MSE is 0.003241 versus 0.005506 after reverting.
input_shift, new_world: keeping the final procedure wins 0/4 paired cases; mean MSE is 0.024656 versus 0.022826 after reverting.

Input-shift transfer has a horizon-dependent tradeoff. On new worlds, keeping the procedure has worse first-quarter mean MSE (0.057436 versus 0.044890), but better last-quarter MSE (0.010928 versus 0.013720). That late benefit has not repaid its initial deficit over this complete continuation. On same-world input shift, keeping is better throughout the quarters. Thus the procedure specializes to a learning/retention tradeoff, rather than simply becoming uniformly better or uniformly worse.

A keep/revert difference is caused by the procedure intervention within this fixed-start comparison, including its later interaction with the retained student and replay. It cannot be explained solely by starting with different selected student weights. However, a benefit confined to the original world is specialization; it does not establish a universally better learning algorithm. Reverting can also disrupt useful student/procedure co-adaptation, which this test does not separate from state-independent procedure quality.

The two procedure bootstraps share each of the two source/new worlds. Four cases per family/mode are dependent crossed cases, not four independent full-pipeline replications. Censored recovery and error are reported separately; improvements in average MSE must not hide unrecovered segments.

## Continuation and cost checks

All four original-world prefix checks match every observation, noisy target and clean target exactly, as well as clipped segment metadata. The originally sampled latent segment continues across the former stop; no artificial switch is forced. The first partial continuation segment is marked left-truncated and retained in per-batch error, but separated from full-segment recovery summaries. No failed full segment is omitted.

For fresh worlds, seeds 19101/19102 replace the generating teachers while all old learner and memory state persists. Clean targets and segment labels remain recorder-only. Starting non-procedure state and first predictions are exact matches. Final procedure tensors are asserted unchanged, and paired experience/context/RNG state is checked to remain identical despite different student weights.

| Arm | Mean seconds | Continuation student examples | Gradient batches | Controller calls | Final stored tensor bytes |
|---|---:|---:|---:|---:|---:|
| keep_final_procedure | 12.212 | 393216 | 4096 | 24576 | 71588 |
| revert_bootstrap_procedure | 12.179 | 393216 | 4096 | 24576 | 71588 |

There are 32 newly executed learner continuations. World data are shared across paired arms/replicates, so repeated learner exposures are not independent data acquisition. Dataset-generation time is stored in `world_checks`; each pair's restoration time is stored twice for reference and must be charged only once. The separate first-prediction check adds 64 student examples per pair (1,024 total), outside continuation counters. Prefix regeneration checks data without retraining source students. Standalone and execution preflights are additional dummy work.

All E9 bootstrap and E18 student/procedure-trial work remains inherited. Equal continuation work does not make the accumulated learned procedure free. Timings overlap the E20 CPU experiment, so they are descriptive. Complete FLOPs, process RAM and energy remain unmeasured.

This is a necessary causal attribution test in full-feedback supervised streams. It does not demonstrate semantic compression, general reasoning, closed-loop counterfactual validity, novelty or open-ended recursive improvement. E18's original failures and extra costs remain in the record regardless of this continuation.
