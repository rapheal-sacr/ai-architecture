# E18: online self-application helps input shift but fails conflicting-function transfer

This implements a bounded online self-application loop. Every learned arm starts from an E9 bootstrap procedure; student weights, optimizer moments, context and reservoir memory then persist throughout 2,304 batches (73,728 observations). Every 48 batches, online arms can propose a procedure change from an eight-update retrospective gradient and test it on sixteen subsequent batches. All trial predictions in the operational record come from the incumbent until admission. No hindsight replacement with the winning branch occurs.

| Procedure replicate | Stream seed | Family | Arm | Clean MSE | Admitted / 48 | Recovered segments | Seconds |
|---|---|---|---|---:|---:|---:|---:|
| 12101 | 18101 | conflicting_functions | adam_01 | 0.171127 | — | 29/65 | 3.175 |
| 12101 | 18101 | conflicting_functions | frozen_bootstrap | 0.167631 | — | 22/65 | 7.019 |
| 12101 | 18101 | conflicting_functions | online_meta_adam | 0.168231 | 22 | 21/65 | 12.029 |
| 12101 | 18101 | conflicting_functions | online_self_applied | 0.169478 | 8 | 22/65 | 12.222 |
| 12101 | 18101 | input_shift | adam_01 | 0.012361 | — | 65/65 | 3.207 |
| 12101 | 18101 | input_shift | frozen_bootstrap | 0.011838 | — | 65/65 | 7.060 |
| 12101 | 18101 | input_shift | online_meta_adam | 0.012146 | 18 | 65/65 | 12.093 |
| 12101 | 18101 | input_shift | online_self_applied | 0.010473 | 31 | 65/65 | 12.273 |
| 12101 | 18102 | conflicting_functions | adam_01 | 0.197342 | — | 40/60 | 3.209 |
| 12101 | 18102 | conflicting_functions | frozen_bootstrap | 0.195748 | — | 24/60 | 7.089 |
| 12101 | 18102 | conflicting_functions | online_meta_adam | 0.197671 | 18 | 21/60 | 12.048 |
| 12101 | 18102 | conflicting_functions | online_self_applied | 0.194635 | 8 | 30/60 | 12.171 |
| 12101 | 18102 | input_shift | adam_01 | 0.018724 | — | 60/60 | 3.195 |
| 12101 | 18102 | input_shift | frozen_bootstrap | 0.017571 | — | 60/60 | 7.076 |
| 12101 | 18102 | input_shift | online_meta_adam | 0.016789 | 27 | 60/60 | 12.139 |
| 12101 | 18102 | input_shift | online_self_applied | 0.016422 | 29 | 60/60 | 12.266 |
| 12102 | 18101 | conflicting_functions | adam_01 | 0.171127 | — | 29/65 | 3.175 |
| 12102 | 18101 | conflicting_functions | frozen_bootstrap | 0.163877 | — | 26/65 | 7.083 |
| 12102 | 18101 | conflicting_functions | online_meta_adam | 0.160015 | 30 | 25/65 | 12.197 |
| 12102 | 18101 | conflicting_functions | online_self_applied | 0.166420 | 7 | 21/65 | 12.094 |
| 12102 | 18101 | input_shift | adam_01 | 0.012361 | — | 65/65 | 3.207 |
| 12102 | 18101 | input_shift | frozen_bootstrap | 0.011648 | — | 65/65 | 7.042 |
| 12102 | 18101 | input_shift | online_meta_adam | 0.010893 | 34 | 65/65 | 12.062 |
| 12102 | 18101 | input_shift | online_self_applied | 0.010536 | 33 | 65/65 | 12.209 |
| 12102 | 18102 | conflicting_functions | adam_01 | 0.197342 | — | 40/60 | 3.209 |
| 12102 | 18102 | conflicting_functions | frozen_bootstrap | 0.190332 | — | 32/60 | 7.148 |
| 12102 | 18102 | conflicting_functions | online_meta_adam | 0.189217 | 25 | 29/60 | 12.097 |
| 12102 | 18102 | conflicting_functions | online_self_applied | 0.195401 | 6 | 26/60 | 12.171 |
| 12102 | 18102 | input_shift | adam_01 | 0.018724 | — | 60/60 | 3.195 |
| 12102 | 18102 | input_shift | frozen_bootstrap | 0.016933 | — | 60/60 | 7.077 |
| 12102 | 18102 | input_shift | online_meta_adam | 0.016465 | 32 | 60/60 | 12.168 |
| 12102 | 18102 | input_shift | online_self_applied | 0.016018 | 32 | 60/60 | 12.220 |

## Comparisons fixed before the test

For conflicting_functions, self-application has lower whole-stream error in 1/4 cases against freezing, 1/4 against conventional meta-updates and 4/4 against fixed Adam 0.01.
For input_shift, self-application has lower whole-stream error in 4/4 cases against freezing, 4/4 against conventional meta-updates and 4/4 against fixed Adam 0.01.

Across the four conflicting-function cases, mean self-applied MSE is 0.181483 versus 0.179397 frozen and 0.178784 with conventional meta-updates. The early mean advantage over freezing reverses during the later stream. Pooled recovery is 99/250 segments for self-application, 104/250 frozen, 96/250 conventional meta-updates and 138/250 fixed Adam. These dependent pooled segments are descriptive, not independent trials. The fixed optimizer can have worse overall MSE yet recover more often under the declared criterion.

Under input shift, mean self-applied MSE is 0.013362 versus 0.014497 frozen and 0.014073 conventional meta-updates. All arms recover in 250/250 pooled segments. The self-applied advantage grows in later quarters here, which supports a narrow online learning benefit. It does not repair the conflicting-function counterexample.

These four cases per family cross two inherited procedure bootstraps with two fresh worlds, not four independent full-pipeline replications. Segments and proposal rounds are dependent. Accepted proposals establish that the fixed admission rule selected them on subsequent observations; they do not establish a positive long-run improvement rate. Whole-stream, quarter and recovery results must be inspected separately.

The retention guard uses 128 old observed examples and can overlap replay training. It does not certify rare knowledge or arbitrary future queries. Retrospective meta-training deliberately uses already-observed data; subsequent trial labels do not exist in the proposal inputs. Exact replay of the actual pre-query student parameters is asserted at every scored proposal. The float64 preflight meta-gradient is -0.0638097403777455 versus -0.06380974037767384 by finite difference.

## Charged work

| Arm | Mean seconds | Student forward examples | Student gradient batches | Controller calls | Meta-gradient calls | Stored tensor bytes (range) | Largest recorded tensor footprint (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adam_01 | 3.197 | 221152 | 2304 | 0 | 0 | 76904–76904 | 76904–76904 |
| frozen_bootstrap | 7.074 | 221152 | 2304 | 13824 | 0 | 77164–77164 | 77164–77164 |
| online_meta_adam | 12.104 | 339424 | 3456 | 20736 | 48 | 77424–77424 | 249616–249616 |
| online_self_applied | 12.203 | 339424 | 3456 | 20928 | 48 | 77424–77424 | 249616–249616 |

Each online case also runs 768 duplicate trial student updates and 384 retrospective student updates, in addition to the 2,304 operational stream updates. Conventional and self-applied online cases receive the same student work and proposal/admission opportunities; self-application adds four outer-controller calls per proposal. The three fixed/frozen/online costs must not be called equal. The historical E9 bootstrap cost is inherited by every learned procedure, while identical Adam cases are reused across procedure replicates (28 newly executed cases, 32 reported combinations). Standalone and execution preflight smoke work is separately retained.

The reported footprint is the maximum of final stored tensors and the runtime's explicitly tracked tensor sample. The original runtime marker omits the final meta-RNG serialization, so its raw baseline value can be lower than final stored state; that raw field is preserved in the JSON. Neither measure includes all autograd intermediates, Python objects or full process RAM. The runner also retains unused outer buffers in fixed/frozen controls. These are partial implementation measurements, not an optimal memory comparison or a complete peak-memory measurement. Checkpoint and full trial-log file sizes are separately recorded. Student-example counts and meta-gradient calls are not complete FLOPs. Local wall times may overlap other research, and no energy or fixed-time comparison was performed.

Online self-application takes about 1.73 times frozen-procedure time and 3.82 times fixed-Adam time here, plus inherited bootstrap work. Its input-shift accuracy benefit is not a demonstration of cost-matched efficiency. Conventional and self-applied online inference times are close, making their different transfer behavior the more direct mechanism comparison.

## What this does and does not establish

Actual online procedure proposals, self-application and persistent student state now operate in the same restricted experiment. The measured comparisons determine their value; neither this implementation nor accepted generations prove efficient or open-ended recursive improvement. The controller family, evaluator and memory policy remain fixed.

A remaining attribution problem is that admission adopts both the changed procedure and its already-adapted student branch. The observed benefit belongs to that package. E18 does not isolate how much persists because the procedure itself became better, independently of the selected student trajectory. A continuation must hold student/optimizer/context/replay state fixed while keeping or reverting procedure weights, then evaluate fresh subsequent observations. Until then, the input-shift gain must not be described as proof of a generally improved learning algorithm.

Both branches can consume the same labels here because this is full-feedback supervised prediction. In a closed-loop environment different actions change subsequent observations; this experiment does not solve that counterfactual problem. It also does not implement a general reasoning workspace, semantic memory compression, reliable rare-fact recall or architectural novelty. Learned optimizers, replay, model copies and admission have prior art. The overall research goal remains unachieved.
