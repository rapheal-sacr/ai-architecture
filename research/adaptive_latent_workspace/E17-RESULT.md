# E17: reset-trained procedures transfer modestly, with higher execution cost

Students, optimizer moments, inferred context and 512-example reservoir replay persist across 2,048 updates (65,536 observed examples). Unannounced regimes last 16–64 batches and recur unpredictably. One family changes the nonlinear target function; the other changes input distributions under one fixed function. Four fresh stream seeds are paired across two inherited procedure-bootstrap replicates and six procedures. The learned procedures themselves are frozen throughout this test.

| Procedure replicate | Family | Arm | Mean clean MSE | Recovered segments | Mean seconds |
|---|---|---|---:|---:|---:|
| 12101 | conflicting_functions | adam_001 | 0.269945 | 4/228 | 2.672 |
| 12101 | conflicting_functions | adam_003 | 0.227398 | 65/228 | 2.670 |
| 12101 | conflicting_functions | adam_01 | 0.190068 | 110/228 | 2.665 |
| 12101 | conflicting_functions | frozen_bootstrap | 0.183594 | 107/228 | 6.041 |
| 12101 | conflicting_functions | self_applied | 0.184519 | 115/228 | 6.037 |
| 12101 | conflicting_functions | fixed_meta_adam | 0.185248 | 102/228 | 6.038 |
| 12101 | input_shift | adam_001 | 0.035220 | 220/228 | 2.659 |
| 12101 | input_shift | adam_003 | 0.017567 | 227/228 | 2.655 |
| 12101 | input_shift | adam_01 | 0.011239 | 227/228 | 2.670 |
| 12101 | input_shift | frozen_bootstrap | 0.010468 | 227/228 | 6.045 |
| 12101 | input_shift | self_applied | 0.010231 | 227/228 | 6.062 |
| 12101 | input_shift | fixed_meta_adam | 0.010424 | 227/228 | 6.074 |
| 12102 | conflicting_functions | adam_001 | 0.269945 | 4/228 | 2.672 |
| 12102 | conflicting_functions | adam_003 | 0.227398 | 65/228 | 2.670 |
| 12102 | conflicting_functions | adam_01 | 0.190068 | 110/228 | 2.665 |
| 12102 | conflicting_functions | frozen_bootstrap | 0.182700 | 121/228 | 6.056 |
| 12102 | conflicting_functions | self_applied | 0.182152 | 115/228 | 6.055 |
| 12102 | conflicting_functions | fixed_meta_adam | 0.182552 | 115/228 | 6.051 |
| 12102 | input_shift | adam_001 | 0.035220 | 220/228 | 2.659 |
| 12102 | input_shift | adam_003 | 0.017567 | 227/228 | 2.655 |
| 12102 | input_shift | adam_01 | 0.011239 | 227/228 | 2.670 |
| 12102 | input_shift | frozen_bootstrap | 0.010329 | 227/228 | 6.074 |
| 12102 | input_shift | self_applied | 0.010288 | 227/228 | 6.083 |
| 12102 | input_shift | fixed_meta_adam | 0.010354 | 227/228 | 6.074 |

The self-applied procedure has lower family-average MSE than the equal-extra-meta controller in all four replicate/family aggregates, but the margins are small and aggregate direction hides stream-level reversals. The independent bootstrap count remains two. The same four worlds appear under both procedure replicates; the repeated fixed arms are reused outputs, not new replications.

Replicate 12101, conflicting_functions: self-application wins 1/4 streams against freezing, 2/4 against conventional meta-updates, and 4/4 against fixed Adam 0.01.
Replicate 12101, input_shift: self-application wins 2/4 streams against freezing, 3/4 against conventional meta-updates, and 4/4 against fixed Adam 0.01.
Replicate 12102, conflicting_functions: self-application wins 2/4 streams against freezing, 2/4 against conventional meta-updates, and 4/4 against fixed Adam 0.01.
Replicate 12102, input_shift: self-application wins 3/4 streams against freezing, 3/4 against conventional meta-updates, and 4/4 against fixed Adam 0.01.

For conflicting functions, the self-applied procedures recover in 115/228 segments each. The second frozen bootstrap recovers in 121/228. Lower average error therefore does not imply faster or more reliable recovery in each setting. In the input-shift family the stronger fixed and learned procedures recover in 227/228 segments; the assay separates the families much more sharply than it separates those procedures.

## Recovery criterion and its limits

Recovery is completion of three consecutive batches whose clean prequential MSE divided by mean squared clean target plus 0.01 is at most 0.2. The clean targets, boundaries and recurrence labels are used only by the recorder. Segments that do not reach the threshold remain right-censored at their actual end. No failed segment is excluded from the denominator, and no uncensored mean recovery time is asserted. The criterion is a declared operational choice, not universal competence. Returning-segment recovery may include relearning; it is not a certificate of retained knowledge.

This supplies a first censored recovery assay for the candidate bottleneck. Averaging the fraction of each segment consumed before recovery or its end, the two self-applied procedures use 82.63% and 81.99% under conflicting functions, versus 11.06% and 10.97% under input shift. An unrecovered segment contributes its full observed lifetime; this is a capped burden statistic, not an uncensored recovery-time estimate. Whole-stream error can improve while most useful time remains spent below the competence threshold. It does not prove one universal quantity bounds every AI system or isolate an information-theoretic minimum.

## Work, storage and scope

All arms consume the same 65,536 observations and use 2,048 real student updates, 65,504 replayed examples and 196,576 student forward examples per run. Learned procedures add 12,288 per-tensor controller calls. They cost about 6.06 seconds versus 2.67 seconds for fixed Adam on this local runtime, before inherited procedure training/selection work. Their modest MSE gains therefore do not establish total-resource efficiency. The three fixed rates are all reported; no final-data optimizer selection is claimed as an agent capability.

Stored tensor counts include student parameters, optimizer moments, replay, context, controller and RNG state. Python metadata, full process RAM, complete FLOPs and energy are not measured. Existing E9 bootstrap/self-trial and E11 conventional-meta costs remain inherited costs; see their reports. Of 96 reported combinations, 24 fixed-optimizer cases are reused across bootstrap replicates, leaving 72 newly executed stream runs. Standalone and execution preflights are additional dummy work.

No final task diverged. Passing this transfer assay does not mean the procedure learns continuously here: it is frozen after E9/E11 while the student retains state. E18 is separately frozen to test actual online procedure changes with persistent students. There is still no closed-loop planner, semantic memory compression, broad reasoning or novelty result in E17.
