# E21: faster evidence repairs much of the conflict failure but hurts input shift

Every arm uses actual preceding observations for prediction; none receives hidden regimes. All have the same 738-parameter student and 512-example replay. The experiment crosses context decay 0.95 versus 0.5 with training/replay assignment before versus after incorporating the current observed outputs. Predictions always precede those outputs, even in the posterior-assignment arm. Four fresh worlds per family are paired across the four cells.

| Family | Context decay | Assignment | Mean clean MSE | Recovered segments | Mean seconds |
|---|---:|---|---:|---:|---:|
| conflicting_functions | 0.95 | prior_assignment | 0.165549 | 195/408 | 5.325 |
| conflicting_functions | 0.95 | posterior_assignment | 0.161353 | 232/408 | 5.538 |
| conflicting_functions | 0.5 | prior_assignment | 0.065703 | 382/408 | 5.325 |
| conflicting_functions | 0.5 | posterior_assignment | 0.060402 | 385/408 | 5.536 |
| input_shift | 0.95 | prior_assignment | 0.007944 | 404/408 | 5.312 |
| input_shift | 0.95 | posterior_assignment | 0.008077 | 404/408 | 5.522 |
| input_shift | 0.5 | prior_assignment | 0.009302 | 404/408 | 5.311 |
| input_shift | 0.5 | posterior_assignment | 0.009678 | 404/408 | 5.546 |

## Non-oracle gain and its counterexample

For conflicting_functions, faster posterior assignment changes mean MSE by -63.5% relative to slow prior assignment, winning 4/4 fresh worlds. Its runtime ratio is 1.040 with identical student-example counts and stored learner tensor bytes.
For input_shift, faster posterior assignment changes mean MSE by +21.8% relative to slow prior assignment, winning 0/4 fresh worlds. Its runtime ratio is 1.044 with identical student-example counts and stored learner tensor bytes.

The factor comparison separates a large recency effect from a smaller assignment-timing effect in the conflicting-function family. A stale context estimate is therefore a major avoidable limitation of the previous implementation, not evidence that more shared weights are intrinsically necessary. Posterior assignment uses current outputs only after their prediction has been scored; its gain is evaluated on future prequential behavior, not lower training loss.

The input-shift counterexample prevents installing faster forgetting as a universal fix. Rapidly changing the contextual representation can add noise or unnecessary variation when the conditional function remains stable. Both task types require a criterion for allocating evidence across timescales; a fixed universally fast or slow context does not settle that problem.

## Costs and limits

Every arm receives 131,072 observations, runs 4,096 real updates and 393,184 student forward examples, and retains 71,328 counted learner tensor bytes. No controller or meta-training is used. The posterior adapter incurs a redundant cross-moment calculation, with its actual elapsed time included. Raw data are shared across paired conditions; repeated learner exposures are not independent data acquisition. Full process RAM, complete FLOPs and energy are unmeasured.

The preflight verifies exact prior-path equality with E17, equal persistent context trajectories between assignment modes at a fixed decay, correct posterior features in replay, and equal counters. Every prediction remains causal. The current-label contribution to a training batch summary could still create a training shortcut; held-out prequential behavior is the relevant test. Recovery uses the frozen E17 criterion and retains all censored segments.

This is a concrete inexpensive repair for a scoped inference failure, with an explicit negative transfer result. It is not learned timescale allocation, semantic compression, general reasoning, novel inference theory, reliable recursive improvement or long-horizon closed-loop success. The broader architecture still needs those mechanisms and independent falsification.
