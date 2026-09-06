# E22: learned evidence allocation does not repair the tradeoff

All 100 frozen cases completed without divergence: four fresh worlds in each of five families, paired across five arms, with 4,096 updates and 131,072 observations per run. The 65-parameter gate actually learns online from each recorded prediction after its output arrives. Student weights, gate, optimizer moments, context and bounded replay persist across hidden changes. There are no oracle boundaries or family inputs.

| Family | Arm | Mean clean MSE | Recovered / all segments | Mean seconds |
|---|---|---:|---:|---:|
| conflicting_functions | slow_prior | 0.214882 | 228/420 | 5.291 |
| conflicting_functions | fast_prior | 0.080296 | 411/420 | 5.274 |
| conflicting_functions | fast_posterior | 0.071199 | 416/420 | 5.506 |
| conflicting_functions | fixed_mixture | 0.111141 | 403/420 | 5.892 |
| conflicting_functions | learned_gate | 0.085562 | 406/420 | 10.261 |
| input_shift | slow_prior | 0.011266 | 420/420 | 5.277 |
| input_shift | fast_prior | 0.012683 | 420/420 | 5.278 |
| input_shift | fast_posterior | 0.012720 | 420/420 | 5.499 |
| input_shift | fixed_mixture | 0.012210 | 420/420 | 5.877 |
| input_shift | learned_gate | 0.011753 | 420/420 | 10.245 |
| coupled_shift | slow_prior | 0.060269 | 397/407 | 5.255 |
| coupled_shift | fast_prior | 0.058526 | 402/407 | 5.264 |
| coupled_shift | fast_posterior | 0.061384 | 404/407 | 5.532 |
| coupled_shift | fixed_mixture | 0.058730 | 401/407 | 5.885 |
| coupled_shift | learned_gate | 0.056665 | 400/407 | 10.334 |
| independent_shift | slow_prior | 0.240644 | 237/811 | 5.274 |
| independent_shift | fast_prior | 0.130799 | 545/811 | 5.283 |
| independent_shift | fast_posterior | 0.115848 | 606/811 | 5.519 |
| independent_shift | fixed_mixture | 0.171394 | 430/811 | 5.890 |
| independent_shift | learned_gate | 0.146832 | 496/811 | 10.291 |
| alternating_dependency | slow_prior | 0.142987 | 333/622 | 5.294 |
| alternating_dependency | fast_prior | 0.100717 | 418/622 | 5.301 |
| alternating_dependency | fast_posterior | 0.088918 | 457/622 | 5.533 |
| alternating_dependency | fixed_mixture | 0.112881 | 392/622 | 5.920 |
| alternating_dependency | learned_gate | 0.123682 | 360/622 | 10.295 |

## Direct falsification

- conflicting_functions: slow_prior: -60.2% error, 4/4 wins; fast_prior: +6.6% error, 0/4 wins; fast_posterior: +20.2% error, 0/4 wins; fixed_mixture: -23.0% error, 4/4 wins.
- input_shift: slow_prior: +4.3% error, 0/4 wins; fast_prior: -7.3% error, 4/4 wins; fast_posterior: -7.6% error, 4/4 wins; fixed_mixture: -3.7% error, 4/4 wins.
- coupled_shift: slow_prior: -6.0% error, 3/4 wins; fast_prior: -3.2% error, 3/4 wins; fast_posterior: -7.7% error, 3/4 wins; fixed_mixture: -3.5% error, 4/4 wins.
- independent_shift: slow_prior: -39.0% error, 4/4 wins; fast_prior: +12.3% error, 0/4 wins; fast_posterior: +26.7% error, 0/4 wins; fixed_mixture: -14.3% error, 4/4 wins.
- alternating_dependency: slow_prior: -13.5% error, 4/4 wins; fast_prior: +22.8% error, 0/4 wins; fast_posterior: +39.1% error, 0/4 wins; fixed_mixture: +9.6% error, 0/4 wins.

The strong fast/prior control removes the post-outcome assignment difference and still defeats the learned gate on every conflicting, independent and alternating world. The weak slow/prior reference alone would have hidden that failure. The gate also loses to slow/prior on every input-shift world. Coupled shifts supply a small scoped gain over fast/prior in some worlds, without a general improvement.

## Persistent adaptation and its boundary

| Family | Gate mean, first quarter | Second | Third | Fourth |
|---|---:|---:|---:|---:|
| conflicting_functions | 0.639 | 0.949 | 0.988 | 0.984 |
| input_shift | 0.230 | 0.046 | 0.020 | 0.027 |
| coupled_shift | 0.495 | 0.719 | 0.765 | 0.728 |
| independent_shift | 0.480 | 0.718 | 0.889 | 0.954 |
| alternating_dependency | 0.234 | 0.163 | 0.295 | 0.417 |

Fast evidence receives increasing weight on conflicting functions and slow evidence on input shift. That movement is compatible with learning a persistent timescale preference; it does not establish rapid detection of each relationship change. The alternating family changes its dependency regime every 512 batches within the same student lifetime. Its gate trajectories and all eight phase errors are retained in the summary and plot. Inference about slow adaptation or saturation is provisional: these results do not isolate a unique cause from representation, gate features, local objective and student/gate coadaptation.

## Work, audit and unmeasured quantities

- slow_prior: mean 5.278 seconds; 71,328 counted learner tensor bytes.
- fast_prior: mean 5.280 seconds; 71,328 counted learner tensor bytes.
- fast_posterior: mean 5.518 seconds; 71,328 counted learner tensor bytes.
- fixed_mixture: mean 5.893 seconds; 71,472 counted learner tensor bytes.
- learned_gate: mean 10.285 seconds; 72,268 counted learner tensor bytes.

Learned-gate execution costs 1.95 times fast/prior time. All arms use 393,184 student forward examples and 4,096 student updates per run. The gate adds 4,096 gate forwards and 4,096 extra backward passes through the student into 65 gate parameters, plus gate optimizer state and two evidence summaries. Equal student forward counts therefore do not mean equal training work. There is no offline gate bootstrap. Data generation, preflight and checkpoint serialization are outside timed learner loops and recorded separately where available; full end-to-end deployment latency was not measured.

All 100 final states have the expected persistent step and bounded replay, finite weights, matching source identity and exactly reproduced restored cost counters. All 80 control comparisons have identical raw reservoir inputs, targets and RNG state; contextual annotations can differ. The preflight checks exact fixed controls, initial half-mixture equality, causal gate derivatives by finite differences, actual gate parameter changes, exact resume and independent generator clocks. Checkpoint hashes are recorded in the audit.

Independent/alternating evaluation segments use the union of input and function changes, so some last fewer than the three batches required by the recovery criterion. Their counts and all censored segments remain in the data. Recovery totals across different families are not directly comparable. Four world seeds remain a small sample; no significance or universal bound is claimed. Full peak autograd memory, process RAM, FLOPs, energy, semantic recall and autonomous task horizon were not measured.

The proposed gate is rejected as a sufficient repair of the evidence-allocation problem. Its implementation is real continual procedure adaptation, but it is neither recursive self-application nor demonstrated general improvement. No claimed success warrants the optional keep/revert gate continuation yet. The broader architecture remains unvalidated.
