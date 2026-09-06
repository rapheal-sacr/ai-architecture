# E23: regularized relationship evidence gives a modest scoped repair

All 80 frozen cases completed without divergence. Four fresh worlds per family are paired across raw cross-moments versus regularized linear coefficients and context decays 0.95 versus 0.5. Predictions use preceding evidence; both descriptors annotate training and replay after the output arrives. The same 738-parameter student, 512-example reservoir, observations and update rule persist throughout. No gate, hidden regimes, extra model width or offline meta-training is supplied.

| Family | Decay | Descriptor | Mean clean MSE | Recovered / all segments | Mean seconds |
|---|---:|---|---:|---:|---:|
| conflicting_functions | 0.95 | raw_cross_moment | 0.176336 | 201/403 | 5.732 |
| conflicting_functions | 0.95 | ridge_coefficients | 0.176133 | 196/403 | 6.247 |
| conflicting_functions | 0.5 | raw_cross_moment | 0.065302 | 372/403 | 5.726 |
| conflicting_functions | 0.5 | ridge_coefficients | 0.063257 | 382/403 | 6.227 |
| input_shift | 0.95 | raw_cross_moment | 0.009063 | 403/403 | 5.727 |
| input_shift | 0.95 | ridge_coefficients | 0.008514 | 403/403 | 6.261 |
| input_shift | 0.5 | raw_cross_moment | 0.010099 | 403/403 | 5.724 |
| input_shift | 0.5 | ridge_coefficients | 0.008831 | 403/403 | 6.257 |
| coupled_shift | 0.95 | raw_cross_moment | 0.060052 | 376/401 | 5.705 |
| coupled_shift | 0.95 | ridge_coefficients | 0.056834 | 376/401 | 6.222 |
| coupled_shift | 0.5 | raw_cross_moment | 0.053799 | 387/401 | 5.703 |
| coupled_shift | 0.5 | ridge_coefficients | 0.048769 | 391/401 | 6.207 |
| independent_shift | 0.95 | raw_cross_moment | 0.205380 | 197/793 | 5.718 |
| independent_shift | 0.95 | ridge_coefficients | 0.188515 | 261/793 | 6.256 |
| independent_shift | 0.5 | raw_cross_moment | 0.104337 | 536/793 | 5.743 |
| independent_shift | 0.5 | ridge_coefficients | 0.090570 | 579/793 | 6.237 |
| alternating_dependency | 0.95 | raw_cross_moment | 0.115981 | 308/618 | 5.722 |
| alternating_dependency | 0.95 | ridge_coefficients | 0.106770 | 348/618 | 6.229 |
| alternating_dependency | 0.5 | raw_cross_moment | 0.076893 | 434/618 | 5.714 |
| alternating_dependency | 0.5 | ridge_coefficients | 0.069308 | 468/618 | 6.222 |

## Paired improvement and counterexamples

- conflicting_functions, decay 0.95: -0.1% mean error; 3/4 wins; 1.090 times runtime. Counterexample seeds: [23103].
- conflicting_functions, decay 0.5: -3.1% mean error; 4/4 wins; 1.087 times runtime. Counterexample seeds: none in these four.
- input_shift, decay 0.95: -6.1% mean error; 4/4 wins; 1.093 times runtime. Counterexample seeds: none in these four.
- input_shift, decay 0.5: -12.6% mean error; 3/4 wins; 1.093 times runtime. Counterexample seeds: [23102].
- coupled_shift, decay 0.95: -5.4% mean error; 4/4 wins; 1.091 times runtime. Counterexample seeds: none in these four.
- coupled_shift, decay 0.5: -9.3% mean error; 4/4 wins; 1.088 times runtime. Counterexample seeds: none in these four.
- independent_shift, decay 0.95: -8.2% mean error; 4/4 wins; 1.094 times runtime. Counterexample seeds: none in these four.
- independent_shift, decay 0.5: -13.2% mean error; 4/4 wins; 1.086 times runtime. Counterexample seeds: none in these four.
- alternating_dependency, decay 0.95: -7.9% mean error; 4/4 wins; 1.089 times runtime. Counterexample seeds: none in these four.
- alternating_dependency, decay 0.5: -9.9% mean error; 4/4 wins; 1.089 times runtime. Counterexample seeds: none in these four.

The representation change improves every paired independent and alternating world at both decays, with a modest gain on top of the stronger fast/post-outcome rule. Input-shift and conflicting-function counterexamples remain. The result supports regularized coefficients as a scoped evidence descriptor; it does not establish invariant task identity or an optimal timescale. Recovery can worsen even where whole-stream error improves, as the individual records show.

In noiseless full-rank linear data, unregularized coefficients remove the input-distribution dependence of cross-moments. The preflight confirms that algebra. Scored teachers are nonlinear, so these coefficients remain distribution-weighted projections, and ridge introduces bias. Descriptor scale and orientation also change. This experiment cannot attribute the entire gain solely to covariance normalization or claim general invariance.

## Costs and audit

Each learner consumes 131,072 observations, 4,096 student updates and 393,184 student forward examples. Ridge adds 4,096 nine-by-nine solves and Gram updates, plus 396 persistent tensor bytes: 71,724 versus 71,328. Its mean local runtime is 1.090 times raw/post-outcome execution. Both implementations incur the inherited redundant context calculation. There is no gate backward pass or bootstrap.

Every final checkpoint matches its frozen source identity and restored counters, has 4,096 persistent steps and a 512-example reservoir. All 40 paired raw reservoir input/target/RNG checks match exactly; contextual annotations may differ. All 40 final coefficient descriptors satisfy their regularized normal equations within the recorded numerical tolerance. Preflight tests exact raw E21 controls, causal initial predictions, post-outcome storage, full trajectory restoration and the linear algebra invariance claim.

All censored recovery segments, including those shorter than the three-batch criterion, are preserved. The twenty shared worlds are the independent seed/family units; repeated learner exposures are not additional acquired data. These local loop times exclude data generation, preflight and checkpoint serialization, and are not complete deployment latency. Full process RAM, peak solver workspace, complete FLOPs, energy, rare semantic recall and autonomous planning horizon remain unmeasured.

This is established adaptive-regression machinery, credited to prior work including ALPaCA, not a novelty claim. It is a stronger fixed baseline for this subproblem, not autonomous recursive improvement. No result here repairs the world-model search failure or establishes safe compression. The next experiment, E24, returns to the outstanding rare-memory audit rather than continuing to tune this family.
