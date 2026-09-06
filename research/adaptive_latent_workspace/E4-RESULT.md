# E4: external transfer fails against context replay

The unchanged isolated-adaptation learner loses to context-conditioned replay on both seeds of the pinned loss-of-plasticity repository's slowly changing regression generator. Both arms receive exactly the same million-example stream per seed. Only learner input/output dimensions change to match the external task.

| Arm | Whole-stream MSE | First ten blocks | Last ten blocks | Seconds | Persistent modules | Capacity hits | Tensor bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| online | 0.14702 | 0.24211 | 0.11528 | 28.18 | 1 | 0 | 66936 |
| context_replay | 0.08128 | 0.26364 | 0.05095 | 41.13 | 1 | 0 | 427128 |
| isolated_adaptation | 0.11336 | 0.24321 | 0.12122 | 63.53 | 8 | 27918 | 601668 |

The source teacher is a fixed LTU network. Fifteen observed input bits change slowly while five bits vary rapidly. It is not the hidden switching-function problem used for E1–E3. Shared prediction can be useful across changes because the conditional target function itself stays fixed.

**Rejected scope:** the E2 rule is not established as an efficient general continual-learning architecture. High prediction error is not enough to infer that a new persistent context is needed. The observed slot exhaustion is consistent with inappropriate partitioning or limited representational capacity; this experiment does not distinguish those causes.

**Interpretation limits:** this is an external-generator transfer screen, not a reproduction of the published loss-of-plasticity findings. We use batch size 32, two seeds, one million scored observations, 64-wide tanh learners and Adam, whereas the published protocols differ. No CBP arm was run, so no comparison with feature renewal is licensed. The generator creates an extra block; only the first million observations are scored. The source's locally generated pickle is never passed to a learner; only X and already-observed Y are. Stream hashes verify identical data across arms.

Blocks contain at least 10,000 samples (usually 10,016 because batches straddle the boundary). Exact observation counts are stored. Timings are local CPU measurements, with no simultaneous long experiment; small reporting/setup tasks may overlap.

The next design must test whether new observations require a different conditional rule or can be incorporated into shared learning, while charging the test and consolidation costs. Simply enlarging the module cap does not resolve the binding storage/routing cost exposed here.
