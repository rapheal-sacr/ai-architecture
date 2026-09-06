# E3: stress results and boundaries

All five frozen stress cases completed on two seeds each. This is an exploratory screen, not a statistically powered generality claim. Prediction occurs before each target is observed. Settings are unchanged from E2.

| Case | Arm | MSE | Final eighth MSE | Seconds | Modules | Capacity hits | Peak tensor bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| noise | online | 0.132841 | 0.145751 | 1.92 | 1.0 | 0 | 60120 |
| noise | context_replay | 0.144534 | 0.135897 | 2.75 | 1.0 | 0 | 480984 |
| noise | isolated_adaptation | 0.122373 | 0.112371 | 2.87 | 4.0 | 0 | 239904 |
| short | online | 0.224081 | 0.354422 | 0.24 | 1.0 | 0 | 60120 |
| short | context_replay | 0.230791 | 0.318620 | 0.36 | 1.0 | 0 | 480984 |
| short | isolated_adaptation | 0.188306 | 0.270557 | 0.31 | 2.5 | 0 | 179928 |
| capacity | online | 0.089859 | 0.103627 | 5.79 | 1.0 | 0 | 60120 |
| capacity | context_replay | 0.068687 | 0.057699 | 8.37 | 1.0 | 0 | 480984 |
| capacity | isolated_adaptation | 0.045719 | 0.061418 | 12.15 | 8.0 | 2938 | 539784 |
| long | online | 0.058445 | 0.062727 | 30.92 | 1.0 | 0 | 60120 |
| long | context_replay | 0.018942 | 0.015397 | 43.80 | 1.0 | 0 | 480984 |
| long | isolated_adaptation | 0.006211 | 0.004914 | 49.08 | 4.0 | 0 | 239904 |
| gradual | online | 0.006333 | 0.005226 | 3.96 | 1.0 | 0 | 60120 |
| gradual | context_replay | 0.012911 | 0.009070 | 5.71 | 1.0 | 0 | 480984 |
| gradual | isolated_adaptation | 0.006333 | 0.005226 | 4.87 | 1.0 | 0 | 59976 |

**Supported in this assay:** the four-context million-observation stream retains a substantial prediction advantage. Noise and short context lifetimes do not reverse the whole-stream error advantage on these two seeds. This establishes neither adversarial-noise robustness nor rapid reliable identification of all short contexts.

**Failed general cost claim:** under gradual interpolation the candidate creates one module and makes exactly the same scored predictions as online learning, with approximately 23% extra runtime. The novelty gate treats the changing function as one adaptable context. It supplies no demonstrable retained history of intermediate functions.

**Unsolved capacity:** sixteen contexts exhaust eight slots. Thousands of subsequent updates satisfy the commit condition but cannot acquire a persistent slot. Lower whole-stream prediction error comes with approximately 45% more runtime than context replay and twice online runtime. In the final eighth, the candidate is worse than context replay (0.06142 versus 0.05770 MSE), so its average advantage does not establish a sustainable long-run advantage. Capacity hits count updates, not distinct rejected tasks. An unbounded archive would evade this test rather than solve its resource constraint.

**Cost caveat:** forward-example counts include all expert scoring, but are not FLOPs when network widths differ. Stored tensor bytes omit Python containers, gradients and transient forward/backward activations. Peak tensor bytes include the temporary model and optimizer, not peak process RAM. Timing includes the stream generator and learner work, excludes held-out diagnostics, and is an indicative local CPU measurement.

**Still unmeasured:** episodic compression with rare-fact recovery, amortized module consolidation, shared learning between related contexts, general reasoning and closed-loop long-horizon performance. A million supervised observations is not a million-step successful autonomous task.
