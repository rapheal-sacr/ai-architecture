# E20: context information versus shared model capacity

A two-by-two diagnostic crosses inferred versus explicitly privileged context with hidden width 16 versus 64. All cells use the same persistent learner, replay budget, fixed Adam rate 0.01 and 4,096-batch streams. Four fresh worlds per family are paired across cells. The oracle receives the true hidden regime before prediction, encoded one-hot within the same 18-component context interface. The real inferred path receives no regime labels.

| Seed | Family | Inferred, width 16 | Inferred, width 64 | Oracle, width 16 | Oracle, width 64 |
|---|---|---:|---:|---:|---:|
| 20101 | conflicting_functions | 0.178542 | 0.160051 | 0.027651 | 0.016752 |
| 20101 | input_shift | 0.004653 | 0.003990 | 0.004766 | 0.003901 |
| 20102 | conflicting_functions | 0.143105 | 0.131879 | 0.030906 | 0.019404 |
| 20102 | input_shift | 0.008546 | 0.006120 | 0.007140 | 0.005374 |
| 20103 | conflicting_functions | 0.220686 | 0.199126 | 0.030749 | 0.020106 |
| 20103 | input_shift | 0.007820 | 0.006534 | 0.006177 | 0.005101 |
| 20104 | conflicting_functions | 0.101069 | 0.091280 | 0.022645 | 0.014884 |
| 20104 | input_shift | 0.007893 | 0.005629 | 0.007241 | 0.004950 |

## Recovery and measured work

| Family | Context | Width | Mean MSE | Recovered segments | Parameters | Stored tensor bytes | Mean seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| conflicting_functions | inferred_ema | 16 | 0.160851 | 234/407 | 738 | 71328 | 5.383 |
| conflicting_functions | inferred_ema | 64 | 0.145584 | 260/407 | 6018 | 134688 | 5.746 |
| conflicting_functions | oracle_onehot_diagnostic | 16 | 0.027988 | 396/407 | 738 | 71328 | 5.455 |
| conflicting_functions | oracle_onehot_diagnostic | 64 | 0.017787 | 401/407 | 6018 | 134688 | 5.829 |
| input_shift | inferred_ema | 16 | 0.007228 | 406/407 | 738 | 71328 | 5.375 |
| input_shift | inferred_ema | 64 | 0.005568 | 407/407 | 6018 | 134688 | 5.744 |
| input_shift | oracle_onehot_diagnostic | 16 | 0.006331 | 407/407 | 738 | 71328 | 5.433 |
| input_shift | oracle_onehot_diagnostic | 64 | 0.004831 | 407/407 | 6018 | 134688 | 5.811 |

## Interpretation boundary

For conflicting_functions, supplying oracle context at width 16 changes mean MSE by -82.6%; widening under inferred context changes it by -9.5%.
For input_shift, supplying oracle context at width 16 changes mean MSE by -12.4%; widening under inferred context changes it by -23.0%.

The oracle intervention changes both information and its encoding. A large improvement locates a bottleneck in the current inference/assignment/interface package; it does not measure an information-theoretic identification lower bound. In particular, E17 stores the preceding inferred context with replay examples, whereas the oracle stores the correct context. Both action-time identification and assigning observations to retained conditional models can contribute. The test does not distinguish those two effects.

The wider model uses more weights and arithmetic. Equal student-example counts are not equal FLOPs. Each cell uses 393,184 student forward examples and 4,096 gradient batches; the cost table charges stored weights, optimizer moments, replay, context and RNG tensors. Dataset-generation time is separately recorded. Timings overlap E19 and are descriptive, with no full process RAM or energy measurement.

Recovery uses the unchanged E17 evaluation-only criterion and retains every censored full segment. Four worlds are independent seed choices within each family, while the paired conditions and within-world segments are dependent. No significance claim or optimal learning-rate tuning is made.

The oracle is never supplied to a deployed candidate and cannot count as autonomous context inference. This diagnostic does not establish memory compression, general reasoning, novel architecture design, reliable recursive improvement or efficient long-horizon control. It constrains the next architecture: a stronger inference/memory-assignment mechanism must earn its gain from actual observed evidence and repay its cost.
