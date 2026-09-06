# E1 result: context reuse is not solved

Four frozen seeds, five arms, 65,536 observations per arm/seed. Every model actually learns nonlinear teacher functions. Context IDs reach only the labeled oracle. All scored predictions precede observation of their targets.

| Arm | Stream MSE | First 32 returning batches MSE | Seconds | Parameters | Stored tensor bytes |
|---|---:|---:|---:|---:|---:|
| online | 0.03027 | 0.14214 | 1.911 | 4996 | 60120 |
| replay | 0.09088 | 0.16026 | 2.562 | 4996 | 158424 |
| context_replay | 0.03822 | 0.08551 | 2.792 | 7300 | 480984 |
| module_bank | 0.02766 | 0.11606 | 2.328 | 11241 | 134946 |
| oracle_modules | 0.01615 | 0.01037 | 1.911 | 19984 | 239904 |

These are equal-experience, one-update-per-batch comparisons, not equal-FLOP or equal-parameter comparisons. Replay uses additional stored samples; module banks evaluate every persistent model after observing each batch. Tensor bytes include model weights, optimizer states and replay storage, but not Python object overhead. CPU timings are local measurements, not frontier hardware estimates.

The module bank does not approach the oracle upper bound and has worse return-task loss than context-conditioned replay. It creates only two or three modules for four contexts. The first architecture hypothesis is therefore **not established**. A lower overall error than a particular baseline is insufficient to claim efficient persistent reuse.

A source-level failure path is that the active module is updated during the novelty-confirmation window. It can partially fit a new regime before the detector decides to allocate capacity, overwriting old behavior. E2 will test temporary adaptation isolated from persistent modules and direct reuse checks against established models. This is a new intervention and protocol; E1 stays unchanged.

No long-horizon closed-loop task, memory compression, plasticity-renewal result or learned routing-cost reduction has been measured by E1. The overall research goal remains active.
