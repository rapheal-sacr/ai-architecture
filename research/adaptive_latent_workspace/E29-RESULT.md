# E29 — variable-depth continual graph training

Source frozen at `c8a0884642b4e168e8c65280f0679d4085592169`. All nine training cases and 27 stage checkpoints completed. Three fresh seeds compare fixed16, fixed24 and variable16–32 recurrence with identical current observations, replay, initialization, model and optimizer. The full goal remains active.

The variable schedule pairs r with 48-r and fixes first/last update at24, yielding exactly the fixed24 graph-weighted message count at every stage, including the first update without replay. This matches recurrence message work and corresponding backpropagation, not measured FLOPs or wall time. All arms retain weights, optimizer and a 128-graph reservoir across Dijkstra → Prim → Dijkstra. Each stage has 512 updates with eight new graphs and up to eight replay graphs per update. Final parent targets provide supervision; there are no intermediate hints, target-derived iteration lengths or test-selected checkpoints.

This is a bounded use of randomized-depth training described in the supplied TTC-LR work. It is not a Raven reproduction, a novel training procedure or recursive self-improvement. Cells average three seed fractions from 24 graphs per seed, so related cells must not be treated as independent replications.

Variable depth gives a scoped repair of inference-depth robustness, not the full reasoning claim. At 32 inference steps, final N16 Dijkstra chain correctness is 95.8% versus 63.9% for equal-work fixed24; the paired gain occurs in two seeds, with a 4.2-point loss in the first. Under public-N inference, final random-N16 Dijkstra correctness is 50.0% versus 29.2% for fixed24, but the cheaper fixed16 reaches 54.2%. Every final N64 task/family/policy cell has zero functionally correct graphs. No seed/arm meets the declared acquisition criteria.

## Primary public-N learning and retention

| Arm | Dijkstra after first stage | Dijkstra after Prim | Prim after Prim | Dijkstra after return | Prim after return |
|---|---:|---:|---:|---:|---:|
| fixed16_replay | 18.1% | 30.6% | 13.9% | 54.2% | 11.1% |
| fixed24_replay | 5.6% | 23.6% | 4.2% | 29.2% | 4.2% |
| variable_replay | 23.6% | 36.1% | 11.1% | 50.0% | 16.7% |

## Final depth robustness and size transfer

| Arm | Inference policy | Dijkstra N16 chain | Dijkstra N64 random | Dijkstra N64 chain | Prim N64 random | Prim N64 chain |
|---|---|---:|---:|---:|---:|---:|
| fixed16_replay | public_N | 93.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed16_replay | fixed24 | 23.6% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed16_replay | public_2N | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed24_replay | public_N | 9.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed24_replay | fixed24 | 91.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed24_replay | public_2N | 63.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| variable_replay | public_N | 56.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| variable_replay | fixed24 | 93.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| variable_replay | public_2N | 95.8% | 0.0% | 0.0% | 0.0% | 0.0% |

All three policies were declared in advance. Fixed24 is an ordinary constant inference budget; public N and public 2N depend only on graph size. None chooses a depth from the correct answer. All policy executions, including overlapping depths on particular sizes, are charged. Stable answers at longer depth and learning a correct larger-graph algorithm are distinct outcomes.

## Charged work

| Arm | Training seconds | All evaluation query seconds | Graph presentations | Forward message candidates | Persistent tensor bytes | Peak CUDA training bytes |
|---|---:|---:|---:|---:|---:|---:|
| fixed16_replay | 89.79 | 32.89 | 24568 | 100630528 | 336800 | 88005120 |
| fixed24_replay | 125.29 | 32.70 | 24568 | 150945792 | 336800 | 96705024 |
| variable_replay | 121.97 | 32.72 | 24568 | 150945792 | 336800 | 105404928 |

Tensor bytes include model, optimizer, valid replay tensors and torch/CUDA RNG. They exclude Python objects, the externally generated depth-schedule list, runtime libraries and common source datasets; full process memory is not measured. CUDA peak allocation is measured separately. Graph generation and reference labels are common source costs recorded in raw identities; evaluation validation, checkpoint I/O and source setup are separate from these training/query timings. Message candidates are not measured FLOPs.

## Registered criteria

Acquisition requires >=80% functional whole-graph correctness on random N16 inputs of the newly taught task. Retention permits <=5 percentage points lost only when acquisition occurred. N64 transfer requires >=80% functional correctness. Failure to acquire cannot be recast as a retention success.

- fixed16_replay: Dijkstra acquisition 0/3; Prim acquisition 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.
- fixed24_replay: Dijkstra acquisition 0/3; Prim acquisition 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.
- variable_replay: Dijkstra acquisition 0/3; Prim acquisition 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.

## Audit

All 27 stage checkpoint hashes, optimizer steps, finite weights, declared depth schedules and graph/message counters were checked. Replay contents and RNG were independently reconstructed from the original stream at each stage. Initialization and current data hashes match across arms; fixed24 and variable have exactly equal message counts in all nine paired stages. All nine restored final models reproduce all final predictions for all three policies exactly in the scored runner. Evaluation preserves model and torch/CUDA RNG state. The separate preflight additionally reproduces E27 baseline losses and model/optimizer/memory/RNG exactly.
