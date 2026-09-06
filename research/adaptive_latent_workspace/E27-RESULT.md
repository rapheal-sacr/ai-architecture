# E27 — final-target recurrent graph learning and retention

Source frozen at `505da5d09a035915efe081da0d42f63b632f95a6`. All twelve cases and 36 stage checkpoints completed. The goal remains active. This is a small supervised graph pilot, not an integrated autonomous agent.

The same 15,177-parameter graph processor consumes Dijkstra → Prim → Dijkstra tasks, 512 updates per stage, eight new N16 graphs per update. Public-N recurrence uses 16 training steps versus two. Replay adds eight earlier graphs per update after the first, held in a 128-graph reservoir. All arms share current observations and initialization; replay trajectories are identical across recurrence depths. Explicit task IDs request algorithms; no gold hints or target-derived stopping lengths enter training or prediction.

Each reported cell averages three seed fractions from 24 fresh graphs per seed. These are small, correlated evaluations; no significance or broad generalization claim follows. No test-selected checkpoints or tuning were used.

The candidate fails the registered acquisition and N64 transfer criteria in every seed and arm. Longer recurrence plus replay reaches 100% final Dijkstra correctness on the tested N16 chains, but all arms score 0% functional correctness on every final N64 task/family cell. Thus familiar-length chain success does not establish a transferable algorithm. Longer recurrence takes about 3.6–3.7 times training time at eight times forward message candidates compared with its matched replay choice.

## Whole-graph correctness on random N16 graphs

| Arm | Dijkstra after first task | Dijkstra after MST | MST after MST | Dijkstra after return | MST after return |
|---|---:|---:|---:|---:|---:|
| short_online | 11.1% | 1.4% | 2.8% | 27.8% | 0.0% |
| short_replay | 13.9% | 16.7% | 0.0% | 19.4% | 0.0% |
| size_online | 23.6% | 1.4% | 4.2% | 59.7% | 1.4% |
| size_replay | 29.2% | 40.3% | 8.3% | 63.9% | 6.9% |

## Larger inputs after the final stage

| Arm | N64 random Dijkstra | N64 random MST | N64 chain Dijkstra | N64 chain MST |
|---|---:|---:|---:|---:|
| short_online | 0.0% | 0.0% | 0.0% | 0.0% |
| short_replay | 0.0% | 0.0% | 0.0% | 0.0% |
| size_online | 0.0% | 0.0% | 0.0% | 0.0% |
| size_replay | 0.0% | 0.0% | 0.0% | 0.0% |

## Charged work

| Arm | Train seconds | All evaluation seconds | Graph presentations | Message candidates | Persistent tensor bytes | Peak CUDA training bytes |
|---|---:|---:|---:|---:|---:|---:|
| short_online | 24.08 | 0.89 | 12288 | 6291456 | 187296 | 70570496 |
| short_replay | 24.58 | 0.85 | 24568 | 12578816 | 336800 | 72780288 |
| size_online | 89.04 | 8.99 | 12288 | 50331648 | 187296 | 78182912 |
| size_replay | 89.20 | 8.96 | 24568 | 100630528 | 336800 | 88005120 |

Training message counts are dense forward candidates, with corresponding backpropagation; they are not measured FLOPs. Public-N recurrence uses eight times two-step message candidates during N16 training. A full loss trace, all evaluation/query counts and predictions are retained. Persistent tensor bytes include model, optimizer, used reservoir tensors and torch/CUDA RNG; they exclude Python object overhead and runtime libraries. Peak CUDA allocation is a separate training measure. Common dataset generation, label production, evaluation validation and checkpoint I/O are not hidden in these per-arm training timings; their available costs appear in source identities and raw records. All runtime measurements are hardware-specific.

Exact CLRS source algorithms produce functionally correct targets, independently checked with NetworkX. Their CPU query timing includes the local probe adapter, excludes independent validation, and is stored separately per evaluation cell. This is not a speed comparison against an optimized or hardware-matched classical baseline. No published CLRS training result was reproduced.

## Registered criteria and falsification

Acquisition requires at least 80% functional whole-graph correctness on random N16 tasks; retention allows at most five percentage points lost after switching; N64 transfer requires at least 80%. Retention is only scored as a pass/failure when the task was acquired. Earlier low accuracy followed by later low accuracy is not evidence of successful retention.

- short_online: Dijkstra acquired 0/3; MST acquired 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.
- short_replay: Dijkstra acquired 0/3; MST acquired 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.
- size_online: Dijkstra acquired 0/3; MST acquired 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.
- size_replay: Dijkstra acquired 0/3; MST acquired 0/3; final both-task N64 random transfer 0/3; N64 chain transfer 0/3.

The untrained two-step locality counterexample is structural. Longer recurrence removes that particular receptive-field limit, but a trained failure can still reflect optimization, representation, supervision or insufficient experience. This experiment does not distinguish all those causes. More recurrence alone is not a demonstrated general reasoning solution.

## Audit

All 36 checkpoint hashes, optimizer step counts, graph/message counters, finite weights and recorded pointer metrics were verified. Reservoir contents and private RNG were reconstructed from the original stream at every stage; paired initialization and dataset hashes match. The scored runner restores all twelve final models and reproduces every saved final prediction exactly on the GPU. Evaluation checks preserve weights, memory sampling RNG and torch/CUDA RNG.
