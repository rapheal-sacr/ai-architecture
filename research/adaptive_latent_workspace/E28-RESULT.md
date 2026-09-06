# E28 — frozen recurrence-depth attribution

Source frozen at `0105e35a60d433f7acb93b02bc027b9f0b7228ee`. All twelve final E27 checkpoints were evaluated at depths 2, 8, 16, 32, 64 and 128; 864 source/size/task/family/depth cells and 20,736 graph queries. No parameters, optimizer moments, episodic memory or RNG changed. The broad architecture goal remains active.

These are the same saved E27 queries, with depth intervened on after training. Every original-budget prediction reproduces E27 exactly. This is causal attribution of inference depth on those states, not an untouched new benchmark or evidence of newly learned competence. No depth is selected using a correct answer.

## Deployable budget comparisons

Entries are mean whole-graph correctness across three seeds, each with 24 graphs per cell. Policy costs use their actual grid rows; aliases reuse work rather than counting another execution. Training depth is two for short models and sixteen for size-trained models.

| Source | Budget | Dijkstra N16 chain | Dijkstra N64 random | Dijkstra N64 chain | Prim N16 random | Prim N64 random | Query seconds / full 288-graph set |
|---|---|---:|---:|---:|---:|---:|---:|
| short_online | training_depth | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.241 |
| short_online | fixed16 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.011 |
| short_online | public_N | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.233 |
| short_online | public_2N | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.372 |
| short_replay | training_depth | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.208 |
| short_replay | fixed16 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.010 |
| short_replay | public_N | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.238 |
| short_replay | public_2N | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.367 |
| size_online | training_depth | 97.2% | 0.0% | 0.0% | 1.4% | 0.0% | 1.010 |
| size_online | fixed16 | 97.2% | 0.0% | 0.0% | 1.4% | 0.0% | 1.010 |
| size_online | public_N | 97.2% | 0.0% | 0.0% | 1.4% | 0.0% | 2.234 |
| size_online | public_2N | 18.1% | 0.0% | 0.0% | 0.0% | 0.0% | 4.365 |
| size_replay | training_depth | 100.0% | 0.0% | 0.0% | 6.9% | 0.0% | 1.016 |
| size_replay | fixed16 | 100.0% | 0.0% | 0.0% | 6.9% | 0.0% | 1.016 |
| size_replay | public_N | 100.0% | 0.0% | 0.0% | 6.9% | 0.0% | 2.234 |
| size_replay | public_2N | 27.8% | 0.0% | 0.0% | 0.0% | 0.0% | 4.382 |

## Does additional recurrence preserve a familiar-length solution?

Dijkstra N16 chain correctness, without changing graph, weights, memory or decoder:

| Source | 2 steps | 8 steps | 16 steps | 32 steps | 64 steps | 128 steps |
|---|---:|---:|---:|---:|---:|---:|
| short_online | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| short_replay | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| size_online | 0.0% | 0.0% | 97.2% | 18.1% | 15.3% | 12.5% |
| size_replay | 0.0% | 0.0% | 100.0% | 27.8% | 9.7% | 5.6% |

Doubling public-N depth improves functional graph accuracy in 0/144 source/task/size/family cells and worsens it in 24/144; the rest tie. These cells share models and graphs and are not independent replications. Raw per-seed contrasts are preserved.

Short fixed-depth local processing still has a structural communication limit on sufficiently long chains. Conversely, increasing a shared transition’s execution length does not guarantee that it preserves an answer it already learned to produce. These interventions cannot fully distinguish optimization, representation, state-transition instability, and training-depth specialization. They do not establish that randomized-depth training or another normalization would repair transfer.

## Cost and state audit

The full diagnostic executed 20,736 graph queries, 1,548,288,000 dense forward message candidates and 37,158,912 decoder candidates. Recorded neural query time totals 178.74 seconds. This includes every tested depth; the per-policy table is not the cost of running the whole diagnostic. Message candidates are not measured FLOPs. GPU timing is hardware-specific and includes transfer of predicted pointers; reference validation and source setup are separate.

All twelve checkpoint file hashes and loaded model/optimizer/memory/RNG state hashes remain equal to their source. All source datasets and completion records retain their hashes. Every original E27 query prediction is exact, all grid work counts and recorded parent metrics check, and primary policy rows match their fixed-depth executions. No retraining, checkpoint selection or test-dependent halting was performed.
