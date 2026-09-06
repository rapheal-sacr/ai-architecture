# Recurrent reasoning after the memory-access tests

Read `FALSIFICATION.md` first. E26 integrates online access with persistent
expert learning, but its gains cost substantially more work. It is still a
supervised predictor and does not implement general multi-step reasoning.
Further variants of that assay cannot close this gap.

## Implemented foundation, not trained competence

`reasoning_processor.py` now implements a 15,177-parameter graph processor.
It encodes the source-node flag and requested task, reinjects those input features
at every step, passes directed edge-conditioned messages, updates node states
with a shared GRU and normalization, and decodes parent pointers. Only graph
weights, source and requested task enter the forward call. The default iteration
budget is the public node count; shorter controls use explicitly fixed budgets.
There is no hint state, reference trajectory length or label-derived stopping
argument. There is no trained reasoning result yet.

This is established message-passing/neural-algorithmic machinery. TTC-LR's supplied
paper and cloned Raven implementation motivate shared recurrence and input
reinjection. CLRS supplies direct processor prior art. This component is not a
novel Transformer, a text encoder or the integrated autonomous architecture.
It currently has no online graph memory or learned self-updater.

`clrs_reference.py` executes the original Dijkstra and Prim function bodies from
the pinned official CLRS clone. It supplies only lightweight rank assertions and
probe-recording plumbing; intermediate hint values are discarded. The original
algorithm code remains in the clone and its hash is recorded. This adapter does
not install or reproduce the complete JAX/TensorFlow CLRS pipeline or any
published benchmark scores. NetworkX independently checks source-component
shortest paths or minimum spanning trees, including disconnected inputs.

## What the preflight establishes

`preflight_reasoning.py` verifies 96 independent reference cases, finite nonzero
neural gradients, permutation-equivariant scores and repeatability at the same
public iteration budget. These are functional checks, not trained task accuracy.

It also constructs a real counterexample to insufficient local depth. On the
same bidirectional 16-node chain, sources at opposite ends require parents 6
and 8 for node 7. With two local processor steps, the scores for node 7 are
exactly equal under those two sources. Neither its own latent state nor the
latent states of its adjacent candidate parents have received the distant source
signal. No choice of parameters or width removes that locality restriction.
Changing the communication mechanism or adding enough steps can change the
restriction, but whether training learns the necessary computation is open.

The bound is specific to this local processor and its inputs. It is not a
universal bound on AI reasoning: a different attention, cached solution or
explicit algorithm changes the available computation. The root attacked here is
turning available graph information into a useful answer within a compute budget.

## Supervision and timing must remain explicit

The inspected CLRS `nets.py` path uses hint trajectories, teacher forcing and
per-example trajectory lengths. It sets message-passing scan length from a hint
tensor's time dimension and freezes output updates according to `lengths`.
Merely turning off hint encoders does not by itself remove all timing metadata
from this path. This observation concerns that implementation, not a claim that
all CLRS settings or results are invalid.

Our proposed experiment will use final parent targets only and a budget derived
from public input size. Gold hint states, hint losses and gold stopping times
must not enter candidate prediction or training. The explicit task identity is
a user-requested operation, not a hidden regime supplied to solve inference.
The classical source algorithms remain a strong explicitly labeled baseline;
learning to approximate them is not itself an efficiency victory.

## Proposed first training and falsification sequence

Train on a sequential shortest-path, spanning-tree, shortest-path curriculum,
retaining neural weights, optimizer and a bounded graph replay buffer. Use
positive continuous edge weights to avoid canonical-parent ambiguity on most
cases. Freeze graph seeds, replay capacity, training budget, iteration controls
and reporting rules before any scored training. The runner and full E27 protocol are now implemented in `e27_reasoning.py`
and `protocol_e27.json`; scoring starts only after their source commit is frozen.

Compare the shared recurrent processor at a public-size budget with the same
parameterization at a short fixed depth. Both need matched observations,
explicit replay costs and complete processor/message counts. Do not describe
larger recurrence as equal compute. Also report the exact classical baseline,
including its label-generation and query cost where used.

Use fresh graphs, larger node counts, sparse/bidirectional long chains and
random relabelings. Graph size alone is not dependency length: dense graphs can
have very short paths. Score canonical pointers separately from functional
correctness where ties permit multiple answers. Report whole-graph correctness,
invalid/cyclic parent structures, task retention after each stage and all failed
acquisitions. A larger local receptive field is necessary on the chain example,
but it is not proof of learned reasoning or transfer.

Only a trained result can justify the next integration with persistent episodic
facts, compressed state and procedural self-application. General reasoning,
long-horizon autonomous efficiency and scientific novelty remain open.

## E27 registered comparison

Three fresh seeds, four arms (two versus public-N steps, each with/without
128-graph reservoir replay), 512 updates per stage and eight new graphs per
update. Sequence: Dijkstra, Prim, Dijkstra. The 15,177-parameter model and Adam
state persist. Replay adds up to eight prior graphs per update and is charged;
longer recurrence costs eight times the training message candidates at N16.
Positive weighted graphs mix relabeled random graphs with chains. Fresh tests
use N16/N32/N64; chain sources lie at an endpoint before relabeling. Each cell
contains 24 graphs, so uncertainty and per-seed counts must remain visible.

Acquisition is assessed as at least 80% functionally correct whole graphs on
fresh random N16 inputs, retention as at most five percentage points lost after
the switch, and N64 transfer as at least 80%. These are declared pilot criteria,
not a universal definition of reasoning. Parent accuracy alone is insufficient:
locally plausible pointers can form invalid cycles. Failure to acquire must not
be mislabeled catastrophic forgetting. Test queries never select checkpoints,
set stopping times, enter replay, or modify training. Full tensor costs and
training/evaluation message candidates are distinct from measured FLOPs.

## Distinguish possible causes before another training change

The supplied TTC-LR review explicitly describes randomized recurrence depth and
truncated backpropagation; E27 deliberately uses fixed train depth and full
backpropagation in a different, much smaller graph processor. E27 failure must
not be reported as a replication failure of that language-model work.

A useful next diagnostic is frozen-checkpoint inference at several declared
budgets, separately from changes in graph size. Evaluate all seeds and both
recurrence-trained models; keep any evaluator-selected best depth explicitly
privileged and unavailable to the candidate. Public N and fixed training depth
are deployable controls. Additional steps can expose a computation-depth
mismatch, but cannot by themselves distinguish poor optimization, an unstable
learned state transition, and insufficient algorithmic representation.

Only after this attribution test consider randomized train recurrence against a
fixed-depth control with matched expected work, or training-only intermediate
supervision against final-only targets. Training supervision and inference-time
oracle information are different resources and must be stated separately. No
such follow-up protocol or scored run is registered yet.

## E27: trained recurrence succeeds locally and fails size transfer

All twelve registered cases completed from `505da5d`; all 36 stage checkpoints
and three paired stream/replay histories pass audit. The shared 15,177-parameter
processor learns Dijkstra, Prim, then Dijkstra, with explicit requested-task IDs
and final parent targets only. No gold hint trajectories or target-derived
stopping times enter the candidate. Initial weights and current observations
match across two/public-N recurrence and no-replay/128-graph-replay arms.

No arm or seed meets the declared 80% random-N16 acquisition criterion after its
first Dijkstra or Prim stage. Final random-N16 Dijkstra correctness averages
27.8% short/no-replay, 19.4% short/replay, 59.7% size/no-replay and 63.9%
size/replay. Prim remains poor. Size/replay nevertheless solves all 72 tested
final N16 Dijkstra chains; size/no-replay solves 70/72. At N64, every final
arm/task/family cell has zero functionally correct graphs. High pointer accuracy
can conceal incorrect solutions: short/no-replay's final N64 random Dijkstra
parent accuracy is 83.4%, with zero optimal whole graphs.

Replay preserves or improves some earlier accuracy during the task switch, but
successful retention of the registered random task cannot be claimed where
acquisition failed. Public-N recurrence costs eight times forward training
message candidates and about 3.6–3.7 times training time at the same replay
choice. Size/replay uses about sixteen times short/no-replay message candidates
and 1.80 times persistent tensor bytes. More computation has not established
size-general algorithmic reasoning or an efficient autonomous agent.

This falsifies the current component's sufficiency, not recurrence in general.
Fixed training depth, optimization, representation and supervision remain
confounded explanations of transfer failure. `E27-RESULT.md` and its preserved
predictions include every failure. The next bounded diagnostic should change
inference depth on frozen checkpoints before changing training. No E28 protocol
or run is yet registered. Full integrated learning, compression, reliable RSI
and architectural novelty remain unproven.

## E28 frozen-depth attribution protocol

`protocol_e28.json` evaluates every final E27 source at fixed depths
2, 8, 16, 32, 64 and 128 on its exact saved queries. The registered usable
policies are training depth, fixed16, public N and public 2N. The grid executes
each distinct depth once; policies reuse the corresponding rows without
pretending the full search is free. No oracle best-depth policy is deployed.

Every original-budget prediction must match E27 exactly. Source files, loaded
checkpoint contents, model weights and torch/CUDA RNG must remain unchanged.
The experiment tests a computation-depth intervention, not new task learning or
fresh benchmark performance. Freeze the runner and protocol after the separate
preflight, then retain every source and failure without changing the grid.
