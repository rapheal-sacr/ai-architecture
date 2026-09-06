# E35: bounded fast-state preservation without context identities

Read FALSIFICATION.md and E34-RESULT.md first. E34's oracle archive restores
competence but cannot show a usable memory system. E35 tests one bounded repair:
a bank of frozen fast-weight states behind the fixed E2E prefix, plus an active
state that can continue learning. No new outer training is performed. This is
a fixed human-written controller around a learned predictor, not a learned
admission rule, recursive self-improvement or established architectural novelty.

The roots attacked are retained distinctions (protect acquired weights from
later destructive updates) and useful access/computation (select a protected
state from already observed prediction error while sharing compatible prefix
work). It cannot create missing identifying observations. With indistinguishable
current histories, a hidden rule switch is not knowable in advance. The first
errors after a switch remain part of the score. The practical question is whether
recovery becomes cheap enough relative to useful context lifetime and bank size.

The exact frozen settings are protocol_e35.json. Both E33 E2E objectives, all
three source seeds, all five policies and all three stream regimes execute.
8192 targets per stream is 42.7 times an E33 training episode's192 targets, but
only a small synthetic horizon. A separate stream is drawn per seed and regime;
paired policies/objectives share it. Three seeds are not a precise population
estimate. No result-dependent stopping, source checkpoint selection or tuning.

Each context is a random directed cycle over16 symbols. A hidden change chooses
another context, with durations uniform over the stated inclusive range. The
new map is applied to the current token: no BOS, random restart or boundary is
inserted. Expanding streams add new maps until12 slots; drifting streams swap
two positions of a selected cycle with probability.35. Such a version can later
recur; map identity is the full successor function, not the slot label. Changes
may fall inside chunks. KV and positional counters never reset at hidden changes.
All context, boundary and map metadata belongs to the evaluator only. Fresh
maps are checked against all E33 training/evaluation maps and other E35 streams;
legitimate recurrence within one drifting stream is permitted.

Operational logits are produced with the currently active state. After targets
arrive, archived candidates are scored on that same observed chunk. The lowest
loss candidate is selected only if it beats active mean CE by more than.1.
Its frozen weights are copied into the active branch, affecting future chunks;
current operational predictions are never replaced. Gated arms update the
selected active branch if any token was incorrect OR mean CE exceeds.5. One
clipped SGD update is charged; selection can require an extra suffix forward.
The always-update control uses the same shared kernel and original SGD rule.

Every16 observations, the current post-update state becomes a frozen probe for
the *next*16 observations. An initial probe starts at initialization. Admission
requires at least75% correctness and mean CE at most1 on those future observations.
Exact argmax agreement with an existing archive throughout that probation sample
and at least75% archive accuracy marks a duplicate; it refreshes that archive's
recency. This is finite-sample deduplication, not proof of functional equivalence
or learned compression. Otherwise admit, evicting least-recently-used if full.
Actual selection also refreshes recency. No context-aligned snapshot or label
is available. One extra probe is always charged for nonzero-capacity policies.

In this core the sole adaptive MLP follows the last attention cache. Prefix
features and KV are therefore independent of candidate fast weights. The fast
MLP, the subsequent static MLP and output head are recomputed for every candidate.
Two or more adaptive layers are explicitly rejected: later attention could then
depend on earlier fast weights. A model/config/policy-version check rejects
incompatible serialized state; it does not transport memories into a new encoder.

Primary scoring includes all actual predictions, warm-up and change costs.
The limited recurrence-benefit criterion requires bank8 to lower aggregate NLL
and errors against active_gated for each source-training arm, with no seed's
recurring NLL worsening. Report every regime and seed even if this fails.
Conditional return metrics require that same learner's previous occurrence of
the exact map passed75% on its final16; count denominators and never label
unacquired rules as forgotten. Report full-stream and latter-half quality.

Count all prefix tokens, dense attention score entries, suffix forwards/tokens,
gradient updates/tokens, bank events, tensor bytes and wall time. Observations
are equal; FLOPs and runtime are not. Wall time is CPU process-level measurement,
not an isolated accelerator throughput benchmark. Candidate tensors, probe,
active state and KV are bounded; model weights, Python/allocator/graph overhead,
serialized diagnostics and elapsed-position metadata are additional. Output
logs grow and are evaluation artifacts, not free persistent agent memory.

The no-selection archive must exactly reproduce active_gated's operational
outputs and fast/KV state while paying archive work. A four-token-delayed
last-observed-successor table is a specialized information-matched control;
report accuracy,256 logical int64 key/value bytes, and runtime separately.
This grammar has an inexpensive explicit solver; beating another neural arm
does not establish efficient general learning. Frozen source hashes, exact
full-core replay, actual save/resume, independent transition/metric/event
checks and capacity invariants must pass before interpreting scored results.
