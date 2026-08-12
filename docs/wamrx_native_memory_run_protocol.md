# WAM-RX registered native-memory run protocol

**Status:** complete; terminal decision `COMPLETE_RETAIN_EXPLICIT_MULTIVIEW`

This protocol closes the gap between the E0.15 non-neural authority assay and
the smallest accuracy-bearing native-working-memory comparison. It is additive:
the selected four-block core, E0.15 authority boundary, task generator, and
split hashes remain unchanged.

## Frozen comparison

E0.16 pairs all three arms by the five selected E0.14 core seeds:

- explicit multiview reconstruction at query time;
- a deterministic exact-key, sixteen-slot session cache;
- one learned 472-by-4 affine gate selecting remember, update, merge, or forget.

The learned gate has exactly 1,892 trainable parameters. It consumes only the
frozen core prelude encoding of the current family, turn, and query. It has no
answer readout, recurrence, hidden layer, adapter, ledger write capability, or
gradient path into the selected reasoner. Reset, expiry, poison rejection,
evidence reinjection, invalidation, and both capacity caps remain mandatory
non-neural controls.

Training is fixed at 400 AdamW updates per seed with a batch of sixteen: four
examples from each operation class. The five schedules, initialization offsets,
checkpoint cadence, optimizer settings, example registries, and content hashes
are frozen in `contracts/wamrx_native_memory_run_v1.json`.

## Evaluation and accounting

Each arm receives one common fixed-core readout for every one of the 36 ID and
36 OOD tasks. The learned arm additionally pays for one frozen-core prelude and
one affine gate call on every gate-eligible turn. Explicit-view scanning,
native-memory operations, serialized slot bytes, gate checkpoints, evaluation
journals, and model inference FLOPs are reported separately.

The primary measure is exact canonical-JSON answer accuracy by arm, seed,
split, family, and protected region. Required secondary reports cover
correction latency, reset leakage, a capacity curve at 0/4/8/12/16 occupied
slots, unsupported durable writes, and immediate behavior after support
invalidation. No ID, OOD, family, protected-region, or secondary performance
metric is read during registration or resume validation.

## Statistical and safety gate

The paired unit is the core checkpoint seed. Learned-versus-explicit and
learned-versus-cache OOD superiority use one-sided paired Student-t bounds with
Holm step-down. ID has a registered -0.02 non-inferiority margin; every
protected-region and compute-normalized lower bound must be nonnegative or
positive as specified by the contract.

Learned memory can be adopted only when the primary and compute-normalized
gates pass and every authority, deletion, reset, poison, capacity, checkpoint,
and accounting manipulation passes. The other complete outcomes explicitly
retain deterministic cache, retain explicit multiview memory, classify a
compute-only accelerator, or limit the gate to one specialist family.

## Checkpoint, resume, and failure rules

Gate checkpoints are written every 100 updates and bind seed, schedule, run
contract, exact core checkpoint, completed updates, examples seen, gate weights,
and AdamW state. The preflight proves two updates are bitwise identical across
an update-one interruption without reading evaluation metrics, and verifies
that the frozen core parameter hash is unchanged.

Only `INCOMPLETE` runs may resume, using `--resume` against the full frozen
manifest and append-only evaluation journal. Identity drift, non-finite values,
authority violations, conflicting journal rows, or any other non-resumable
integrity failure produce `INVALID`; an `INVALID` run requires a new versioned
protocol. Missing seeds or rows are never dropped or imputed.

## Artifact and runner boundary

All five exact E0.14 selected checkpoints are present at their registered,
ignored relocation paths. The artifact checker reports `READY` after matching
their byte counts, file hashes, metadata identities, state hashes, and E0.14
records. Substitution and core retraining remain prohibited.

The runner is inert unless the explicit execution flag is present:

```bash
python3 tools/check_native_memory_core_artifacts.py
python3 tools/check_native_memory_run_registration.py
.venv-m3/bin/python rig_a/experiments/e0_16_native_memory_comparison.py
.venv-m3/bin/python rig_a/experiments/e0_16_native_memory_comparison.py --execute-registered
```

The registered budget is consumed. The default invocation prints the checked-in
terminal record; it must not be rerun or resumed as memory development.

## Terminal result and selection

E0.16 completed every seed, 2,000 gate updates, 1,080 unique task rows, all
capacity/invalidation probes, and all eleven manipulations with no invalid or
incomplete reason. Every arm scored 4/36 per seed on ID and OOD. Both
learned-versus-simple OOD accuracy deltas were zero, so neither Holm superiority
claim rejected. The learned compute-normalized mean delta was -0.05366 versus
explicit and -0.05365 versus cache, and learned correction success was zero.

The operation gate itself classified its auxiliary labels at mean 94.8% ID and
93.6% OOD, but this did not improve frozen-core answer accuracy. Reset leakage,
unsupported durable writes, and stale post-invalidation emissions were all
zero; the capacity curves reached the sixteen-slot cap and failed closed.

`contracts/wamrx_native_memory_selection_v1.json` binds the exact terminal
result SHA-256 and run manifest to `explicit-multiview-v1`. The cache and
learned gate are preserved as negative evidence and are not selected for
general memory. The next metamemory/compression policy may operate only over
explicit ledger-derived views. MoE is not blocked by this negative memory
result, but it still requires a separate frozen contract; LoRA, continual
consolidation, autonomous promotion, and self-improvement remain unauthorized.
