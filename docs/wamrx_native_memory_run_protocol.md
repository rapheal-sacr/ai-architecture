# WAM-RX registered native-memory run protocol

**Status:** frozen and structurally preflighted; E0.16 `NOT_RUN`

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

The default invocation prints the checked-in `NOT_RUN` record. If interrupted,
the only valid continuation is the same command with `--resume`. MoE, LoRA,
continual consolidation, autonomous promotion, and self-improvement remain out
of scope until E0.16 reaches one registered terminal outcome.
