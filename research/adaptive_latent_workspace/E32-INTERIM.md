# E32 interim record — experiment still running

Read FALSIFICATION.md first. Frozen scorer/protocol7d4edc9. Unified process
session62513 runs all12 cases/24 snapshots on WD. Use RUNS/ACTIVE-STATE.json
and e32.log for current state; never restart on an observation timeout.
This partial snapshot must be superseded by a complete audited report.

Ten completed cases/twenty snapshots are independently audited. Two cases
remain. Historical256-update replay and independent accumulation checks passed.
The partial audit reconstructs references, data/order, optimizer/RNG/work state,
real trajectories and memory. All30 scored world specifications are disjoint.

| Seed | Arm | Examples | Tutorial | Gate | Initial /4 | Larger /4 | Total /12 |
|---|---|---:|---:|---|---:|---:|---:|
|32101|batch1_lr1e3|256|129/142|False|2|0|4|
|32101|batch1_lr1e3|1024|93/142|False|0|0|0|
|32101|batch1_lr1e4|256|119/142|False|2|0|4|
|32101|batch1_lr1e4|1024|140/142|True|3|0|5|
|32101|batch8_lr1e3|256|127/142|False|2|0|5|
|32101|batch8_lr1e3|1024|139/142|True|3|0|5|
|32101|batch8_lr1e4|256|110/142|False|2|0|5|
|32101|batch8_lr1e4|1024|132/142|False|2|0|4|
|32102|batch1_lr1e3|256|124/140|False|2|1|5|
|32102|batch1_lr1e3|1024|136/140|True|4|0|8|
|32102|batch1_lr1e4|256|111/140|False|3|0|7|
|32102|batch1_lr1e4|1024|135/140|True|2|0|3|
|32102|batch8_lr1e3|256|127/140|False|0|0|1|
|32102|batch8_lr1e3|1024|140/140|True|4|1|7|
|32102|batch8_lr1e4|256|99/140|False|3|1|8|
|32102|batch8_lr1e4|1024|122/140|False|3|1|8|
|32103|batch1_lr1e3|256|114/140|False|3|1|7|
|32103|batch1_lr1e3|1024|127/140|False|1|0|3|
|32103|batch1_lr1e4|256|114/140|False|4|2|10|
|32103|batch1_lr1e4|1024|134/140|False|4|1|8|

Both improved fitting and deterioration remain in this record. Passing tutorial
queries does not establish larger-world competence. These are partial cells,
not a completed comparison; do not select an arm before all cases are audited.
Training/evaluation times include shared-machine operation. Later CPU E2E work
can overlap the unchanged GPU scorer; no isolated-hardware timing claim follows.

E2E-REVISION.md and E33-PROTOCOL.md introduce a different meta-objective; they
change no E32 source or settings. Full architecture, learned compression,
general reasoning, reliable RSI and long-horizon efficiency remain unachieved.
