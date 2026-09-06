# E32 interim record — experiment still running

Read FALSIFICATION.md first. Frozen scorer/protocol:7d4edc9. Unified process
session62513 runs all12 cases/24 snapshots on WD. See RUNS/ACTIVE-STATE.json
and alw-runs/e32.log for current state; never restart on observation timeout.
This record is a snapshot and must be superseded by a complete audited report.

The historical256-update replay, independent mean-gradient calculation and
all eight functional-preflight snapshots pass. The independent audit verifies
example/group order, optimizer steps, RNG/work state, reference metrics, real
world trajectories and memory. All30 scored world specifications and world
labels are disjoint.

Two cases/four snapshots for32101 are independently audited. Ten cases remain.

| Arm | Gradient examples | Tutorial correct | Initial /4 | Larger /4 | Total /12 |
|---|---:|---:|---:|---:|---:|
| batch1_lr1e3 |256|129/142|2|0|4|
| batch1_lr1e3 |1024|93/142|0|0|0|
| batch1_lr1e4 |256|119/142|2|0|4|
| batch1_lr1e4 |1024|140/142|3|0|5|

The lower-rate final state passes the tutorial gate and initial acquisition,
but solves no larger-world mission. The higher-rate state deteriorates with
more fitting. This is an interim distinction between fitting and execution,
not a completed comparison of all four controls and three worlds. Do not select
an arm or declare the full experiment successful before every case is audited.

The source audit of imitation is separate; it changes no E32 code or settings.
The full architecture goal, learned compression, general reasoning, reliable
RSI and efficient long-horizon learning remain unachieved.
