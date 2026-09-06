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

The first complete case,32101 batch1_lr1e3, is independently audited at both
milestones. Tutorial rule correctness falls129/142→93/142 from256→1024 gradient
examples; actual deliveries fall4/12→0/12. This is an interim counterexample to
more fitting as an automatic repair, not a comparison of the four controls.
Do not select an arm or interpret the experiment before all cases finish.

The source audit of imitation is separate; it changes no E32 code or settings.
The full architecture goal, learned compression, general reasoning, reliable
RSI and efficient long-horizon learning remain unachieved.
