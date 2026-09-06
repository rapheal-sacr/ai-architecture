# E33 execution record — no completed result yet

Read FALSIFICATION.md first. Frozen source63baff8; CPU scorer session3205.
WD output is alw-runs/e33, log alw-runs/e33.log. Continue that process; never
restart because a tool observation times out. E32 scorer62513 continues on GPU.
The unchanged protocols record shared-machine timing limitations.

All nine training cases must finish128 outer updates each, then all252 policy/
stream evaluations must finish. Run record_e33.py against the completed output
and independently replay all1152 outer updates and all per-session states.
Keep every acquisition failure and all reset/carry comparisons. E33-PROTOCOL.md
sets criteria before scores; do not select a checkpoint from interim losses.

Functional evidence only: core causal/finite-difference/state checks pass;
six development outer updates and21 sessions replay exactly. The design/source
mapping is in E2E-REVISION.md. No grammar test, even a future pass, establishes
the full architecture, learned episodic compression, general reasoning,
long-horizon autonomous efficiency, novelty or reliable RSI.
