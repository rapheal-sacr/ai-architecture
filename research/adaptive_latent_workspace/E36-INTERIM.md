# E36 historical progress record — superseded

E36 main and audits are complete. Read E36-RESULT.md for results and process
provenance. The live-handle statements below are historical only; do not restart
51019/24107. The current supplemental intervention is tracked separately in
RUNS/ACTIVE-STATE.json.

# E36 in progress — no scored conclusion

Scorer51019 is running under frozen commit
`2d3eff32a79a1561ba76d98a81b6a7f10bc92c04`. It writes to the WD research drive's
`alw-runs/e36`. Poll the actual handle; a manifest saying running is not proof
that it remains live. No second scorer or restart is authorized by a quiet poll.

Data generation confirms768 unique training maps and26 unique evaluation maps
with no overlap. Three seeds share exact within-seed training data across static,
first-order and full E2E objectives. The scorer runs128 updates for all9 models,
then120 neural and24 specialized-control closed-loop evaluation cases. See
E36-PROTOCOL.md for which quantities are private evaluator diagnostics.

The latest verified poll delivered E36_TRAIN32 for all9 models. Process1097874
was concurrently observed active at about100% of one CPU with554,644KiB RSS.
These are historical observations, not substitutes for polling51019 again.
No scored action-evaluation or audit result is available at this update.
summarize_e36.py is prepared and exercised on development data; it requires an
audit matching the scored complete-file hash before generating scored tables.

The development run and audit are complete and preserve both earlier attempts.
They establish mechanical correctness on reduced budgets, not acquired action
competence. The final source-frozen scored run requires its own complete audit
before interpreting improvement, failure, acquisition or efficiency.

The broad architecture goal is still active. Passing this six-state finite
planner assay would not establish learned goal invention, general reasoning,
learned memory compression or recursive self-improvement.

Later verified progress: scorer51019 finished128 training rounds for all9 models
and entered action evaluation. Read-only training auditor24107 was then launched
after the final training checkpoints were available. Staged training/evaluation
audits match the complete development audit on all counts and numerical checks.
Do not repeat scored training replay when the training-stage artifact completes;
evaluation mode validates its source/data/final-checkpoint hashes before using it.
Both processes remain live at this update. Some action timings now overlap the
one-thread audit on another CPU; they are observed timings, not isolated latency.
Training timings precede this overlap. The scorer's seven frozen sources remain
byte-identical to2d3eff3.

E36-CONSTANT-CONTROL.md adds8192 independently checked actions and312 cycle
checks. Both fixed actions complete many goals without learning, so the final
tables must include them. This was added before scored action outputs were
available and changes none of the primary experiment. The updated summarizer
accepts --constant-control, checks the frozen data hash, and includes both fixed
policies without per-world best-selection.
