# Current continuation: diagnose frozen computation depth

E27 is complete and audited. Read `E27-RESULT.md` before another design revision.
Longer recurrence learns N16 chains but fails N64 whole-graph transfer, while
random-task acquisition remains below the declared threshold. No ongoing run
needs restarting. E28 is not yet implemented or registered.

Freeze a diagnostic using the existing final checkpoints and multiple explicit
inference depths, separating input size from recurrence length. Primary usable
controls should include the fixed training depth and public node-count budgets.
Report all source seeds and arms, charge every query, and verify weights,
optimizer, memory and RNG remain unchanged. Any best-depth choice based on gold
answers is only a privileged diagnostic, never an agent capability. No new
training, test-based checkpoint selection or admission policy should enter this
attribution test.

Use the outcome to distinguish too little recurrence from damage caused by
running a learned transition beyond its training depth. Neither alone isolates
optimization versus representation. Randomized-depth training and training-only
intermediate supervision are later controlled hypotheses, not automatic repairs.
A graph-only pilot remains insufficient for the full architecture goal.
