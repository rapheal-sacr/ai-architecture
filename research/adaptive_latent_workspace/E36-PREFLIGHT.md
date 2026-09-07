# E36 development preflight — mechanics, not learning evidence

The final development run97793 finished32 evaluation cases in9.407s. Auditor51479
finished in7.521s with every scored source byte unchanged. Six outer updates,
144 actual training transitions,1,024 evaluation actions and384 inference updates
replay;70 noninitial checkpoints match exactly. The480 serial map queries differ
from batched queries by at most9.09e-7, and an independent NumPy/explicit-branch
planner differs from saved action probabilities by at most5.14e-7.

Training replay reuses the training function and verifies every saved logit,
outer gradient norm, optimizer state and model checkpoint. Evaluation updates
use the core directly, bypassing ActionE2E. Physical transition, noise and goal
receipts are separately reconstructed from maps and RNGs without NavigationWorld.
Controller payloads replay, and rolling counts agree with their raw histories.
Executed neural work counters are reconstructed from tensor/query dimensions.

An earlier auditor failed after adaptive updates because its algebraically
equivalent floating-point multiplication order differed from the scorer. Matching
the declared clipped-update operation order restored exact replay; no learner
behavior or hyperparameter changed. The first development scorer predates added
base-weight invariance assertions and final manifest-status recording. The final
development run includes those assertions and passes without a source-change
exception. Both original development directories remain on the WD drive.

These models received only two updates on distinct development seeds and smaller
networks/budgets. Their goal counts are not scored capability evidence and were
not used to select learning settings. The full frozen E36 settings are in
protocol_e36.json and E36-PROTOCOL.md. Full goal remains active.
