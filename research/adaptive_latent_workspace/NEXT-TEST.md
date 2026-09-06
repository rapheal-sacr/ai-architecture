# Current continuation: finish E29 variable-depth training

E28 is complete and audited. E29 is implemented and has passed its separate GPU
preflight plus an exact E27-baseline equivalence check. `protocol_e29.json` freezes
three seeds, fixed16/fixed24/variable16–32 replay arms, and public-N/fixed24/public-
2N evaluation. Fixed24 and variable depth have identical message work at every
stage. Verify the recorded live session or terminal completion before restarting.
Do not change the protocol after scored outcomes.

When complete, audit all 27 stage checkpoints, raw graph stream identities,
all three paired replay/RNG histories, realized depth schedules, optimizer steps
and message counts. Preserve all acquisitions that fail, all N64 outcomes and
all inference policies. The runner restores all nine final models and checks
all policy predictions. Build a result report that distinguishes depth stability
from acquisition, retention and generalization; extra recurrence is not free.

The broad architecture remains unproven. A fixed randomized training schedule
is not recursive self-improvement, and a graph-only pilot cannot establish the
full goal. No new downstream design should be selected from partial scores.
