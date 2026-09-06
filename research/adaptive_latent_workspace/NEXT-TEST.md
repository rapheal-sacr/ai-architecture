# Current continuation: train for recurrence-depth robustness

E28 is complete and audited. Extra recurrence damages familiar-length answers
and does not repair N64 transfer. Read `E28-RESULT.md` before another change.
The next bounded training test should compare fixed16, fixed24 and variable
16–32 recurrence with exactly matched mean message work for the latter pair,
using fresh seeds, the same model, persistent optimizer and replay, identical
current observations, and the Dijkstra/Prim/Dijkstra sequence. Charge every
backward pass, replay sample and evaluation budget. Do not call the variable
schedule recursive self-improvement; it is a fixed training policy.

Pre-register public-N, fixed24 and public-2N evaluation without target-dependent
halting. Retain all failed acquisitions and N64 results. Randomized recurrence
is prior art described in the supplied TTC-LR paper; this is a bounded transfer
of an idea, not a faithful Raven reproduction or novelty claim. E29 has not yet
been implemented, frozen or scored.
