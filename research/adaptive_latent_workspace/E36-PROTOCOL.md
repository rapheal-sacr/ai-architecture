# E36: action learning before learned goal control

Read FALSIFICATION.md, including the attachment P1. The prior goal turn was
progress: it produced audited falsifiers, a full hypothesis ledger and checked
action-planning machinery. E36 uses that machinery to learn from actual actions.

Three equal-initialization arms train on the same uniform-action trajectories:
ordinary static prediction, first-order meta-learning with fast updates, and
full E2E differentiation through those updates. Each outer example executes96
real transitions on new maps in A/B/A segments of32 actions. Fast weights and
attention history persist across hidden changes; both reset for the next new
training world. Optimize average pre-update transition cross entropy. The outer
objective is predictive, not task reward: testing whether it supports action
success is the point of this stage. A future learned goal policy must not be
credited to this fixed planner.

All three seeds run128 AdamW updates with fixed settings and no checkpoint
selection. Three arms × three seeds ×128 updates ×96 transitions =110,592 paid
training transitions, plus all inner and outer differentiation. Between-arm
input equality does not imply equal compute. Save optimizer/model checkpoints
and per-example receipts so updates and trajectories can be replayed.

Evaluation has eight fresh worlds: two each stationary, randomly recurring,
drifting and irreducibly noisy. Every action follows a random directed cycle;
drift swaps two state labels for one action's cycle. Noisy worlds replace a
transition with a uniform destination20% of the time. No training/evaluation
map overlaps are permitted; repeated maps within a declared recurring world
are intentional. The6-state/2-action alphabet is public and fixed, which limits
the transfer claim. Goals arrive externally in a reproducible sequence; this
is not autonomous goal generation.

Evaluate five neural conditions per seed/world: static frozen/static TTT,
first-order TTT, E2E TTT/E2E frozen. Each uses the same finite Bellman planner,
six-step horizon, .95 discount, temperature .1 and10% uniform exploration.
Independent hypothetical state/action queries share the current real history;
discarded queries never install imagined outcomes or updates. Charge all12
queries per action plus the executed-action forecast and its real update.

Controls are uniform actions, observed-edge BFS, and rolling8-observation
Dirichlet transition counts with the same probabilistic Bellman planner. The
counts use .25 pseudocount per outcome. BFS has no noise-aware special handling.
These small specialized controls are deliberately strong: a much larger neural
memory must earn its extra work, not merely beat a memoryless agent.

Each evaluation lasts512 paid actions with hidden durations uniformly32–96.
Maps, schedules, goal RNG and noise RNG are shared across policies. Different
actions produce different observations and different goal completion times, so
this is a whole-policy comparison, not identical-evidence evaluation. Preserve
every receipt and actual prediction. The model's already-computed all-state
predictions can be evaluated privately against the current map, but the planner
never receives that label or diagnostic. The learned controller is never told
that a context changed. Save complete real states every128 actions.

Primary outcome is goals completed per512 actions; report all seeds/worlds,
actual transition NLL/accuracy, map coverage, censored time to11/12 transition
argmax acquisition, updates, hypothetical queries, attention work, Bellman terms,
training and acting wall time, and stored tensor/control state. Environment and
diagnostic time are not model latency; all timings retain their scope.

Reject the sufficient action-usefulness claim if better transition learning does
not improve goal completion, or if a simple observation-memory control dominates
at much lower cost. Preserve occasional gains and failures across regimes.
No finite-world success can establish general reasoning, arbitrary memory
compression, learned goal-space invention or recursive self-improvement. The
next design decision must follow the outcome, not silently redefine success.
