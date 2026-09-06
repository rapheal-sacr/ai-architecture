# Research state — goal still active

Read `FALSIFICATION.md` first. No result in this branch establishes a novel,
general continuously improving agent or efficient very long-horizon reasoning.
The experiments have narrowed the design problem and rejected several attractive
extensions.

## Locally established

- E1's unprotected module bank does not adequately retain returning regimes.
- E2/E5 protected temporary adaptation improves reuse of four conflicting hidden
  functions through actual neural updates. Wider replay controls do not remove
  the gain; copying temporary updates into persistent weights largely removes it.
- E5 active-first routing preserves all stored scored prediction metrics on four
  fresh seeds while reducing counted model evaluations and measured time.
- E3/E4 show that this rule has limited scope: it wastes work under gradual
  drift, exhausts capacity, and loses to shared context replay on an independent
  fixed-function, changing-input problem.
- E6's sample-tested sharing does not establish efficiency. Its whole-stream
  error and time are worse than context replay; a better final window makes
  longer-run amortization an open question rather than a demonstrated benefit.
- E7 feature renewal helps whole-stream prediction in an optimizer-controlled
  external screen. With replay the gain is modest, runtime increases and the
  final window favors replay without renewal.

These statements are scoped to the stored experiments and seeds. They are not
claims about all modular models, all replay algorithms or CBP in general.

## Active experiment

E8 is the first registered closed-loop mechanism pilot. It uses the pinned
MiniGrid/CNN-LSTM/PPO implementations with and without actor/critic feature
renewal, preserving weights and optimizer state across two-room, four-room,
then returning two-room environments. The policy sees partial symbolic images
and receives sparse task reward; it cannot read the hidden grid or transition
function. The preflight exercised real learning and feature replacement.

E8 has completed and is recorded in `E8-RESULT.md`. It exposes severe old-task
loss during unsuccessful harder-task training in both arms. Head renewal improves
later transfer after returning to the easier task, but does not prevent that
loss. These 40/80-step tasks are a foothold, not proof of very long-horizon
competence or the full modular/planning architecture.

E9 and E11 completed. The controller persists but students reset. On fresh E11 tasks, self-application beats equal-budget conventional meta-updates in only 4/9 and 5/9 settings across the two bootstrap states. Reliable superiority and cost-matched efficiency remain unestablished.

E10 completed after a recorded checkpoint-device launch failure and successful CUDA preflight. With shared critic gradients, removing entropy changes harder-task success from 0/50 to 50/50 in both development seeds and retains the old task at 50/50. Detaching the critic is inconsistent across seeds. Baseline evaluation episodes reproduce E8 exactly; training floating-point trajectories do not. Read `E10-RESULT.md`.

E12 passed CPU/CUDA preflights and an exact CPU checkpoint-resume test. Scored execution is running on fresh seeds: both entropy choices start from scratch, then encounter an unseen key-door action chain with a 640-step limit and return to the original task. This remains a fixed-procedure mechanism test, not a planner or learned procedure.

E13 completed its five-million-observation extension with exact prefix error/counter reproduction in all four arms. Cumulative error is lower with sharing in both seeds, reversing the earlier first-million deficit. Mean error improves 21.0%, but forward examples rise 2.42×, runtime 1.64×, and stored tensor bytes 1.45×. It still uses eight persistent modules. Accuracy amortization is established on this continuation; total-resource efficiency and rare retention are not.

E14 passed a real neural world-model/planner preflight and is now running on CPU. The learner predicts observed state changes and rewards, plans only through its learned model, and preserves weights/replay/modules across hidden changes in Pendulum gravity. It compares random actions, one-step and twenty-step replay planning, context replay and guarded sharing. Independent frozen probes measure how one-step errors accumulate over forty imagined transitions. This is fully observable control, not a general reasoner or recursively learned procedure.

## Binding constraint and unresolved measurement

The candidate quantity is amortized recovery cost relative to useful context
lifetime, after charging identification, updates, replay, routing, memory growth
and consolidation. Existing tests measure early/late errors, operation counts,
storage and time, but not a complete recovery-time distribution. No universal
single binding quantity has been measured. `CONSTRAINTS.md` explains the three
roots and why identifying evidence, retained information and useful computation
cannot be substituted for one another.

## What remains to earn confidence

1. A criterion that separates genuinely conflicting conditional functions from
   learnable changes in input distribution or temporary representational underfit.
2. Memory consolidation that repays its full cost, including rare and old-context
   tests outside the admission sample. Extending E6 beyond its first million
   observations with the same prefix is a useful test of the possible late gain.
3. A thresholded, censored recovery-time assay with unpredictable switches,
   overlapping contexts and recurrence, rather than only regular fixed segments.
4. Closed-loop transfer and substantially longer tasks; learned multi-step models
   and planning must be tested against strong explicit/learned baselines without
   privileged transition access.
5. A separately frozen final evaluation after the exploratory design decisions,
   plus a broader novelty comparison. `NOVELTY-AUDIT.md` already prevents claiming
   that task-free replay, copied models, gating or renewal are new inventions.

The working design should incorporate only mechanisms that survive those tests.
The next architecture claim must explain both E2's need for protection and E4's
need for sharing, rather than optimizing one of those cases and ignoring the
other. No current experiment licenses completion of the overall goal.

The user's updated objective explicitly requires recursive self-improvement and
persistent memory. The manual E1–E8 research sequence is not itself evidence of an
autonomous self-improving architecture. A learned, persistent update/memory
controller and a test of its own improvement process are now explicit missing
requirements. Student-weight learning, feature renewal with a fixed rule and
better researcher-chosen hyperparameters do not satisfy them.
