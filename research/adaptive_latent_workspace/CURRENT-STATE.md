# Research state — goal still active

Read `FALSIFICATION.md` first. No result establishes a scientifically novel,
general continually improving agent or efficient very long-horizon reasoning.
E1–E24 are complete; no experiment is currently running.

## What is locally established

- E2/E5 support protected temporary adaptation for reuse of four conflicting
  hidden functions. The overwrite ablation loses that benefit, and wider replay
  controls do not eliminate it. Active-first routing saves substantial scoring
  work on the tested family.
- E3/E4 reject general transfer of that allocation rule: drift wastes work,
  capacity is exhausted, and a fixed-function input-shift problem favors sharing.
- E13 extends the exact E6 prefix to five million observations. Guarded sharing
  eventually improves mean cumulative error by 21%, but uses 2.42 times the
  forward examples, 1.64 times the time and 1.45 times stored tensor bytes.
  Efficient compression and rare retention remain unproved.
- E7 feature renewal improves whole-stream prediction at extra cost. E8 shows
  that it does not prevent closed-loop forgetting during failed adaptation.
- E10 repairs one PPO checkpoint setup by removing entropy pressure. E12 rejects
  that as a sufficient general repair: both fresh zero-entropy seeds collapse
  after harder-task training; only one reacquires the original task. Late
  collapse can happen within a stationary stage. Its exact cause is unisolated.
- E14 implements a learned neural dynamics/reward model with bounded replay and
  planning. Its protection/merge mechanism never activates. E15/E16 show that
  high-gravity search stays poor even with accurate dynamics and simple temporal
  proposal structure. Model accuracy and search are separate limits.
- E9/E11 implement bounded self-application on reset students, with mixed results
  against conventional extra meta-training. E17 carries those frozen procedures
  into persistent students: accuracy gains are modest, execution costs about
  2.27 times fixed Adam, and self-application comparisons remain mixed.
- E18 actually changes the procedure online while student weights, optimizer,
  context and bounded experience persist. Self-application beats equal-trial
  conventional meta-updates in all four input-shift cases but only one of four
  conflicting-function cases. It also loses to freezing in three conflicting
  cases. Accepted updates do not establish reliable future improvement.

These are scoped results on the recorded seeds and tasks. Read individual
`E*-RESULT.md` reports for controls, full costs and failures. E12's interim record
is historical and is superseded by its complete report.

## The current candidate and its boundary

`DESIGN-REVISION.md` proposes an ordered recurrent workspace, protected adaptive
state, reusable modules, query-sensitive episodic compression, learned dynamics
and persistent procedure state. Only subsets are implemented and tested.
`ONLINE-PROCEDURE-DESIGN.md` gives the implemented E18 procedure loop: propose
from already-observed update traces, test a temporary branch on subsequent
observations, retain incumbent operational predictions until admission, and
charge every trial. The comparator uses ordinary meta-updates at the same trial
budget. Neither mechanism invents arbitrary architectures or changes its own
evaluator.

E18's online self-application takes about 1.73 times frozen-procedure time and
3.82 times fixed-Adam time, plus inherited bootstrap. Its input-shift gain is
real within this experiment but is not cost-matched efficiency. Both branches
receive full supervised feedback; this does not provide their counterfactual
action outcomes in a closed-loop environment. Memory policy, model family and
admission rule remain fixed.

E19 now holds student/optimizer/context/replay/RNG state fixed and changes only
the procedure. The updater matters, but its benefit depends on world and horizon:
input-shift keeping wins 4/4 original-world continuations and 0/4 new-world
transfers on whole-stream error. New-world late error is better after a worse
start, without repaying that deficit at this horizon. This supports a changed
learning tradeoff, not uniform general improvement.

E20 finds a much larger conflicting-function gain from privileged context than
from width: 82.6% versus 9.5% less mean error. Correct context raises recovery
from 234/407 to 396/407 segments; width alone reaches 260/407. The oracle changes
information and encoding and also repairs replay assignment, so the result
locates a limit in the inference/assignment package rather than proving an
irreducible identification bound.

E21 completed those non-oracle changes. Faster decay plus post-outcome assignment
reduces conflicting-function error 63.5% and raises recovery from 195/408 to
385/408 at the same model-example count and stored learner state, with about
4% extra runtime. Input-shift error rises 21.8%. E22 now implements the learned
causal gate described in `INFERENCE-REVISION.md`. It changes online but loses to
fast/prior in every conflicting, independent and alternating world, and to
slow/prior in every input-shift world. Coupled changes give a small scoped gain.
At 1.95 times fast/prior runtime, it is rejected as a sufficient repair.

E23 implements regularized coefficient descriptors with the same student and
post-outcome assignment. They improve 38/40 matched raw-descriptor pairs and
reduce fast independent/alternating error 13.2%/9.9%. Runtime rises 9.0%, storage
396 tensor bytes, and each run adds 4,096 nine-by-nine solves. Counterexamples
remain; coefficients of a nonlinear function are still distribution-weighted
projections. This is established regression machinery and a stronger fixed
baseline, not autonomous self-improvement or a novelty claim.

E24 completed the actual E6 merge audit. All arms acquire rare competence.
Guarded sharing accepts ten merges from competent sources; none crosses the
frozen material rare-loss threshold on independent probes. That attempted
falsification did not occur. Its final rare error still rises 4.21 times from
acquisition and narrowly fails the 0.05 threshold in three of four worlds, while
beating both controls on mean final rare error. This is neither universal
preservation nor evidence that an accepted merge caused each later loss.

Three of four non-merging worlds retain a rare-competent persistent module while
the operational active prediction is not competent. The next causal intervention
should test input-dependent access to that retained state using only actual
stored observations to train a router. The privileged best-module audit is not
an existing agent capability. No wider memory bank is justified before this
selection gap is addressed.

## Binding constraint and unresolved measurement

The operational candidate is **recovery cost relative to useful context life**,
charging identification, adaptation, replay, routing, search, consolidation and
procedure trials. E17 supplies a censored assay: self-applied procedures recover
in 115/228 conflicting segments versus 227/228 input-shift segments. About
82% versus 11% of segment life is consumed before recovery or censoring. E18
further shows that mean error and recovery frequency can favor different arms.

No universal single numerical rate bound has been measured. Identifying evidence,
retained distinctions and useful computation remain separable roots. A universal
claim without specifying tasks, queries, resources and competence would hide
that missing measurement. `CONSTRAINTS.md` gives the limited analytical bounds.

## What must change before confidence

1. Separate context-identification delay from representational capacity and
   destructive sharing. E17/E18's conflicting-function failures cannot be fixed
   merely by citing their easier input-shift gains.
2. Demonstrate memory compression with lower total cost and preserved rare/old
   queries outside the compression and admission samples.
3. Extend the censored recovery assay to closed-loop tasks and explicit resource
   budgets; supervised stream length is not autonomous planning horizon.
4. Show repeated procedure improvement against ordinary meta-training without
   losing retention, and repay bootstrap, rejected trials and duplicate work.
5. Establish general reasoning and a defensible novelty distinction against
   existing modular, recurrent, learned-optimizer and policy-proposal systems.

The broad goal remains unachieved. Current evidence supports narrow mechanisms
and provides concrete counterexamples to stronger interpretations.
