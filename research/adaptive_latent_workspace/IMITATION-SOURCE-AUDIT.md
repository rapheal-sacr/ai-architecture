# Imitation source audit: guidance is not an executed transition

Read FALSIFICATION.md first. This is a source audit and a small executable
counterexample, not an upstream benchmark reproduction or an adopted repair.
Clone: WD repos/imitation, HumanCompatibleAI/imitation, clean commit
`e5ef18806c449ca47153b494a02471c5e2ae3a14`. Files and hashes are preserved in
results/imitation_source_audit.json; audit_imitation_source.py reproduces the probe.
No upstream policy training was run and no new dependency environment was installed.

## What the collector actually stores

`src/imitation/algorithms/dagger.py` InteractiveTrajectoryCollector accepts an
expert action at each step. Its beta mask chooses whether the environment takes
that expert action or the learner's action. It stores `_last_user_actions`, the
expert labels, while recording the observation/reward produced by the action
actually executed. This is the intended behavior-cloning supervision interface.
It is not a bug claim against the library.

The probe executes the unmodified upstream step_async/step_wait method bodies
with a deterministic one-state additive environment and minimal recorder. With
beta0, the expert recommends2, the learner executes1, and the next observation is1.
The saved action label is2. With beta1, executed action and label both equal2,
and next observation is2. Treating the beta0 label as the executed action gives
a false action/transition pairing. No neural model or source policy was trained.

For the proposed integrated architecture, episodic records used across imitation,
world-model fitting and procedure evaluation must retain separate fields for
observed state, executed action, observed consequence, advisory target and its
provenance. A guidance label must not silently overwrite an executed action.
Compression must preserve this semantic distinction. This attacks the retained-
distinctions root; it does not repair the E31 neural fitting or binding failures.
Current E30/E32 operational logs already record actual actions; this is a
constraint on future aggregation, not evidence that those logs are corrupted.

## The stronger control has real costs

SimpleDAggerTrainer collects states under a mixture of expert/learner actions,
then trains against expert labels on those visited states. Expert labels are
queried and retained even when learner actions execute. This may expose recovery
states absent from successful expert-only tutorials, but a measured improvement
would rely on additional guidance. It is not autonomous discovery from reward.
A fair use must charge actual environment steps, expert calls, learner calls,
retained data, repeated fitting and any evaluation rollouts.

`_load_all_demos` retains all prior demonstration trajectories and flattens them.
`extend_and_update` defaults to four behavior-cloning epochs if no epoch/batch
budget is supplied. With m new transitions per round, complete batches and R
rounds, this schedule presents4m(1+...+R) training examples, before other work.
That conditional count is not the requested environment timestep count. The
implementation also rounds rollouts to finish minimum steps/episodes, so its
requested total_timesteps is a lower bound on actual interaction. Full retained
history is not bounded memory compression.

The BC implementation already supports gradient accumulation and defaults to
an entropy term. E32's accumulation controls are conventional; E32 does not
reproduce this package's whole BC objective or DAgger procedure. Any future
aggregation comparison needs its own frozen protocol and an equal-label/work
control. E32 continues unchanged while this source audit is recorded.
