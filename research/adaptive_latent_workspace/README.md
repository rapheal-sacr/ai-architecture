# Adaptive Latent Workspace — active research

Status: **design under investigation; continuous-learning and long-horizon
efficiency goal not yet achieved**.

Current boundary: E2 supports reuse under abrupt recurring functions, but E3
exposes capacity and drift costs and E4 rejects efficient transfer to an
independent changing-input benchmark. The current admission rule is not a
general solution. Read `E4-RESULT.md` before extrapolating the earlier gains.

This is a new model architecture research line. It does not extend WAM's ledger
stack or the finite-hypothesis witness compiler. Historical files remain in the
repository as prior evidence, not implementation dependencies. Read
`FALSIFICATION.md` before interpreting the design.

## Proposed organizing principle

Learn *which dynamics are currently operating*, adapt a temporary neural state
quickly, and retain reusable transformations in a bounded library of slow modules.
Use the same learned dynamics for prediction and planning. Memory is a resource
allocation problem across recent observations, temporary weights, and persistent
modules, rather than one authoritative database from which all competence is
compiled.

The proposed system has four interacting mechanisms:

1. A recurrent working state infers context from observations, actions and their
   outcomes. It receives no oracle task or regime IDs.
2. A temporary adaptive model learns from the current trajectory. Its updates
   can be reset or routed elsewhere without overwriting established modules.
3. Persistent modules retain reusable dynamics and procedures. Context inference
   routes to them; novel dynamics can recruit capacity. A learned router should
   eventually replace expensive evaluation of every module.
4. A bounded episodic buffer retains observations poorly explained by the model.
   Consolidation must demonstrate lower total cost and maintained predictive and
   task quality before replacing observations with weights or summaries.

Planning uses learned multi-step predictions, not privileged access to the
environment transition function. Model disagreement and prediction error are
diagnostics; neither is assumed to be calibrated novelty or useful information.

## Sources of the mechanisms

- TTT-E2E: align training with the process of adapting at test time; retain a
  static path alongside changing contextual weights.
- Titans: associative neural state, adaptive updates and multiple memory scales.
- Macaron: independently updated reusable modules and context isolation.
- Recuris and NeSyFS: explicit working-state maintenance and outcome-grounded
  planning, while distinguishing diagnosis from causal identification.
- Continual Backprop: measure loss of plasticity and test feature renewal rather
  than assuming ordinary gradient descent remains indefinitely adaptable.
- AdaMM and OaK: compose learned representations with executable operations where
  appropriate; compare against strong explicit-memory/tool baselines.
- RecHarness: count all trial resource costs and keep search feedback distinct
  from final evaluation.

The source papers and cloned source revisions are catalogued in the earlier
research dossier on the WD volume. Mechanism inspiration is not evidence that
their combination works.

## Roots and candidate binding constraint

The unresolved roots are context identification, stable adaptation/retention,
and useful model-based computation. Cheap storage cannot repair a wrong context
assignment; preserving every past model cannot repair a wrong prediction model;
more planning can compound model error.

For changing tasks, the immediate quantity to measure is **time and observations
required to recover useful performance relative to how long that context remains
relevant**. Long-horizon efficiency also requires retained skills to lower this
cost on recurrence after charging routing, inference, updates, replay, growth and
consolidation. No universal binding resource is declared before measurement.

## Evidence required before confidence

- Actual parameter learning from new experience with no pre-enumerated correct
  procedure, no hidden task IDs, and no future targets in inputs.
- Retention and reacquisition across long streams, unseen contexts, returning
  contexts, drift, noise, rare contexts and exhausted capacity.
- Strong baselines: current-only learning, ordinary replay, context-conditioned
  replay, modular continual learning, explicit memory/tools, and appropriately
  identified oracle upper bounds.
- Component ablations and matched resource comparisons. Fewer examples alone
  cannot establish efficiency.
- Long-horizon closed-loop tasks where the model's errors change future
  observations and task success. Prediction-only tests are necessary foundations,
  not completion of this requirement.
- At least one independently specified external benchmark, plus separately
  frozen evaluation after exploratory tuning.

## Experiment ladder

E1 tests hidden-context discovery, actual neural learning and return-task reuse
under abrupt changes. It is a small foundational experiment, not an end-state
substitute. Subsequent experiments must test temporary/slow weight separation,
bounded memory, router cost, feature renewal, and model-based planning on external
long-horizon environments. The research goal remains active until the broader
requirements have adequate evidence.

The code lives under this directory. Large runtimes, checkpoints and detailed
traces live on WD outside tracked source. Results and methodological corrections
are committed here, including negative outcomes.
