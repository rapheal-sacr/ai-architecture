# Fixed action planner and world: preflight only

`action_navigation.py` defines unknown directed action ports, external goals and
unannounced map changes. Every action separately forms a Hamiltonian cycle in
these fixtures; this is a strong artificial structure, not a general environment.
An observed-transition table supplies a strong observation-only planner control.

`action_planning.py` predicts every state/action destination distribution from
the current real E2E state. Independent hypothetical queries are batched with
vmap and their cache outputs discarded. A fixed Bellman calculation plans goal
arrival within a horizon. It does not simulate future learning or infer a full
belief over hidden maps. Public enumerated states/actions bound its scope.

Preflight82650 completed in2.366s with one CPU thread, float64:

- 576 deterministic values agree exactly with exhaustive action-string search;
  stochastic values agree exactly with explicit conditional branch recursion.
- 30 fully observed-map actions are shortest-path first actions.
- World and bounded table serialize together and resume38 real steps exactly,
  including hidden changes, rewards referring to the prior goal, and new goals.
- Twelve batched hypothetical queries agree with serial execution to1.89e-15;
  gradients through prior real updates agree to1.38e-15.
- A full outer objective includes one real fast update followed by planning.
  Directional finite differences at1e-4/1e-5 agree to1.64e-9/5.64e-12.
- Planning leaves real adaptive state and KV exactly unchanged. The checked
  six-state/two-action fixture charges12 queries,24 tokens and864 attention-score
  elements, plus separate Bellman transition work; it performs no inner updates
  from imagined outcomes.

All-state model queries cost S×A forward queries; each produces S destination
scores. Bellman work is H×S²×A. Batching does not remove those costs. This fixed
planner is a control/diagnostic and cannot be credited as learned reasoning.
No training, acquired action competence, autonomous goal invention, broad
transfer, adaptive runtime allocation or recursive improvement is measured here.
