# Current continuation: online access while models and memory change

E25 completed: observed-memory routing repairs much of the frozen non-merging
access gap, but nearest anchors are strong and neural fitting has not amortized.
One-expert limitations and rare-query regressions remain. Read `E25-RESULT.md`.

The next integration should continue each E24 source with active selection,
periodically refreshed nearest-anchor labels, and an incrementally trained
input router. Keep the underlying E6 learning policy identical across arms so
expert/memory/RNG trajectories can be checked exactly. Predictions change;
observed outcomes train the router only afterwards. Charge all target-building,
query routing, bootstrap and incremental fitting work. Stable expert slots and
explicit validity masks must handle temporary models and merges. Do not train
on audit outcomes or region flags.

E24 checkpoint files do not retain the global torch reservoir RNG. Reconstruct
it by replaying the frozen E24 prefix with the same seed and verify every prefix
error plus model/optimizer/anchor/counter state against its saved checkpoint
before continuation. Charge this reconstruction separately. Verify a longer
world preserves the whole old prefix. The online continuation protocol must be
frozen before scoring; E26 is not yet implemented or registered.

After this causal integration, move to the missing ordered-reasoning component.
Further supervised routing variants cannot establish general reasoning,
self-improvement or autonomous task efficiency. No component pass is a substitute
for an integrated architecture facing those failures.

The original E19 design is retained below as history.

# Next falsification: procedure improvement or selected student trajectory?

E18 is complete. All final learner states are on WD with verified hashes. The
online package helps input shift, but adopting a candidate changes both its
procedure and its adapted student. Do not attribute the package gain solely to
an improved learning algorithm.

The next bounded causal test should fork each completed self-applied learner
state into two identical students, including optimizer moments, context and
replay. One keeps the final procedure; one restores its original E9 bootstrap
procedure. Freeze both procedures and charge all continuation work. Both students
must consume exactly the same fresh observations without resets or oracle IDs.
The final student state, not the final procedure's independent training run, is
the paired starting point.

Use both a continuation of the original world and a new-world transfer stream.
For the original-world extension, first verify exact equality of every generated
prefix observation, target and clean target; explicitly specify how the last
truncated latent segment continues. Do not silently change the prefix when
requesting a larger dataset. E13 already demonstrates why that check matters.
For new worlds, retain old student/memory state and use newly frozen seeds.

Register the continuation length, seeds, recovery rule and work accounting
before execution. Report all original cases, even where E18 self-application
already failed. If reverting performs as well or better, selected student
trajectory or extra training explains the claimed benefit more plausibly than
persistent procedure improvement. No one counterfactual alone proves causality
outside these full-feedback streams.

After that attribution test, separate identification delay from interference
and representational capacity in the conflicting-function failure. More
controller rounds, capacity or context-memory claims are not justified merely
because the easier input-shift case improved. Semantic compression and general
closed-loop reasoning remain independent unmet requirements.
