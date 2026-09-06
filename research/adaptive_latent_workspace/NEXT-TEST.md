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
