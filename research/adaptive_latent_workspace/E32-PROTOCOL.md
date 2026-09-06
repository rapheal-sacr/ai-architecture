# E32: conventional optimization controls before architectural attribution

Read FALSIFICATION.md first. E31 shows failed tutorial fitting and an example of
training damaging immediate-home decisions. More memory or a procedure learner
cannot be credited for fixing an execution baseline that was never competent.
This is a prerequisite experiment, not a substitute for the complete objective.
Freeze this source and protocol_e32.json before scored data generation/fitting.

Three learning seeds each receive disjoint tutorial and evaluation worlds.
Domain-separated hashed seeds replace overlapping consecutive ranges. Assert
all30 world seed specifications and visible world labels are distinct before
training. Within each learning seed, all four arms share exactly the same
initial adapter, examples and ordered index sequence. This does not make
repeated states or the shared task generator independent observations.

The factorial controls are learning rates0.001/0.0001 and accumulation of1/8
examples per Adam step. Both256 and1024 gradient-example milestones are scored;
all arms continue to1024 regardless of results. Batch8 evaluates each example
separately without padding, scales by1/8, accumulates at fixed weights, clips
once and updates. It makes eight times fewer optimizer steps. This changes the
update schedule and cannot uniquely isolate gradient noise. Every gradient
example, backward pass, Adam step and wall-clock cost is reported.

All teaching uses an external observed-map planner on eight four-room worlds.
It is charged scaffolding, not autonomous acquisition of the teaching rule.
At both milestones, query all tutorial states with tie-aware references and
then run twelve actual missions on fresh four/eight-room worlds. Full observed
memory persists across those missions, removing truncation as a fit confound.
No evaluation feedback enters fitting. Save and restore exact actor, optimizer,
RNG and work state around these separately charged laboratory evaluations.
No policy sees free counterfactual outcomes or hidden world state. The invariant
observed-map control runs once per seed and is shared across milestone reports.

Tutorial mastery requires95% overall rule fidelity and80% in categories with
at least10 examples; supplied legal choices and tied routes do not establish
general reasoning. Initial and larger-world acquisition each require3/4 actual
deliveries. Only acquired initial tasks support retention interpretations on
unchanged return missions8–9. Relocation10 and other-object11 remain separate.
No best arm/checkpoint selection is permitted as a replacement for full reporting.

## Preflight evidence

The new batch1 path exactly replays all256 E30 bootstrap updates for historical
seed30101: losses, gradients, complete adapter/optimizer/RNG/work state match.
This reuses a known historical baseline for compatibility only; it is not an
E32 scored result. Four-example accumulation agrees with an independent joint
mean-loss backward calculation: maximum clipped-gradient difference7.45e-9,
maximum post-Adam parameter difference3.61e-7. The tolerance is documented.

A full-size CPU development check uses1911/1912/1913, verifies disjoint worlds
and checks reference sets independently. A separate neural functional preflight
uses1911, one tutorial world,8/16 example milestones and two short missions.
All four arms/eight snapshots preserve base identity, match example work and
restore complete training state after evaluation. Mission failures are retained;
these checks do not establish competence. No scored seed informed settings.

## Architecture boundary

This attacks useful learning/computation and interference, holding evidence and
memory representation fixed within comparisons. It does not implement learned
compression, a new world model, or an improved learning procedure chosen by the
system itself. Ordinary human-set optimizer controls are not RSI or novelty.
Successful fitting would justify a stronger baseline for the full architecture;
it would not establish efficient long-horizon continual learning. The operational
binding-constraint hypothesis remains recovery cost relative to useful context
life, with no universal improvement-rate bound measured.
