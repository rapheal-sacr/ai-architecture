# Next integration: persistent language-guided action and learning

Read `FALSIFICATION.md` and `GOAL-EVIDENCE.md` first. E29 repairs some depth
sensitivity but fails acquisition and N64 transfer. More graph-depth variants
are not an adequate route to the user's full objective. The pinned language
core and low-rank updates now execute locally; no integrated agent is built or
validated yet.

The next assay should put language interpretation, explicit retained evidence,
trainable state and action-dependent outcomes in one loop. Use a generated
persistent delivery world: local observations describe rooms, exits and objects;
missions require finding and delivering an object; subsequent missions revisit
the same world, with some unannounced changes. Fresh aliases and layouts prevent
pretraining from containing the specific answers. Initial short missions must
establish that an arm can act before failures on longer missions are called
memory failures. This is a bounded laboratory task, not the goal's definition
of general intelligence.

Required observation boundary: the agent sees only local observations, requested
missions, its chosen actions and resulting feedback. Hidden maps, object state,
shortest routes and evaluator probes do not enter retrieval, adaptation or
stopping decisions. A read-only exact planner using the agent's observed map is
a strong explicit-memory baseline; a full-state oracle, if measured, remains a
separately labeled diagnostic. Legal action encoding must be the same across
arms and must not reveal hidden solutions.

Required retained state: preserve the base model, adapters, optimizer, episode
memory and any procedure state across missions. Compare a strong full-event-log
or explicit-state baseline with the bounded candidate. Distinguish storage from
which records are exposed to the model. Never call a shorter prompt compression
if the complete history remains available in an uncharged archive. Track memory
changes, old/rare queries, reacquisition and action mistakes separately.

Required learning boundary: parameter updates use already observed outcomes.
Any bootstrap teacher or demonstration set must be declared and charged; it
cannot silently supply hidden route labels online. Learning a fixed adapter
procedure does not establish recursive self-improvement. Subsequent procedure
changes must be persistent, tested on actual later behavior against an equal-
work conventional update control, and charged for every accepted/rejected trial.
Do not pretend a real world can be freely forked for counterfactual actions.

Required reporting: acquisition before retention, held-out worlds and longer
missions, recovery cost relative to useful context lifetime, raw task success,
all model-token evaluations, update work, memories/indices/optimizer bytes,
wall time and failures. Neither a pretrained language response nor a successful
45-token adapter update is evidence for these outcomes.

Implement and validate the environment/observation boundary and baselines, then
freeze a full protocol before scoring any candidate. Task sizes, seeds, training
budget, reward, legal-action representation, memory budgets and acceptance rules
remain to be specified. No E30 protocol or scored experiment is registered.
The graph trace utility and equilibrium source audit remain available for
independent diagnostics; neither is currently a new solver or chosen repair.

## E30 implementation supersedes the planning status above

The environment, observed-map baseline, language actor, persistent observation
memory and actual-feedback adapter updates now exist. E30-PROTOCOL.md and
protocol_e30.json freeze the scored pilot. Functional checks alone do not satisfy
the requirements above. Run and independently audit all registered cases next.

## E30 complete

E30-RESULT.md supersedes the pending run status. All eight cases,96 mission
checkpoints and192 online updates are audited. Integration executes but does
not meet reliable competence or efficient continual improvement. NEXT-TEST.md
now targets frozen-checkpoint acquisition/binding diagnostics.
