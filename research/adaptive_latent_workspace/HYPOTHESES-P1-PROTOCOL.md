# Attachment hypothesis probes, P1 — frozen before scored execution

Input: `user-hypotheses-2026-09-07/pasted-text.txt` in the run directory,
SHA256 `f6ed22b03f06cb69e372c4c28d988ce33b8f8f362b4aaf17495800b22de3a46e`.
These are targeted falsifiers, not a full implementation of the proposed agent.
Do not interpret mechanism probes as trained-model replications.

## P1a: surprise as an acquisition controller (H4, related H1/H12)

The learner chooses which of 32 Bernoulli sources to observe, one paid observation
per step. It must predict all sources. Compare uniform acquisition, exponentially
averaged realized surprise, predictive entropy, and expected one-observation
information gain under a Beta posterior. All receive two initial observations per
source and the same 10% exploration. No policy sees true source probabilities,
source type, change announcement, evaluation losses, or future labels.

Use 24 fixed seeds 37001–37024, 4,096 paid observations inclusive of warmup,
and three conditions: all deterministic sources; half deterministic and half
irreducibly fair; the same mixed case with all deterministic bits flipped at
step 2,048. Randomly permute source identities independently per seed. These are
constructed interventions, not samples from a natural task population.

Cross every condition with lifetime counts and a last-64-observations-per-source
estimator. Each uses Beta(1,1) pseudocounts. The rolling estimator forgets selected
source observations by age within that source, not wall time. Information gain
uses this finite-data posterior; under change it is a misspecified stationary
belief, not a calibrated nonstationary oracle. Surprise is updated with rate .1.
Predictive entropy and information gain are recomputed for the selected source.
All choose uniformly among exact maximizers, then apply the common exploration.

Primary measurement: pre-observation mean squared error of predicted probabilities
against current true probabilities, averaged over every source and every paid
step. This equals excess expected Brier loss above irreducible noise. Also retain
endpoint and post-change losses, deterministic-source loss, observation allocation,
policy runtime, operation counts, persistent array bytes, and individual seeds.
Evaluation uses private probabilities only outside policy/update code. Common
random tapes indexed by time, seed, source couple comparisons reproducibly.

Kill the sufficient claim "more surprise identifies more useful learning" if
surprise systematically allocates to irreducible noise and has worse prediction
loss than a control at the same observation budget. Do not promote information
gain to a universal repair if change exposes overconfidence or slow recovery.
No learned goal generator, neural optimizer, memory compression, or radical
environment transfer is tested here. Equal observations do not mean equal CPU
cost; report both. No hyperparameter search on scored seeds.

## P1b: execute unchanged upstream source in bounded fixtures (H4/H8)

CTM commit `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`: extract unchanged
`ContinuousThoughtMachine.forward`, `compute_certainty`, entropy helper and parity
thinking-time analysis with AST. Supply deterministic counted fixture modules,
confident predictions on the first tick, and count actual module execution at
1, 4, 16 ticks. Distinguish retrospective answer selection from runtime halting.
Test confidently wrong logits separately. This is not a trained CTM evaluation.

Metis commit `22f7aabc8d3e9c39fcea676b11a10007bc8b3748`: import unchanged
standalone HyperMemory source. Probe its shared gated delta update with explicit
chosen gates/keys/values and an independent matrix calculation. Test orthogonal
old information under decay, and order sensitivity lost by the batched parallel
approximation. Gates are chosen fixtures, not observed trained settings. The
default class calls this mixin but full selection/read/backbone are not run.

## P1c: representation claims (H1–H3)

Check that a policy over explicit desirability vectors and success probabilities
has the same choices as a scalar expected utility conditioned on those same
quantities. Contrast with a stale scalar that omits changed preference inputs,
clearly identifying that weaker control. This tests equivalence, not the relative
sample efficiency of learning these representations. Goal generation changes
the experience policy; its meta-objective still defines what counts as improvement.

Future stages must train goal/acquisition and consolidation policies, test held-out
worlds and actual action consequences, and charge outer training and all trials.
Passing these probes cannot satisfy the broad autonomous-improvement goal.
