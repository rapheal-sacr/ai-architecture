# E30: first integrated delivery pilot

Read FALSIFICATION.md first. This is a bounded falsification of language-guided
closed-loop action with persistent observed state and parameter adaptation.
It is not the full proposed architecture and supplies no novelty or RSI result.
The exact settings and criteria are in protocol_e30.json. Freeze this source
before either scored seed or its bootstrap is executed; preserve every failure.

## Roots and comparisons

The binding-constraint hypothesis remains amortized recovery cost relative to
useful context life; no universal rate bound has been established. This assay
measures actual actions, task returns, learning work and retained state. Two
worlds and twelve missions cannot establish very long-horizon efficiency.

- Identifying evidence: local observations and action consequences only. Shelves
  persist, and one relocation occurs without a notification to the actor. The
  environment cannot provide counterfactual outcomes for actions not taken.
- Retained distinctions: compare full latest-observation state with four-room
  LRU across worlds. Count all replay prompts too. This deliberately simple
  truncation is not learned compression and has no preservation guarantee.
- Useful computation/access: an observed-map BFS planner is the strong control;
  neural arms must interpret the same observed facts and legal choices. Online
  adapter updates use real feedback and may damage prior competence. Acquisition
  must precede a retention interpretation.

All neural arms inherit the same per-seed tutorial-trained adapter. The external
teacher uses only its observed map, and all teacher actions plus 256 supervised
updates are charged. The adaptive arm retains bootstrap Adam moments; it makes
eight clipped, fixed-procedure updates after each actual mission, with a bounded
64-record reservoir. None of these updates changes the learning procedure.
The legal-action interface supplies validity; that is not learned reasoning.

## Preflight and design corrections before scoring

The first environment check exposed stale-shelf search in the planner. When
all rooms had been observed it lacked a systematic reinspection fallback. The
corrected planner revisits the oldest observed room using known edges. It then
completed all54 missions on the three development seeds at sizes4/8/16. This is
a corrected baseline, not a candidate improvement. Hidden-shelf perturbation
leaves the same observation, prompt and planner choice; capacity really evicts.

The initial planned three-room tutorial worlds were fully connected and never
taught a two-edge route. Before freezing, the scored bootstrap was changed from
two three-room worlds/96 updates to eight four-room worlds/256 updates. A CPU
development audit completed32 tutorial missions, including two requested routes
of at least two edges. Extra teaching and fitting are charged. No scored seed
was used to choose these settings.

The reduced neural preflight uses seed1711, one three-room tutorial, four
supervised updates, two operational missions and two online updates per mission.
The planner completes2/2; all three neural arms complete0/2. Those failures remain
in the record. Base identity, exact actor-state/logit restoration, actual online
adapter changes, paired worlds and identical full/bounded trajectories when the
cap removes no relevant facts pass. These are functional checks, not competence.
After that preflight, only criteria prose was clarified to distinguish missions
8–9 unchanged returns, mission10 relocation and mission11 the other object.
Execution source hashes remain exactly those checked by the preflight.

## Frozen reporting boundary

Two fresh seeds, four arms, twelve missions each. Report acquisition0–3,
larger-world4–7, unchanged return8–9, relocation10 and other-object11 separately.
Acquisition and larger-world competence each require at least3/4 successes per
seed/arm. Report all mission actions and returns even when acquisition fails.
No best checkpoint, test-based stopping, oracle access or outcome-selected retry.

Count inherited teaching/training separately from operational inference and
updates. Attention-score elements measure forward attention matrix size, not
FLOPs; gradients, base, optimizer, RNG, replay and room serialization all matter.
CPU Python object overhead, energy, pretraining cost and a general improvement
rate are not measured. Audit-only logs/checkpoints and transient trajectory
copies must be distinguished from persistent policy-accessible memory. Saved
checkpoints support audit; the policy cannot query them. Future results require
independent action/memory/replay reconstruction before interpretation.
