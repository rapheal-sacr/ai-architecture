# E30 result: integrated feedback does not establish reliable improvement

Read FALSIFICATION.md first. All eight registered cases completed from frozen
source `028bffe`. All96 mission checkpoints pass independent trajectory, memory,
replay, RNG-progression and work reconstruction. Reexecuting all192 online
updates reproduces every adaptive mission adapter, optimizer, replay, learning
RNG, loss and update count exactly. No scored source or settings were changed.

## Registered outcomes

Each seed has four initial short missions, four larger-world missions, two
unchanged return missions, one relocation and one other-object mission. Initial
acquisition and larger-world competence each require3/4 successes. Return
outcomes cannot establish retention when initial acquisition failed.

| Seed | Arm | Initial /4 | Larger /4 | Return /2 | Relocation /1 | Other /1 | Total /12 | Actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 30101 | observed_planner | 4 | 4 | 2 | 1 | 1 | 12 | 60 |
| 30101 | frozen_full | 0 | 2 | 0 | 0 | 0 | 2 | 300 |
| 30101 | frozen_bounded | 0 | 2 | 0 | 0 | 0 | 2 | 300 |
| 30101 | adaptive_bounded | 0 | 2 | 0 | 1 | 0 | 3 | 331 |
| 30102 | observed_planner | 4 | 4 | 2 | 1 | 1 | 12 | 59 |
| 30102 | frozen_full | 3 | 0 | 1 | 0 | 0 | 4 | 323 |
| 30102 | frozen_bounded | 3 | 2 | 1 | 0 | 0 | 6 | 285 |
| 30102 | adaptive_bounded | 3 | 1 | 1 | 1 | 0 | 6 | 310 |

The observed-map planner completes24/24 deliveries in119 actions. It sees local
observations only and computes routes using its accumulated observed edges.
Neural acquisition fails on seed30101 even with full memory. All neural arms
meet initial acquisition on30102 but complete only one of two unchanged return
missions. No neural seed/arm meets the larger-world3/4 criterion. The frozen
arms also fail return missions with unchanged weights, so these failures cannot
all be called parameter forgetting.

Adaptive bounded memory gains one total success versus frozen bounded memory
(9/24 versus8/24), using641 versus585 actions. It solves both relocation missions
where the frozen controls solve neither. However, it loses one larger-world
success and does not improve unchanged returns. Two seeds and two relocations
are insufficient to establish reliable adaptation. The registered step-penalized
return does improve, from2.15 to2.59 across seeds (successes minus0.01 per action).
The protocol does not assign runtime a conversion to task utility, so these are
a cost/benefit tradeoff rather than a demonstrated net-utility loss. The observed
planner has both much higher return22.81 and much lower work. The adaptive
return improves from−1.00 to−0.31 on30101 and worsens from3.15 to2.90 on30102;
the aggregate gain must not hide this reversal.

Full and bounded frozen agents have identical initial four-mission trajectories
for each seed, verified from records. Their later observed prompts really differ
after eviction; the cap is not a cosmetic prompt setting. Bounded frozen memory
nevertheless beats full memory on the second seed and ties its success count on
the first. More stored facts alone do not repair this policy. Different later
actions and observations prevent treating these as paired static-memory queries.

## Costs and state

All values below sum both seeds. Acting/update time excludes model setup,
checkpoint serialization and separately reported audit work. This scope is
explicit; it is not complete end-to-end training or deployment cost.

| Arm | Actual actions | Inference tokens | Online update tokens | Acting + online-update seconds |
|---|---:|---:|---:|---:|
| observed_planner | 119 | 0 | 0 | 0.014 |
| frozen_full | 623 | 175,161 | 0 | 75.628 |
| frozen_bounded | 585 | 152,954 | 0 | 66.761 |
| adaptive_bounded | 641 | 170,878 | 50,798 | 123.092 |

Adaptive acting/update time is1.84 times frozen bounded time. Every neural arm
also inherits306 external teacher actions and512 bootstrap gradient updates
across the two seeds:131.51 seconds fitting,4.82 seconds loading and0.082 seconds
generating demonstrations. Tutorials are declared scaffolding, not autonomous
discovery. They were physically computed once per seed and shared across arms;
a standalone arm must still pay its inherited bootstrap. Adding common bootstrap
reduces the relative time ratio, but does not erase its cost. Operational setup
takes about5.8–5.9 seconds per neural arm across seeds. Full-loop times, including
checkpoint handling, are76.30/67.44/125.40 seconds for full/bounded/adaptive.

Each neural policy retains988,065,536 base-parameter bytes and1,081,344 adapter
bytes. Adaptive Adam state adds2,163,072 bytes and live gradient buffers add
1,081,344 bytes. Its64 replay records occupy41,973/41,030 serialized UTF-8 bytes
at the end, much more than its385/394 bytes of room records. The bounded frozen
room records use385/359 bytes versus full frozen931/866 bytes. Tensor and NumPy
RNG state are also charged in the audit. These are serialized/tensor counts,
not complete Python process memory. The base model dominates this small pilot.

Peak CUDA allocation is about1.080–1.082 GB frozen and1.622–1.634 GB adaptive.
Transient trajectories, all mission memory states, forward attention-score
elements and per-phase work are preserved. Attention elements are not FLOPs.
Model buffers, Python overhead, energy and pretraining costs are not completely
measured. Audit-only trajectory logs, source examples and checkpoints remain
inaccessible to policies; they are not a free retrieval archive.

CPU reconstruction takes2.04 seconds. Duplicate online-update replay takes
48.60 seconds plus audit model loading/hash checks. This is experimental audit
work, not useful policy adaptation, and is kept separate. The exact replay
verifies online updates, not bootstrap fitting or every alternate-action logit.

## What the falsification locates

Post hoc local-action diagnostics use only the observations already available
at each real step. They do not send labels or counterfactual outcomes to agents.
On seed30101, full frozen language picks a wrong object53 times, discards the
requested object8 times, and misses14 of16 immediate moves home while carrying
the target. The bounded frozen counts are52,8 and14/16. Adaptive counts are19,13
and17/21. These are local action-use failures even when the relevant distinction
is visible; unavailable long-term history is not a sufficient explanation.
They are descriptive diagnostics added during audit, not registered performance
criteria, and overlapping counts must not be summed as independent errors.

Root collapse remains useful:

- Identifying evidence: the observed-map control establishes that this task can
  be solved without hidden-world access. Neural action paths differ, so it does
  not prove that every neural step has seen every fact needed for a full route.
- Retained distinctions: real LRU evictions occur, but short acquisition already
  fails on one seed without relevant evictions. This assay does not establish
  learned compression, rare-query preservation or indefinite memory.
- Useful computation/access: observable local decision errors and full-memory
  failure show that using present facts is a live limit. Fixed online updates
  trade relocation gains for longer routes and lost larger-world successes.

The operational binding-constraint hypothesis remains amortized recovery cost
relative to useful context lifetime. This pilot does not measure a universal
improvement-rate bound. Where initial competence was never acquired, a finite
retention or reacquisition time cannot honestly be assigned.

## Settled here, open, and next

Locally settled: this implementation executes a persistent language/memory/
actual-feedback loop, preserves base identity and restores state exactly. Its
fixed learning procedure is not a sufficient route to reliable efficient
continual competence on these worlds. The observed-map baseline is much better.

Open: whether the neural limitation is primarily symbol/address binding,
insufficient tutorial fitting, exploration, credit assignment or destructive
adaptation. These mechanisms remain confounded. Lower aggregate tutorial loss
has not been established by an independent fit audit; do not assume generalization
failed after successful training. Subsequent diagnostics should use frozen
checkpoints to test known tutorial decisions and renaming sensitivity before
adding new learning mechanisms. Preserve complete costs and fresh evaluation
seeds for any changed design. No E31 protocol is frozen or run yet.

Unmeasured: learned memory compression with preserved rare knowledge, broad
reasoning, repeated useful procedure self-improvement, architectural novelty,
very long-horizon autonomous efficiency and a universal improvement rate. E30
does not implement the complete ordered latent workspace, learned world model,
compression/admission process or procedure-trial loop. Conventional language
LoRA, LRU memory and fixed PPO-style replay are not a novelty claim. The full
goal remains active and unachieved.

## Later provenance correction from the E31 audit

The two learning/action seeds use different operational worlds, but their
tutorial world seeds were constructed as seed+700000+i. They therefore share
seven of eight tutorial worlds. Do not treat the two seeds as independent
training corpora. This changes no recorded outcome, but weakens replication
claims. Future corpus generation should use disjoint hierarchical seed streams.
