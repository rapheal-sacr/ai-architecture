# ai-architecture

Falsification harness for **WAM** (Write-Ahead Memory) — a memory-first AI
architecture built around one authoritative append-only ledger, with every
other tier (latent index, compiled views, weights, harness code) treated as a
derived, recompilable view of it.

The design lives in `Architecture Design/` alongside this repo. This repo is
the part that tries to break it.

**[STATUS.md](STATUS.md)** is the entry point if you want the answer: every
load-bearing claim ordered *by the design*, with its status, the constant it
rests on, and what would change it.

**[PLAN.md](PLAN.md)** is the entry point if you want the working: the claim
ledger ordered by experiment, kill criteria, and the phase schedule.

---

## Why a simulator comes before a model

Most of WAM's load-bearing claims are not claims about neural networks. They
are claims about the dynamics of a bookkeeping system — whether a threshold is
well posed, whether a gate can be steered, whether a curriculum converges,
whether a provenance cascade is transitive. Those are testable in a
discrete-event simulator with a synthetic solver, with no model in the way.

That ordering is also the only one that yields clean answers. Build L7 on a GPU
and watch it underperform, and you cannot tell whether the mechanism is wrong
or the model is small.

| Rig | What | Where |
|---|---|---|
| **A** | Ledger simulator, no model. Synthetic world with known ground truth. | Laptop, seconds–minutes |
| **B** | Small real model, MLX, ≤3B 4-bit | Laptop, 8 GB ceiling — memory cliff at seq 1024 (1.5B) |
| **C** | Rented GPU, episodic | Survivors only |

## Status

| ID | Claim | Verdict |
|---|---|---|
| [E1.1](rig_a/experiments/e1_1_spectrum_knee.py) | `#{σ_k > ε}` is a computed budget | **PARTIAL** — the literal ε-threshold formulation fails on 2 of 4 streams. Leakage is quantised in units of one committed direction, and the auto-placed ε missed by two: placing it correctly requires already knowing the rank it computes |
| [E1.1b](rig_a/experiments/e1_1b_energy_criterion.py) | the budget works under the energy/GPM criterion | **PASS** — all 4 streams, power law included. **Reverses E1.1's conclusion**: the mechanism is sound, §A just states it in the one form that does not work. *But its operating point is corrected by E1.1c* |
| [E1.1c](rig_a/experiments/e1_1c_tail_domain_exposure.py) | …does that hold *per domain*, not just on the traffic mean? | **FAIL** — partially reverses E1.1b. At ρ=0.95, **12/16 domains** exceed the interference bar while the traffic mean passes. Spearman(rate, leakage) = −0.965, monotone in rarity, so structural. Tail-safe is ρ=0.999 with free rank **17–18/128**, not ~50% |
| [E1.1d](rig_a/experiments/e1_1d_subscription_curve.py) | tail-safe free rank vs subscription ratio | **the missing curve.** At overlap 0–0.4 the feasibility boundary is **subscription 1.0×** — exactly where E1.1c and E1.2 were both measured, so the whole record sits on the cliff edge. 1.5× is already unusable (5–6 free). Overlap 0.7 pushes the boundary past 3.0× — so overlap decides whether an envelope exists at all |
| [E1.2](rig_a/experiments/e1_2_lambda_unanimity.py) | three-λ unanimity makes rank release safe | **FAIL** on both, in opposite directions. Unanimity collapses the budget from 69 free directions to **6.3** and starves **3.29×** more than the fast rule. And U2 is a *composition*, not a measurement: allocation reads only the fast estimator and the only downstream catch is traffic-weighted, so tail errors are undetectable **for any setting**. Sized at **5.2× blind** (traffic check reports 16.6% where the worst domain is at 85.8%) — the ratio is the finding, the level is contingent on 1.0× subscription and zero overlap. Tightening one breaks the other — **architectural** |
| [E1.4](rig_a/experiments/e1_4_aleatoric_magnet.py) | posterior variance is an epistemic gap detector | **FAIL** — one coin-flip region in twelve captures 51% of the practice budget, because the frontier reward `4p(1−p)` peaks exactly where a coin flip lives. Switching to epistemic variance fixes the magnet and inverts the gate to ρ = −0.99 |
| [E0.1](rig_a/experiments/e0_1_verified_recompilability.py) | **I4** — competence regenerates from provenance | **FAIL** on three criteria. Pooled over-forgetting **6.7%** while the worst region loses **79.8%** — a **12.4× blindness factor**, the largest here; a pooled-only test reports a pass. A pure ontology re-partition changes competence **21.7%**, so Root 3 reaches into Root 1. And I4 turns out to be a *relative* invariant: it checks recompile fidelity, not compile adequacy |
| [E0.2](rig_a/experiments/e0_2_transitive_unlearning.py) | the tombstone cascade reaches the weights | **PARTIAL** — the `transitive` arm was a **tautology** and is withdrawn; the 95.9% figure for `direct` is a knob. The finding it owed and did not produce: cascade invalidates ~54% of adapters per tombstone, implying a tombstone-rate ceiling |
| [E0.2b](rig_a/experiments/e0_2b_influence_and_ceiling.py) | *rebuild* — functional ground truth, third influence path | **FAIL** — `transitive` provenance recalls **0.913, not 1.0**, and no set-based closure can reach 1.0 because the residual dependency is on a retrieval that was never run. And the correct cascade touches 69% of the fleet, putting sustainable deletions **below 1/day** across much of the plausible cost space |
| [E0.2c](rig_a/experiments/e0_2c_deletion_policies.py) | *does that ceiling hold under better policies?* | **PASS** — it does not. Disabling is 28,800× cheaper than recompiling and is what makes deletion sound; batching makes throughput independent of cascade breadth. E0.2b demoted. Replaced by a real finding: **cascade breadth rises 63%→99% with card-bank duplication, so L3 admission sets L7 deletion cost** |
| [E0.2d](rig_a/experiments/e0_2d_admission_lever.py) | is admission control a lever on cascade breadth? | **FAIL** — inverts E0.2c's D3. With provenance held fixed, card cosine moves **0.498** and breadth moves **0.009**. cos ≤ 0.93 scores *content*; breadth is set by *provenance overlap*. The design has **no lever on cascade breadth** — a missing control surface, not a threshold to retune |
| [E3.1](rig_a/experiments/e3_1_transfer_ranking.py) | net-transfer ranking accumulates abstractions | **PARTIAL** — margin *grows* under noise (+0.147, artifact-free). T4's inversion turned out to be an asymmetry artifact — see E3.1b. Clean-regime numbers contaminated — see E3.1d |
| [E3.1b](rig_a/experiments/e3_1b_spillover_symmetry.py) | is T4's inversion caused by spillover, or by *asymmetry*? | **FAIL** — asymmetry artifact. **Symmetric** spillover *preserves* the margin (+0.176, +0.210); only patch-only inverts. Net transfer is a sum, so a constant added to every candidate cannot change the ranking. §B's condition is **differential** (excess ≤ ~0.03), not absolute — much weaker, and it makes the Rig B measurement a paired difference |
| [E3.1c](rig_a/experiments/e3_1c_narrowness_baseline.py) | does the transfer *matrix* beat a one-line breadth penalty? | **PARTIAL** — the **statistic** wins (by 0.093/0.031/0.088) so it earns its keep. But no arm consults accumulated history, so the **matrix** is unevidenced in either direction — and τ storage, the partition objective and the signature-ontology view all hang off it |
| [E3.1d](rig_a/experiments/e3_1d_tiebreak_audit.py) | is E3.1's compositional-only figure real? | **FAIL** — patches, comp-only skills and already-learned skills all score exactly 0.000 under net transfer, so pool emission order decides. comp_only spans **0.005–0.780** across tiebreak rules. The published 55%/72% pair is **withdrawn**; clean-regime margins are contaminated too, noisy ones are not |
| [E4.2](rig_a/experiments/e4_2_blast_radius_seal.py) | the blast-radius fixed point seals the Assay | **FAIL** twice — all four Assay tiers execute through L9-editable code; and the rule bounds *which* thresholds are editable while placing no bound on their *values*, so the reachable set is unbounded and compliance is reported across all of it |
| [E3.3](rig_a/experiments/e3_3_offdiagonal_degeneracy.py) | "maximize off-diagonal mass" is a partition objective | **FAIL** — monotone in fineness, so its argmax is total atomisation, and it ranks the planted true structure near the *bottom*. The replacement objective is validated only inside a family containing the answer |
| [E2.1](rig_a/experiments/e2_1_tier_laundering.py) | probe harvesting widens the verifiable surface | **FAIL** on 2 of 3 — 29–63% of an unfiltered "T2" suite is laundered T3 judge opinion, and the harvested slice tests each domain's easy corner. But the strict-filter **yield is high** (33–68%), contradicting this plan's own prediction |

| [E0.2e](rig_a/experiments/e0_2e_r9_breadth_coverage.py) | **R9** — provenance-aware admission has an acceptable point | **FAIL** — the curve cannot be drawn. Tightening the cap cuts cards-per-entry 9.18→1.00 (*the mechanism works*) **and** shrinks the bank, raising fleet reach. Opposed at every pool structure, and the net depends on a fleet coupling Parts I–III never specify. The expected I11 tension doesn't bind — coverage stays at 1.000 |
| [E2.3](rig_a/experiments/e2_3_staged_ladder.py) | the staged ladder makes an archive affordable | **FAIL** on both. Rank correlation drops to **0.361** where the ladder saves most, and **~100% of good rare specialists are dropped** vs ~36% of generalists. Not a weighting problem — the *unbiased control concentrates identically*, because ranking by a **mean** discards specialists structurally. Underneath: the design never says whether promotion serves mean value or coverage, so the loss isn't even well posed |
| [E5.1](rig_a/experiments/e5_1_joint_feasibility.py) | **joint feasibility** — do the coupled constraints intersect? | **9.1%** of 291,600 configurations, interior on every axis. At the design's own profile the window is empty *conditionally on cascade breadth* (needs ≤0.31 against ~0.65). Four routes out and only one isn't a hardware purchase, which promotes **R9**. The real output is that the design's uncertainty is now **four unmeasured numbers with named resolution paths** — see PLAN.md |

| [EB.1](rig_b/eb_1_recompile_wallclock.py) | **`H`** — L7 adapters compile in "hours–days" | **FAIL** — `H` is not a constant but `draw × tokens × epochs / throughput`, and the design states none of the three. Measured: at E0.1's draw cap of 300 entries, **18 minutes at 1.5B, not 8 hours**. E5.1's C3∧C4 window goes **EMPTY → OPEN** at the measured value, so its infeasibility at the design's own profile was carried entirely by the assumption |

**Two published conclusions have been withdrawn** (E1.1's "realistic traffic
leaves 6 free directions", E0.2's "transitive closure fixes it completely").
Both are recorded in `claims/claims.yaml` under `retracted:`.

**"Every failure is a specification defect" was selection, not evidence, and is
withdrawn.** The first eight experiments all asked *local mechanism* questions,
and local errors are locally repairable by construction — cheap-to-test
correlated with locally-repairable throughout.

Three claims that could fail architecturally have now run: **E3.1** passes
conditionally (on a condition the design does not state, and E3.1b shows that
condition is much weaker than first reported); **E0.2d** fails in a way no
rewording fixes — the design has no lever on cascade breadth; and **E1.2** fails
in *both* directions at once, with opposite repairs. **E0.1 has now run and fails hardest of all.** One remains untested: **E2.3**.

**The counterweight, and it is the number worth watching.** Six rig errors
against zero confirmed architectural failures. That ratio is expected when new
code tests prose that has already been through three review passes — the useful
signal is whether *detection* is improving, and it is: bugs 1–2 by inspection,
3 by contradiction between two measurements, 4 by asking whether a striking
number was a parameter I chose, 5 by an implausible direction, 6 by counting
what should have been there. Only the fourth mechanism scales.

The strategic consequence: **generator error is now the dominant error source,
and no Rig A work reduces it.** Rig B has moved ahead of the remaining 13
claims — see PLAN.md §3.

Eleven repairs proposed (R1–R11), **none adopted**. **R9 cannot yet be evaluated** — see STATUS.md. R8's admission-threshold
component is withdrawn by E0.2d; **R9 (provenance-aware admission) is promoted
by E5.1** from a missing mechanism to the one that decides feasibility.

One new invariant proposed — **I11 · Compile adequacy** — because every finding
here has been rare-region blindness and no invariant asserts that the training
draw covers what the ledger supports. A system that bounds its costs by never
learning the tail satisfies I2 and I4 completely. See PLAN.md §5.

## WAM-RX milestones 1 and 2

The repository now also contains the first two implementation slices of WAM-RX: a
small, stdlib-only authoritative memory kernel. It includes a hash-chained,
append-only SQLite event store; deterministic replay; explicit correction and
tombstone events; artifact stamps and support manifests; a non-neural hybrid
retrieval baseline; complete selection journaling; and regional adequacy and
two-sided deletion checks.

Its contracts and kill criteria are frozen in
[`contracts/wamrx_milestone1.json`](contracts/wamrx_milestone1.json). E0.10 is
the falsification experiment. It includes a deliberately biased compiler that
passes pooled coverage (0.923) while failing rare-region coverage (0.0), so the
worst-region gate is demonstrated rather than merely asserted.

Milestone 2 adds typed grounding and no-compounding gates, ledger-sequence
transaction semantics with normalized UTC timestamps, explicit artifact-runtime
compatibility, provenance-linked temporal analytics with immutable query
journals, and a belief/constraint graph that preserves rejected claims and
unresolved constraints. Its contracts are
[`contracts/wamrx_milestone2_foundation.json`](contracts/wamrx_milestone2_foundation.json)
and
[`contracts/wamrx_multiview_memory.json`](contracts/wamrx_multiview_memory.json).
E0.3, E0.4, and E0.11 pass at their registered small synthetic scope, including
malformed-schema, rare-region, evidence-laundering, stale-ontology,
contradiction, tombstone, and unjournaled-query controls. Neural recurrence and
expert routing remain deferred.

Milestone 3 is frozen at the pre-model boundary in
[`contracts/wamrx_recurrent_reasoner.json`](contracts/wamrx_recurrent_reasoner.json).
E0.12 validates the matched fixed/flat/hierarchical comparison, randomized-depth
protocol, external residual halt gate, 960 content-hashed deterministic tasks,
and executable trace/compute schemas. All 12 checks and eight manipulations pass,
but the model comparison is explicitly `NOT_RUN`; this is assay readiness, not
evidence for recurrence.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy
```

```bash
.venv/bin/python rig_a/experiments/e1_1_spectrum_knee.py
```

The WAM-RX milestones need only Python's standard library:

```bash
python3 -m unittest discover -v
python3 rig_a/experiments/e0_3_no_compounding.py
python3 rig_a/experiments/e0_4_grounding_audit.py
python3 rig_a/experiments/e0_10_wamrx_memory_kernel.py
python3 rig_a/experiments/e0_11_multiview_memory.py
python3 rig_a/experiments/e0_12_recurrent_reasoner_assay.py
python3 tools/check_record.py
```

Every experiment is seeded and writes a JSON record to `results/`. Kill
criteria are stated in each experiment's module docstring, before the first
run — a result can always be narrated into a pass otherwise.

## Layout

```
PLAN.md              the plan: claim ledger, rigs, kill criteria, phases
claims/claims.yaml   machine-readable claim ledger with pre-registered predictions
docs/                design amendments and reviews
contracts/           frozen WAM-RX mechanism declarations and kill criteria
wamrx/               authority, multiview memory, recurrent contracts, and tasks
rig_a/core/          spectrum.py (R_t, three readings) · world.py (practice world)
                     ledger.py (typed entries, cards, adapters) · influence.py (functional ground truth)
rig_a/experiments/   one file per experiment
results/             seeded JSON records
tests/               stdlib unit and end-to-end checks for the milestone kernel
schemas/             executable recurrent trace and compute-accounting schemas
```
