# ai-architecture

Falsification harness for **WAM** (Write-Ahead Memory) — a memory-first AI
architecture built around one authoritative append-only ledger, with every
other tier (latent index, compiled views, weights, harness code) treated as a
derived, recompilable view of it.

The design lives in `Architecture Design/` alongside this repo. This repo is
the part that tries to break it.

**[PLAN.md](PLAN.md)** is the entry point: the claim ledger, the rigs, the
kill criteria, and the phase schedule.

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
| **B** | Small real model, MLX, ≤3B 4-bit | Laptop, 8 GB ceiling |
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

Eight repairs proposed (R1–R8), **none adopted**. R8's admission-threshold
component is withdrawn by E0.2d and replaced by provenance-aware admission. See PLAN.md §5.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy
```

```bash
.venv/bin/python rig_a/experiments/e1_1_spectrum_knee.py
```

Every experiment is seeded and writes a JSON record to `results/`. Kill
criteria are stated in each experiment's module docstring, before the first
run — a result can always be narrated into a pass otherwise.

## Layout

```
PLAN.md              the plan: claim ledger, rigs, kill criteria, phases
claims/claims.yaml   machine-readable claim ledger with pre-registered predictions
docs/                design amendments and reviews
rig_a/core/          spectrum.py (R_t, three readings) · world.py (practice world)
                     ledger.py (typed entries, cards, adapters) · influence.py (functional ground truth)
rig_a/experiments/   one file per experiment
results/             seeded JSON records
```
