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
| [E1.1b](rig_a/experiments/e1_1b_energy_criterion.py) | the budget works under the energy/GPM criterion | **PASS** — all 4 streams, power law included. Budget empties only at 5× over-subscription. **Reverses E1.1's conclusion**: the mechanism is sound, §A just states it in the one form that does not work |
| [E1.4](rig_a/experiments/e1_4_aleatoric_magnet.py) | posterior variance is an epistemic gap detector | **FAIL** — one coin-flip region in twelve captures 51% of the practice budget, because the frontier reward `4p(1−p)` peaks exactly where a coin flip lives. Switching to epistemic variance fixes the magnet and inverts the gate to ρ = −0.99 |
| [E0.2](rig_a/experiments/e0_2_transitive_unlearning.py) | the tombstone cascade reaches the weights | **PARTIAL** — the `transitive` arm was a **tautology** and is withdrawn; the 95.9% figure for `direct` is a knob. The finding it owed and did not produce: cascade invalidates ~54% of adapters per tombstone, implying a tombstone-rate ceiling |
| [E0.2b](rig_a/experiments/e0_2b_influence_and_ceiling.py) | *rebuild* — functional ground truth, third influence path | **FAIL** — `transitive` provenance recalls **0.913, not 1.0**, and no set-based closure can reach 1.0 because the residual dependency is on a retrieval that was never run. And the correct cascade touches 69% of the fleet, putting sustainable deletions **below 1/day** across much of the plausible cost space |
| [E4.2](rig_a/experiments/e4_2_blast_radius_seal.py) | the blast-radius fixed point seals the Assay | **FAIL** twice — all four Assay tiers execute through L9-editable code; and the rule bounds *which* thresholds are editable while placing no bound on their *values*, so the reachable set is unbounded and compliance is reported across all of it |
| [E3.3](rig_a/experiments/e3_3_offdiagonal_degeneracy.py) | "maximize off-diagonal mass" is a partition objective | **FAIL** — monotone in fineness, so its argmax is total atomisation, and it ranks the planted true structure near the *bottom*. The replacement objective is validated only inside a family containing the answer |
| [E2.1](rig_a/experiments/e2_1_tier_laundering.py) | probe harvesting widens the verifiable surface | **FAIL** on 2 of 3 — 29–63% of an unfiltered "T2" suite is laundered T3 judge opinion, and the harvested slice tests each domain's easy corner. But the strict-filter **yield is high** (33–68%), contradicting this plan's own prediction |

**Two published conclusions have been withdrawn** (E1.1's "realistic traffic
leaves 6 free directions", E0.2's "transitive closure fixes it completely").
Both are recorded in `claims/claims.yaml` under `retracted:`.

Almost every failure so far is a **specification defect**, repairable with
machinery already in the design. The exception is E0.2b's deletion ceiling,
which is arithmetic on the design's central move — derived weights + cascade +
hours-to-days recompiles fix deletion throughput at
`capacity / (cascade × recompile)`, and no rewording changes that. It is the
first genuinely architectural cost found, and the only finding with no repair
on the list.

Nothing found so far threatens the ledger-first thesis. The counterweight:
**both serious errors in Phase 1 were in the rig, not the architecture** — see
`bugs:` in the claim ledger — and one manufactured a headline that survived a
full write-up.

Seven repairs proposed (R1–R7), **none adopted**. See PLAN.md §5.

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
rig_a/core/          spectrum.py (R_t, three readings) · world.py (practice world)
                     ledger.py (typed entries, cards, adapters) · influence.py (functional ground truth)
rig_a/experiments/   one file per experiment
results/             seeded JSON records
```
