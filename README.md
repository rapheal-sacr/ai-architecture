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

**Phase 1 complete — five experiments, five breakages.** The three marked ★
were predicted to fail before running, and failed for the predicted reason.

| ID | Claim | Verdict |
|---|---|---|
| [E1.1](rig_a/experiments/e1_1_spectrum_knee.py) | `occupied rank = #{σ_k > ε}` is a computed budget | **FAIL** on 3 of 4 streams — no stable ε on power-law features (75% of held-out query energy sits in the "free" subspace); realistic mixed traffic leaves 6 free directions of 128; and even the design's own assumed bimodal shape misplaces the cut by two whole committed directions |
| [E1.4](rig_a/experiments/e1_4_aleatoric_magnet.py) | posterior variance is an epistemic gap detector | **FAIL** — one coin-flip region in twelve captures 51% of the practice budget, because the frontier-shaping reward `4p(1−p)` peaks exactly where a coin flip lives. Switching to epistemic variance fixes the magnet and inverts the gate to ρ = −0.99 |
| [E0.2](rig_a/experiments/e0_2_transitive_unlearning.py) ★ | the tombstone cascade reaches the weights | **FAIL** — under provenance recorded at compile time, 95.9% of genuine influence relationships survive deletion and the cascade fires on 0.13 adapters per tombstone instead of 3.23. A recording-policy bug, not a mechanism failure: transitive closure through the conditioning card closes it completely |
| [E4.2](rig_a/experiments/e4_2_blast_radius_seal.py) ★ | the blast-radius fixed point seals the Assay | **FAIL** twice — all four Assay tiers execute through L9-editable code; and moving only *permitted* thresholds takes the promotion rate from 0.005 to 0.322 while every tier requirement stays untouched and the rule reports compliance |
| [E3.3](rig_a/experiments/e3_3_offdiagonal_degeneracy.py) ★ | "maximize off-diagonal mass" is a partition objective | **FAIL** — monotone in fineness, so its argmax is total atomisation, and it ranks the planted true structure near the *bottom*. Scoring partitions by held-out predictive error instead recovers the truth exactly |

Four of the five are specification defects with repairs that reuse machinery
already in the design. **E1.1 is the exception** — a claim about the shape of
real feature spectra that fails on the shape transformers produce, and the one
Phase 1 mechanism that may need replacing rather than correcting.

Five repairs proposed (R1–R5), **none adopted** — each survived one test, which
is not the same as a fix. See PLAN.md §5.

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
rig_a/experiments/   one file per experiment
results/             seeded JSON records
```
