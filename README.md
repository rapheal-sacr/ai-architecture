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

Two experiments run. Both found breakage.

| ID | Claim | Verdict |
|---|---|---|
| [E1.1](rig_a/experiments/e1_1_spectrum_knee.py) | `occupied rank = #{σ_k > ε}` is a computed budget | **FAIL** on 3 of 4 streams — no stable ε on power-law features (75% of held-out query energy sits in the "free" subspace); realistic mixed traffic leaves 6 free directions of 128; and even the design's own assumed bimodal shape misplaces the cut by two whole committed directions |
| [E1.4](rig_a/experiments/e1_4_aleatoric_magnet.py) | posterior variance is an epistemic gap detector | **FAIL** — one coin-flip region in twelve captures 51% of the practice budget, because the frontier-shaping reward `4p(1−p)` peaks exactly where a coin flip lives. Switching to epistemic variance fixes the magnet and inverts the gate to ρ = −0.99 |

Two repairs proposed, neither adopted: see PLAN.md §5.

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
