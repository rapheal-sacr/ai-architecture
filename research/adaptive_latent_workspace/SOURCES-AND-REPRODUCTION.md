# Sources and reproduction

Read `FALSIFICATION.md` first. This branch is active research, not a verified
continuous-learning system. Results include failed proposals and controls.

## Cloned external sources

| Source | Pinned commit | Used for |
|---|---|---|
| [loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity) | `a6b79580d85f3025bdb601566d3627c5f489f13b` | E4 official data generator; read teacher, BP, CBP and generate-and-test source |
| [Minigrid](https://github.com/Farama-Foundation/Minigrid) | `622fa297cb9e97b6f62f9085d9e053d1ce191b05` | E8 completed closed-loop mechanism pilot; E10 objective counterfactual |
| [rl-starter-files](https://github.com/lcswillems/rl-starter-files) | `317da04a9a6fb26506bbd7f6c7c7e10fc0de86e0` | Read CNN/LSTM policy, training and observation preprocessing; E8 executes the policy |
| [torch-ac](https://github.com/lcswillems/torch-ac) | `b6602c8ce843a8bfe9cef1723d0151dd7a3c22c3` | Read PPO, rollout and recurrent update source; E8 executes PPO with synchronous environment stepping |
| [continual-learning](https://github.com/GMvandeVen/continual-learning) | `e6d795aa81b9cef742b8de76cb71222d4d1ce00b` | Novelty audit: source for task-free replay, model copies and optional context labels; no reproduced benchmark yet |
| [learned_optimization](https://github.com/google/learned_optimization) | `9561bb68c880ddc4b4eeba7a6ec82c25fe1530d5` | Read learned MLP update policy and full unrolled meta-gradient implementations; conceptual prior art for E9, not a reproduced Google benchmark |

The original 24-paper/641-page reading corpus and past-branch audit are in the
earlier WD dossier at `AI Architecture Research/2026-09-05`, including full text
extracts, `SOURCE-INDEX.md`, `PAPER-REVIEW.md` and the falsification record.
The new learner does not depend on the old WAM or finite-program implementation.

## Runtime and storage

Experiments run on the WD drive under `AI Architecture Research/alw-runs`.
The dedicated `continual-runtime` uses Python 3.12, PyTorch 2.5.1+cu118,
NumPy 2.2.6, Matplotlib 3.10.1 and tqdm 4.67.1. E1–E5 use CPU execution with
one PyTorch thread. The GTX 1060 6 GB passed a real forward/backward preflight,
but the reported tiny-network timings are CPU measurements.

Minigrid is installed from its pinned editable clone with Gymnasium 1.1.1 and
pygame-ce 2.5.7. torch-ac 1.4.0 is installed from its pinned editable clone.
E8 completed; see its full scored outcome and negative transfer result in E8-RESULT.md.
Installation or a runtime check is not a benchmark result.

From this directory, use the dedicated Python runtime (shown as `python` below)
and choose output paths on a drive with sufficient free space:

```bash
python e1_stream.py --execute --out /path/on/large/drive/e1
python e2_isolation.py --execute --out /path/on/large/drive/e2-dev
python e2_isolation.py --execute --fresh --out /path/on/large/drive/e2-fresh
python e3_stress.py --execute --out /path/on/large/drive/e3
python e4_external.py --execute --repo /path/to/pinned/loss-of-plasticity --out /path/on/large/drive/e4
python e5_controls.py --execute --out /path/on/large/drive/e5
python e6_sharing.py --execute --repo /path/to/pinned/loss-of-plasticity --out /path/on/large/drive/e6
python e7_renewal.py --execute --repo /path/to/pinned/loss-of-plasticity --out /path/on/large/drive/e7
python e8_closed_loop.py --execute --repos /path/to/pinned/repository/parent --out /path/on/large/drive/e8
python e9_recursive_updater.py --execute --out /path/on/large/drive/e9
python e10_objective.py --execute --repos /path/to/pinned/repository/parent --runs /path/on/large/drive --out /path/on/large/drive/e10-v2
python e11_meta_control.py --execute --runs /path/on/large/drive --out /path/on/large/drive/e11
python e12_transfer.py --execute --repos /path/to/pinned/repository/parent --out /path/on/large/drive/e12
python e13_amortization.py --execute --repo /path/to/pinned/loss-of-plasticity --runs /path/on/large/drive --out /path/on/large/drive/e13
```

Protocols are JSON files. Completed arm results record protocol and source
hashes; reuse of an output directory with changed sources fails. E4 also checks
the external commit and records the generated stream hash. Reports copy complete
JSON outcomes into `results/`; local report scripts currently default to the
researcher's WD run path. Do not mistake generated plots or rounded tables for
the complete observations.

Equal examples or optimizer updates do not imply equal compute. Reports retain
forward-example counts, training-example counts, parameter counts, tensor bytes
and measured time. Tensor-byte measurements exclude gradients, transient
activations and Python overhead; they are not peak process RAM.

## Benchmark issues found before use

Minigrid's `MiniGrid-MultiRoom-N4-S5-v0` is documented in the pinned source as a
legacy six-room configuration. Use `v1` for four rooms. Standard observations
are partial 7×7 views; a future learner must not read the hidden full grid,
agent coordinates or room list as extra inputs. Environment code can be audited
without granting the agent its private state or transition function.

The official slowly-changing-regression generator rounds up by an extra flip
block. E4 explicitly slices to one million scored observations. Its file is
locally generated, not an externally supplied pickle. Source experiment code
often uses individual-example updates and many more seeds; E4 does not reproduce
those published experiments.

## E14 learned-dynamics source

The full [Gymnasium clone](https://github.com/Farama-Foundation/Gymnasium) is checked out at v1.1.1, commit `17eff00220d210beda933f78b0e52850022b0690`. The installed Pendulum implementation and cloned source both hash to `ce3b08152cccb75ef6c401c399860311cbaf85f44340445f28ae6eb48d1034f8`. Source inspected: `gymnasium/envs/classic_control/pendulum.py`, including its observation/action interface, reward, reset and step equations. The experiment harness creates differing-gravity environments; the planner receives only observations and its learned model. It does not call those equations or read private simulator state. This is an independent small PyTorch world-model/planning pilot, not a PETS or Dreamer replication.

```bash
python e14_world_model.py --execute --repo /path/to/pinned/Gymnasium --out /path/on/large/drive/e14
```

## E16 and policy-proposal prior art

`protocol_e16.json` and `e16_temporal_proposals.py` were committed as `2b0704c` before scored execution. Standalone preflight passed, the scored run completed, and `record_e16.py` retains all 48 cases and inherited/new work separately. `alw-runs/e16` holds original output; frozen learned predictors come from E14 checkpoints with SHA-256 identities in each case.

POPLIN was cloned on WD at `edd8dba50f9049c6164eda774602bef0c299cb51`. No package install, training or numerical replication was performed. The source audit and differences from E14 are in `NOVELTY-AUDIT.md`.

## Persistent student and online procedure tests

E17 source/protocol were frozen at `6063e32`. Scored output is complete in `alw-runs/e17`; all 96 combinations are preserved, including the 24 explicitly reused fixed-optimizer records. Standalone preflight verifies zero-controller equality, persistent state, deterministic repeat and exact state restoration. All learned starting checkpoints carry E9/E11 identities and per-file SHA-256 hashes.

E18 source/protocol were frozen at `d49a3b6`. Standalone preflight passes exact retrospective parameter replay, branch-copy equality, zero-controller proposal equality and a double-precision finite-difference meta-gradient (-0.0638097403777455 analytic versus -0.06380974037767384 numerical). Dummy end-to-end trials also exercise a trained E9 procedure. The scored run is in `alw-runs/e18`; its original logs and incomplete/completed status must be inspected before any restart. Both standalone and execution preflights add work.

E18 scored execution completed. `record_e18.py` and `plot_e18.py` preserve all 32 combinations, mark four reused fixed-optimizer cases and expose the context-dependent failure. The plot was rendered and visually inspected. `results/e18_state_audit.json` verifies all 28 newly written final checkpoints: 2,304 persistent student steps, 512 retained replay examples, identical frozen procedures and changed online procedures. The report discloses incomplete peak-tensor accounting rather than interpreting the raw tracked marker as full process memory.

## E19–E21 and additional prior-art audits

E19 was frozen at `0352ce3` and completed in `alw-runs/e19`. Original-world
prefixes match every observation/target exactly; final students reach 6,400
global updates. `record_e19.py` retains all paired states and both continuation
modes. E20 was frozen at `5b5b81f` and completed in `alw-runs/e20`;
`record_e20.py` preserves all context/capacity cells. E21 was frozen at `b3a45ab`
and passed its evidence-timing preflight before scored execution in
`alw-runs/e21`. Inspect live handles or complete output before any restart.

Cloned on WD: LEO at `de9a0c2a77dd7a42c1986b1eef18d184a86e294a` and HyperCL at
`e32567889f772f8de783a437ff0beb2d426bc7b6`. Their source audits are recorded in
`NOVELTY-AUDIT.md`. Neither repository was installed or numerically reproduced.

E21 also completed. `record_e21.py` retains all 32 cases and the negative
input-shift transfer. `plot_e20_e21.py` renders a combined constraint/repair
figure, visually inspected. `results/e19_state_audit.json` verifies all sixteen
final keep/revert experience-state pairs at 6,400 updates;
`results/e20_state_audit.json` verifies all 32 final diagnostic states and every
oracle replay code. No checkpoint has been used to introduce oracle labels into
an inferred learner.

## E22 and conditional evidence prior art

E22's five-family protocol and implementation were frozen at `2fcffe1` before
scoring. All 100 cases completed; `record_e22.py` retains the complete errors,
gate trajectories, recovery censoring and state audit. The three-panel plot was
rendered and visually inspected. The gate derivative preflight gives
0.00018065443364264556 analytically and agrees with finite differences. All
100 restored state-cost counters and 80 raw reservoir/RNG pair checks match.

[ALPaCA](https://github.com/StanfordASL/ALPaCA) was cloned on WD at
`d06391a1bf11beda573078a6a1d62418aac367c4`. Inspected `README.md` and
`main/alpaca.py` through training, prediction, save/restore and matrix helpers.
The implementation learns a neural basis and prior mean/precision for a linear
output model, computes contextual posterior weights from a regularized Gram
matrix and scores query predictive negative log likelihood. Its training code
samples tasks and varying context lengths. It does not establish arbitrary
hidden change detection or protection against a drifting learned basis in our
persistent stream. No dependencies were installed, notebooks executed or
published numerical results reproduced.

E23 was frozen at `7e8bc9f` and completed all 80 cases. `record_e23.py` verifies
all source identities, 80 restored final state-cost counters, 40 raw
reservoir/RNG pairs and 40 regularized normal-equation residuals. Its plot was
rendered and visually inspected. The scored teachers remain nonlinear; only the
separate algebra preflight claims exact noiseless linear invariance.

E24 was frozen at `db9f711` after its standalone preflight passed. The preflight
exercises accepted E6 merges and verifies that audit instrumentation preserves
predictions, weights, anchors, counters and RNG exactly. Its scored run is in
`alw-runs/e24`; inspect completion and the live handle before any restart.
