# Sources and reproduction

Read `FALSIFICATION.md` first. This branch is active research, not a verified
continuous-learning system. Results include failed proposals and controls.

## Cloned external sources

| Source | Pinned commit | Used for |
|---|---|---|
| [loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity) | `a6b79580d85f3025bdb601566d3627c5f489f13b` | E4 official data generator; read teacher, BP, CBP and generate-and-test source |
| [Minigrid](https://github.com/Farama-Foundation/Minigrid) | `622fa297cb9e97b6f62f9085d9e053d1ce191b05` | Closed-loop benchmark preparation; no scored candidate run yet |
| [rl-starter-files](https://github.com/lcswillems/rl-starter-files) | `317da04a9a6fb26506bbd7f6c7c7e10fc0de86e0` | Read CNN/LSTM policy, training and observation preprocessing; not yet executed |
| [torch-ac](https://github.com/lcswillems/torch-ac) | `b6602c8ce843a8bfe9cef1723d0151dd7a3c22c3` | Read PPO, rollout and recurrent update source; not yet executed |
| [continual-learning](https://github.com/GMvandeVen/continual-learning) | `e6d795aa81b9cef742b8de76cb71222d4d1ce00b` | Novelty audit: source for task-free replay, model copies and optional context labels; no reproduced benchmark yet |

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
pygame-ce 2.5.7. Installation or a runtime check is not a benchmark result.

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
