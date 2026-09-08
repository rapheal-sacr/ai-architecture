# A controller repair improves learned action use, but does not repay neural cost

All80 registered cases and their audit completed at terminal exit0. Changing
only action selection and its declared Bellman arithmetic improves every E2E,
first-order and count-memory case on the reused E36 worlds. No model is retrained,
no additional model query is made and no private world label enters decisions.
The original soft planner was a substantial R3 limitation, not merely a symptom
of absent memory. Read E36-RESULT.md for the original experiment and exact-model
diagnosis; E36-GREEDY-PROTOCOL.md was frozen at fd84c69 before the80-case run.

| Model | Stationary soft → greedy | Recurring | Drifting | Noisy |
|---|---:|---:|---:|---:|
| Static + updates | 137.00 → 142.67 | 131.33 → 169.00 | 122.50 → 125.17 | 114.67 → 149.83 |
| First-order meta + updates | 185.33 → 251.33 | 164.00 → 223.33 | 166.83 → 227.33 | 145.83 → 215.17 |
| E2E meta + updates | 182.17 → 252.50 | 171.67 → 225.00 | 164.67 → 229.67 | 148.67 → 216.00 |
| Rolling counts | 180.00 → 246.00 | 153.50 → 204.50 | 169.50 → 219.00 | 154.50 → 219.50 |

Numbers are mean goals per512paid actions. Each neural cell has three
initializations on the same two worlds; counts have those two worlds once.
The80cases contain only eight distinct worlds. They were reused after diagnosis,
so this is a post hoc intervention, not a fresh transfer experiment.

![All paired cases, with means](</media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/adaptive-latent-workspace/research/adaptive_latent_workspace/results/e36_greedy.png>)

## Positive result and retained counterexamples

Greedy improves E2E24/24cases by44–101goals, mean64.00. First-order improves
24/24, mean63.79; counts improve8/8, mean57.88. Static-plus-updates improves18/24
and loses6/24, with a worst regression of49goals. The three stationary-world1
static cases regress; one recurring and two drifting cases also regress.
Thus the exact-model benefit did not imply a universal imperfect-model benefit.
These whole-policy interventions change later experienced evidence. They do not
isolate the effect of choosing a different action on a fixed learned trajectory.

Under greedy selection, E2E beats first-order on14 paired cases, ties3 and loses7.
Mean gains are only .83–2.33goals by regime. Both use nearly identical acting
time. This weak descriptive ordering on shared worlds does not establish a
general second-order advantage. The unchanged predictive training objective is
still different from learning a goal policy through task returns.

The strong memory control remains competitive. Greedy E2E versus greedy counts
has better/tied/worse cases4/1/1stationary,6/0/0recurring,3/0/3drifting,1/1/4noisy.
It gains20.5mean recurring goals and loses3.5mean noisy goals. E2E observed acting
time remains24.5–25.8times counts. It retains28,672persistent tensor bytes plus
192,128base bytes, versus288count tensor bytes and at most96logical history bytes.
Python/allocator overhead is excluded. This does not pass the full efficiency
requirement, even though action use is now substantially better.

The original observed-edge BFS remains a relevant different controller: its
mean drifting236goals exceeds greedy E2E229.67. Its stationary170.5goals exploit
the cycle structure. We have not changed BFS or selected a favorable control per
world. Constant-action shortcuts still preclude interpreting raw goals alone as
general learned reasoning. The task family must change before a broader claim.

## What changed and what was checked

Softmax(Q/.1) is replaced by uniform selection among Q values within absolute1e-6
of the maximum, followed by the original10%uniform exploration. Bellman values
use float64 in the intervention; horizon6 and discount.95 are unchanged. Real
update equations, model checkpoints, fast-state reset policy, world seeds and
all neural query/update counts match the original cases. A run-local function
substitution is explicit in e36_greedy_intervention.py and hashed in the record;
the original seven E36 sources remain unchanged. Original input manifests are
preserved as provenance, while intervention.json describes the changed policy.

The audit independently reconstructs40,960physical actions, checks36,864real
gradient updates and320evaluation checkpoints, and compares3,456serial map queries
with maximum error3.34e-6. The independent NumPy Bellman policy has zero decision-
probability discrepancy for all neural cases. Counts replay raw bounded histories
with the substituted Torch policy; that part is not a separate policy
implementation. The36training checkpoints and1,152training updates in the audit
totals are inherited via verified hashes; they were not newly replayed.

Scoring took397.762s; the new evaluation audit took212.837s. The audit's larger
1,777.386s field includes the prior1,564.549s training audit and must not be billed
as newly executed work. Training cost belongs to each compared learner even
though the experiment reused checkpoints. Float64 arithmetic and the earlier
main run's overlapping audit make small timing differences unsuitable as a
kernel-efficiency claim. No process energy or full deployment cost was measured.

Source SHA256: 2ea752ac54f095a322dea4446f1ff207e56226a3e1fd26ea4844178e6639622a.
Complete SHA256: 633d8b09a5a73e118b4712de3644a18d59b627bb815b391dbf07ed9c2ee0f0b8.
Full cases and hashes are in results/e36_greedy_summary.json; compact audit in
results/e36_greedy_audit.json. The four-case preflight is separate and supplies
no additional independent-world evidence.

## Architectural consequence

For these models and worlds, changing how knowledge selects actions is a more
direct repair than adding memory or more prediction fitting. That is a local
binding R3 limitation; it does not establish a universal improvement-rate bound.
Recovery cost relative to useful knowledge lifetime remains the operational
candidate, with identifying evidence, retained distinctions and useful execution
accounted separately. No single scalar surprise signal diagnoses those roots.

The attachment still motivates learned goal/compute allocation and slower
consolidation, but their benefits are open. LEARNING-STATE-MIGRATION.md proposes
the next H5/H6 falsifier: equal current readouts can conceal different responses
to later evidence. That diagnostic has not run. The controller repair here was
written by the researcher; it is not the AI improving its own learning procedure.
General reasoning, learned compression, novelty and reliable RSI remain unproven.
