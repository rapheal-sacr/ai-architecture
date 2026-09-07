# E35: locally useful switches can accompany a failing learning trajectory

The first diagnostic contradicts the simple explanation that the archive
generally wins on past observations but immediately worsens the next prediction.
Across17,709 actual switches, its next-chunk mean NLL is lower than staying on
the prior active state in four of six objective/regime aggregates. Yet E35's
complete archive trajectories are much worse than never enabling selection.
Local selection quality and the trajectory's capacity to accumulate learning
must therefore be measured separately.

This is a post hoc diagnosis of the failed E35 policy, not a proposed controller
or a new benchmark pass. It replays all18 bank8 cases exactly. At each switch
with a following chunk, two branches start from that same actual history:

- The original branch copies the selected snapshot and applies its original
  current-data update gate.
- The counterfactual keeps the prior active fast state and applies the same
  gate to that state's own current-data loss/errors.

Both are scored on the next four real tokens before another update or selection.
The counterfactual is then discarded. It never changes the audited operational
trajectory. Original full-core operations calculate the counterfactual gradient
and both next predictions. The selected next logits must exactly equal the
already-audited next operational logits. Original events/logits/losses and all18
final archive/probe/fast/KV payloads also replay exactly.

| Training | Regime | Switches with next chunk | Current-sample archive NLL gain | Next NLL: selected minus stay | Selected next accuracy | Stay next accuracy |
|---|---|---:|---:|---:|---:|---:|
|Reset|Recurring|2,240|.450|−.169|30.35%|25.74%|
|Carry|Recurring|2,056|.374|−.110|49.70%|44.59%|
|Reset|Expanding|3,307|.428|+.062|15.26%|16.52%|
|Carry|Expanding|3,109|.388|−.045|13.59%|12.83%|
|Reset|Drifting|3,524|.474|−.023|22.36%|21.15%|
|Carry|Drifting|3,473|.474|+.002|19.56%|19.96%|

Negative next-NLL differences favor selection. These are conditional-on-switch
cases from already selected histories, not all stream predictions or estimates
of a new no-selection policy's overall accuracy. Very poor accuracy in both
branches does not contradict active-from-start's good E35 accuracy: the prior
states and earlier learning paths are different.

Excluding every pair of chunks that crosses a true hidden-map change leaves the
same four-of-six NLL direction pattern. Unchanged-map next NLL differences are
−.152/−.083 for reset/carry recurrence, +.082/−.041 for expansion, and−.017/+.008
for drift. This rejects an explanation relying solely on a change just after
selection. Individual switches still hurt often (about31–51% across groups),
and the aggregate direction is not a guarantee for a particular switch.

The9 switches at the last chunk have no next-chunk outcome and are excluded
explicitly. The diagnosis verifies36,864 original chunks and uses53,127 extra
full-core forwards (212,508 tokens),17,708 counterfactual updates (70,832 gradient
tokens), and27,201,024 extra attention score elements. Total runtime424.1952s
includes replay, checks, counterfactuals and output. None is attributed to the
original learner's operational time or advertised as a speed advantage.

The untrained development assay exercises the replay/report path but has zero
switches; it is not evidence of counterfactual-branch coverage. Coverage comes
from the scored-history diagnosis's17,709 interventions and exact original-core
checks. Raw per-case records and the prewritten diagnostic protocol are on WD
`alw-runs/e35-selection-diagnosis`; aggregate evidence is
`results/e35_selection_diagnosis.json`.

The mechanism still open is how repeated replacement changes future learning
over multiple updates and contexts. No-selection from the beginning already
identifies an overall policy effect. A second, separately specified post hoc
intervention starts from each actual bank state at chunks1024 and1536, disables
only subsequent selection, and continues for1,024 real targets. It retains the
bank/probe and their costs, with an independent full-core active continuation.
Horizon4/64/256/1024 outcomes distinguish immediate loss from later recovery.

## Finite-horizon intervention: stopping switches often restores learning

The second diagnostic completed all36 saved-state branches in131.2283s. At each
checkpoint, it preserves fast weights, KV, archives, probation, counters and
history, changing only `select=True` to `False`. It continues the original
update gate, archive scoring/admission and all associated costs. An independent
full-core active-only continuation from the same fast/KV values exactly verifies
all9,216 new chunks and their resulting fast/cache states. The original trajectory
is already audited and is used for the paired future comparison.

| Future observations | Branches with lower NLL after stopping | Branches with higher accuracy after stopping | Keep-selecting pooled accuracy | Stop-selecting pooled accuracy |
|---|---:|---:|---:|---:|
|4|0/36 (all identical)|0/36 (all identical)|42.36%|42.36%|
|64|23/36|22/36|49.18%|60.68%|
|256|30/36|29/36|45.66%|66.86%|
|1,024|34/36|34/36|43.70%|78.66%|

The first four predictions must match: they use the identical inherited active
state before any differing selection. Short-horizon harm remains real. For
reset-trained recurrence at the midpoint, stopping drops the first64 accuracy
from92.71% to73.96%, but raises the1,024-target accuracy from67.77% to87.04%.
The corresponding NLL switches from being worse1.2977vs.8015 at64 targets to
better.9503vs1.5050 at1,024. This is a measured horizon-dependent reversal.

All12 objective/regime/checkpoint aggregates improve at1,024 targets:

| Training | Regime | Starting chunk | Keep accuracy | Stop accuracy | Keep NLL | Stop NLL |
|---|---|---:|---:|---:|---:|---:|
|Reset|Recurring|1,024|67.77%|87.04%|1.5050|.9503|
|Reset|Recurring|1,536|59.47%|87.24%|1.6917|.9481|
|Carry|Recurring|1,024|80.05%|89.97%|1.1824|.8280|
|Carry|Recurring|1,536|75.94%|89.45%|1.2794|.8410|
|Reset|Expanding|1,024|37.66%|67.71%|2.2955|1.4947|
|Reset|Expanding|1,536|21.88%|63.80%|2.6890|1.6196|
|Carry|Expanding|1,024|39.45%|68.10%|2.2383|1.4572|
|Carry|Expanding|1,536|32.55%|66.37%|2.3769|1.4982|
|Reset|Drifting|1,024|25.52%|81.32%|2.5156|1.1168|
|Reset|Drifting|1,536|33.33%|80.60%|2.3524|1.1378|
|Carry|Drifting|1,024|28.65%|81.28%|2.4742|1.0813|
|Carry|Drifting|1,536|22.10%|81.09%|2.6344|1.0970|

The two1,024-target counterexamples are both carry-trained seed33101 recurrence.
At chunks1024/1536, keeping selection gets972/970 correct of1,024; stopping gets
909/906. NLL also worsens in both. Preserved snapshots remain useful in some
states; “always disable selection” is not established as a universal repair.

Stopping does not restore a past good checkpoint or grant labels/maps. Improvement
comes from subsequent real observations and ongoing active updates. This supports
the learning-trajectory interference explanation for many failed states, but
does not uniquely identify which microscopic gradient interactions cause it.
The horizon reversal explains why a locally sensible selector can still be a
poor continual-learning controller. It also argues against using immediate
predictive fit as the only criterion for admitting procedure changes.

The36 branches together perform36,864 prefix tokens,366,768 suffix tokens,
7,941 updates (31,764 gradient tokens),396 admissions,839 duplicates,1,069
rejections and387 evictions, with zero switches. The independent reference adds
its own work, recorded separately. Runtime includes that reference and is not a
new operational throughput comparison. Source/data/state hashes, all seed and
horizon metrics, branch work and reference work are in
`results/e35_stop_selection.json`; raw outputs are on WD `alw-runs/e35-stop-selection`.

Neither diagnosis trains a repaired selector, action planner or recursively
improving procedure. The next integration uses active E2E as a baseline and
must test actual decisions, rather than extending grammar diagnostics indefinitely.
