# E35: protected snapshots damage continuous E2E learning

The preregistered archive fails its limited recurrence criterion for both E2E
training objectives. Eight snapshots worsen whole-stream NLL and accuracy on
every seed in all three regimes. More stored alternatives make this controller
worse. Disabling selection restores the active-only learner's exact predictions,
losses, fast weights and KV, despite retaining the archive and its scoring costs.
Read FALSIFICATION.md before treating the earlier archive proposal as established.

## What was executed and independently checked

Frozen source `4e7730fc5445313455dd85f44bc9d00472aaa574`; both E33 reset/carry
E2E initializations, three seeds, three continuous unseen streams per seed,
five policies: 90 cases, 8,192 targets each. No new outer training or checkpoint
selection. Changes are unannounced, can cross four-token chunk boundaries, and
do not reset KV or positions. All policies receive identical observations on
each paired stream. Maps are disjoint from E33 and other E35 streams.

Scorer14837 finished in951.7907s; auditor52212 finished in3685.6357s, both exit0.
The audit checks184,320 exact chunks,149,810 exact gradient updates,923,525
full-core candidate forwards and360 exact saved checkpoints. Original-core
recomputation, independent policy events, metrics and transition regeneration
agree. Source and base weights are unchanged. The scorer's shared suffix is
exact only for the single final adaptive MLP; each candidate also reevaluates
the following static MLP. E35-PROTOCOL.md retains the frozen conditions.

Full raw evidence lives on WD `alw-runs/e35`; hashes and audit are in
`results/e35_complete.json`, `results/e35_audit.json`, and
`results/e35_summary.json`. E35-TABLES.md includes all30 aggregate cells,
18 paired seed comparisons, later-half results, work, time and table controls.

## Main outcome

Accuracy percentages aggregate all three seeds. Relative time is the sum of
bank8 compute time divided by active-gated compute time; it excludes checkpoint
serialization and is not isolated accelerator throughput.

| E2E training | Regime | Active gated | Bank2 gated | Bank8 gated | Bank8/active CPU time |
|---|---|---:|---:|---:|---:|
|Reset|Recurring|94.99|76.31|62.69|1.85|
|Carry|Recurring|95.53|83.50|73.35|1.80|
|Reset|Expanding|83.28|43.98|34.43|1.72|
|Carry|Expanding|85.45|48.55|39.03|1.78|
|Reset|Drifting|87.98|60.40|38.11|1.79|
|Carry|Drifting|88.68|59.29|34.92|1.83|

For recurring streams, reset-trained active/bank8 NLL is.6482/1.6224;
carry-trained is.6049/1.3552. All six seed/objective recurrence comparisons
worsen; no favorable seed rescues the criterion. The failure persists late:
carry-trained drifting bank8 falls to26.87% in the latter half, versus90.55%
active-gated. Reset-trained recurring bank8 is61.71% late, versus97.14% active.

Updating every chunk is a strong additional control. Active-always gives
carry-trained95.51/84.83/88.59% across recurring/expanding/drifting, with
NLL.5312/.9514/.8137. Gating skips72.3%/28.3%/37.7% of updates, retains similar
accuracy and saves about4–10% compute time, but raises NLL. Therefore the gate
is an accuracy/work tradeoff, not an unconditional predictive improvement.

The specialized delayed transition table reaches83.35/75.26/83.23%, using256
logical int64 key/value bytes plus Python overhead and.0087/.0089/.0104s across
the respective three streams. Both active neural objectives beat its aggregate
accuracy, but at much higher cost and with inherited meta-training. The table
uses the frozen four-token update cadence. Neural causal attention also sees
within-chunk inputs; a more sophisticated causal table is not ruled out by this
control. These results do not establish neural efficiency over all cheap solvers.

## Protected returns are not the same as overall learning

Eligibility requires the same learner to have acquired the last occurrence of
the exact map (at least75% of its last16 tokens). Bank8 acquires fewer contexts;
comparing only each learner's qualifying returns would conceal that selection.
The paired common-acquired subset is therefore also reported below. Values
are first-window return accuracy; windows can be shorter at stream end.

| Training | Regime | Common return occurrences | Active gated | Bank8 |
|---|---|---:|---:|---:|
|Reset|Recurring|177|81.01%|76.75%|
|Carry|Recurring|210|83.35%|76.05%|
|Reset|Expanding|118|61.23%|73.99%|
|Carry|Expanding|141|70.83%|75.22%|
|Reset|Drifting|48|66.28%|76.04%|
|Carry|Drifting|47|75.66%|76.73%|

Some selected return subsets benefit, especially reset-trained expansion/drift.
This preserves evidence for useful old competence rather than calling every
snapshot worthless. The subset is conditioned on post-treatment acquisition
by both arms; it is not the unconditional effect of selection. Whole-stream
learning and the recurring criterion remain decisively worse.

## Costs and likely mechanism to challenge next

Active state plus bounded KV uses28 KiB of persistent tensors. Bank2 reaches
100 KiB; bank8 reaches244 KiB, including one probation snapshot. Shared base
parameters add193,152 bytes to every arm. These exclude Python objects,
temporary graphs, allocator overhead, accumulated evaluator traces and logits.
Archive capacity is a tensor-state bound, not a claim of constant process RAM.

Across six objective/regime groups, bank8 uses8.68–9.76 suffix-token evaluations
per input token and updates84.9–97.0% of chunks. Prefix/attention execution is
shared, so this is not an8.68–9.76x whole-model FLOP ratio. The exact no-selection
control uses similar candidate work and matches active-gated behavior; its
state/time costs alone buy no improvement. Original bootstrap, admission
probation, rejected probes, duplicates and evictions are not free learning.

Bank8 performs2,056–3,527 switches per group of6,144 chunks, despite only33–61
admissions and9–37 evictions. Bank2 makes826–1,765 switches with more admissions
and evictions. Simply adding capacity is therefore contradicted as a repair
for this policy. The recorded four-token selection criterion tests retrospective
fit, not whether the chosen state will learn better on subsequent observations.
Frequent replacement may interrupt useful accumulation; this is a hypothesis,
not yet a causal explanation of the complete failure.

`diagnose_e35_selection.py` specifies a post hoc intervention: at every bank8
switch, compare the actual selected-and-updated branch with keeping the prior
active state and applying its own original update gate. Score both on the next
real chunk, stratifying unchanged maps separately. The original trajectory must
still replay exactly. This diagnosis cannot estimate a new controller's complete
trajectory; it isolates local effects before a repair is proposed.

## Design consequence and remaining uncertainty

E34's oracle snapshot recovery did not imply a useful autonomous selector.
E35 falsifies the proposed short-sample selection rule, not all protected memory
or E2E. The no-selection intervention identifies a repairable policy component
within root3 (access/use), interacting with root2 (which learned state survives
each replacement). It does not establish an irreducible information shortage.

Keep active E2E learning as the action-integration baseline. Do not carry the
failed archive selector into the agent merely because its state is bounded.
The operative bottleneck candidate remains recovery cost relative to useful
knowledge lifetime; this experiment measures stream/return performance and
costs, not a universal improvement-rate bound. A complete thresholded recovery
time/censoring distribution is not established here.

ActionE2E now has separate real and imagined interfaces and checked gradients,
but no trained action agent. Learned compression, unknown-context evidence
selection, slow-base migration, general reasoning and repeated successful
procedure self-improvement remain open. No part of E35 establishes the full goal.
