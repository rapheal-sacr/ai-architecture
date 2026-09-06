# Current continuation: implement an integrated closed-loop assay

E29 is complete and audited. It improves a scoped inference-depth comparison at
equal message work, but all N64 whole-graph outcomes still fail. No experiment
is running and none needs restarting. Read `E29-RESULT.md` and
`INTEGRATION-NEXT.md` before designing another narrow component variant.

The pinned Qwen language core passes CPU execution and GPU low-rank-adapter
preflights. Use WD `language-runtime/bin/python`; it imports the separately
cloned Transformers v4.46.3 source and shares the existing torch/numpy packages
read-only. The executable model snapshot and exact hashes are documented in
`LANGUAGE-CORE-PREFLIGHT.md`. Large files remain on WD.

Next implement and audit the persistent delivery-world observation/action loop
and strong explicit-memory controls. Then specify and freeze all task, seed,
training, memory and evaluation budgets before scoring. Preserve the distinction
between pretrained competence, acquired facts, adapter learning and genuine
procedure improvement. The full goal remains the integrated capability in
`GOAL-EVIDENCE.md`; a narrow language or environment pass is insufficient.

No E30 protocol or scored run is registered. Exact read-only graph tracing and
fixed-point source audits are available but do not constitute a new solver or
a validated reasoning repair.
