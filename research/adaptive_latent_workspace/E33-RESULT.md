# E33: update-trained memory acquires and reuses; persistent training is not uniformly better

Read FALSIFICATION.md first. All nine cases and252 evaluation streams completed
from63baff850bac6f5d1b65fd6e41a6312a0a13598f in740.606s. Independent audit71977
reexecutes all1152 outer updates to exact model/Adam and all1008 session states
to exact logits/losses/fast weights/KV. Audit time741.402s.768 training maps and
108 evaluation maps are distinct and disjoint. Three model seeds, four paired
worlds per regime; correlated policy/checkpoint cells are not extra replications.

Both E2E training methods pass final-cycle acquisition for every context in
every tested stream. Initial last-cycle correctness is about98–100%. Static
training plus identical SGD fails short-context initial acquisition in every
stream; even16-cycle sessions acquire firstA in only7/12 streams. This supports
training the initialization through its future updates on this synthetic family.
It is not a released-model/paper replication or an autonomous agent result.

Carrying E2E fast weights improves first-return-cycle accuracy: carry-trained
73.4% versus6.2% resetting on A→B→A,67.7% versus9.4% on longer contexts, and
40.1% versus8.3% after A→B→C→D→E. This is useful within-stream persistence after
measured acquisition. However, the return cycle is scored in four chunks with updates between them.
It does not by itself separate retained competence from accelerated relearning.
A frozen-return intervention is the next diagnostic, with no change to E33.

The incremental proposed change—training across persistent contexts—is mixed
against reset-trained E2E, with both using carried updates at evaluation.
It improves overall NLL on long-life and extra-interference regimes for every
seed. On familiar A→B→A it worsens mean NLL1.4722→1.4797 and overall accuracy
71.7%→70.0%, with two seeds worse in NLL. First-return accuracy falls3.12 points
on one seed. The persistent objective is not uniformly superior.

A16-entry last-observed-successor table has higher overall accuracy in all
regimes:76.2%,94.1%,76.6% versus carry-trained70.0%,92.5%,61.5% respectively.
Its first-return accuracy is only4.2%,6.2%,7.3% because it overwrites old rules,
but one complete new cycle restores perfect performance. It is a specialized
hand-written control, not a learned general agent. Preserve both its cheaper
acquisition and the neural learner's better early return; no universal efficiency
claim follows. Original/context-boundary information and simple deterministic
cycle structure are substantial scaffolding.

All-seed training times: static172.145s, reset-E2E208.546s, carry-E2E210.834s.
Each sees73,728 target tokens and computes9,142,272 forward attention-score
entries. Each makes384 outer steps; E2E also makes18,432 inner updates. Equal
observations/forward entries are not equal second-order FLOPs. Wall times are
shared-machine measurements, not isolated throughput. Full scorer and audit
costs above include additional evaluation/record work. Pretraining, broad
reasoning, learned episodic compression and autonomous procedure changes are
not implemented by this small test. Model tensors/output logs/outer graphs
must not be confused with a constant total-process memory claim.

The complete cells and per-seed reversals follow; machine-readable data are in
results/e33_audit.json and results/e33_summary.json. The narrow positive result
justifies extending and challenging this mechanism, not declaring the full
architecture goal achieved or research novelty established.

All registered cases completed and exact replay passed. Interpret with E33-PROTOCOL.md.

Cells share trained checkpoints and paired data. Counts are streams, not independent model replications.

| Regime | Training | Evaluation | Initial acquired | All before return acquired | Return acquired | First-return accuracy | Overall NLL |
|---|---|---|---:|---:|---:|---:|---:|
|extra_interference|e2e_carry|carry_ttt|12/12|12/12|12/12|40.1%|1.6736|
|familiar_schedule|e2e_carry|carry_ttt|12/12|12/12|12/12|73.4%|1.4797|
|long_life|e2e_carry|carry_ttt|12/12|12/12|12/12|67.7%|0.6853|
|extra_interference|e2e_carry|reset_ttt|12/12|12/12|12/12|8.3%|1.6949|
|familiar_schedule|e2e_carry|reset_ttt|12/12|12/12|12/12|6.2%|1.7832|
|long_life|e2e_carry|reset_ttt|12/12|12/12|12/12|9.4%|0.7898|
|extra_interference|e2e_reset|carry_ttt|12/12|12/12|12/12|30.7%|1.7348|
|familiar_schedule|e2e_reset|carry_ttt|12/12|12/12|12/12|69.8%|1.4722|
|long_life|e2e_reset|carry_ttt|12/12|12/12|12/12|59.9%|0.7756|
|extra_interference|e2e_reset|reset_ttt|12/12|12/12|12/12|7.8%|1.6667|
|familiar_schedule|e2e_reset|reset_ttt|12/12|12/12|12/12|6.2%|1.7064|
|long_life|e2e_reset|reset_ttt|12/12|12/12|12/12|5.7%|0.8641|
|extra_interference|static|carry_ttt|0/12|0/12|1/12|16.1%|2.4325|
|familiar_schedule|static|carry_ttt|0/12|0/12|3/12|24.5%|2.3989|
|long_life|static|carry_ttt|7/12|5/12|8/12|33.9%|2.0863|
|extra_interference|static|frozen|0/12|0/12|0/12|6.8%|2.6822|
|familiar_schedule|static|frozen|0/12|0/12|0/12|7.8%|2.7050|
|long_life|static|frozen|0/12|0/12|0/12|7.3%|2.7104|
|extra_interference|static|reset_ttt|0/12|0/12|0/12|7.3%|2.4814|
|familiar_schedule|static|reset_ttt|0/12|0/12|0/12|7.8%|2.5088|
|long_life|static|reset_ttt|7/12|5/12|5/12|7.3%|2.1625|

Carry-trained minus reset-trained; both use carried SGD at evaluation:

| Seed | Regime | Overall NLL delta | First-return accuracy delta | First-return NLL delta |
|---|---|---:|---:|---:|
|33101|familiar_schedule|+0.0270|-3.12 pp|-0.0444|
|33101|long_life|-0.0870|+10.94 pp|-0.2773|
|33101|extra_interference|-0.0680|+15.62 pp|-0.2733|
|33102|familiar_schedule|+0.0065|+10.94 pp|-0.1353|
|33102|long_life|-0.0694|+0.00 pp|-0.2025|
|33102|extra_interference|-0.0244|+3.12 pp|-0.0118|
|33103|familiar_schedule|-0.0107|+3.12 pp|+0.0125|
|33103|long_life|-0.1145|+12.50 pp|-0.2882|
|33103|extra_interference|-0.0913|+9.38 pp|-0.3711|

Specialized last-observed-successor table; no hidden context ID:

| Regime | Initial last-cycle accuracy | First-return accuracy | Return last-cycle accuracy | Overall accuracy |
|---|---:|---:|---:|---:|
|extra_interference|100.0%|7.3%|100.0%|76.6%|
|familiar_schedule|100.0%|4.2%|100.0%|76.2%|
|long_life|100.0%|6.2%|100.0%|94.1%|

All-seed training costs; equal observations and forward attention entries, not equal FLOPs:

| Arm | Seconds | Forward tokens | Attention-score entries | Inner updates | Outer updates |
|---|---:|---:|---:|---:|---:|
|e2e_carry|210.834|73728|9142272|18432|384|
|e2e_reset|208.546|73728|9142272|18432|384|
|static|172.145|73728|9142272|0|384|

Wall times were measured on the shared machine. They are not isolated throughput benchmarks.
Table storage excludes Python overhead; neural tensor storage excludes process/graph/output overhead.
No source-model numerical replication, broad reasoning or autonomous procedure improvement is inferred.
