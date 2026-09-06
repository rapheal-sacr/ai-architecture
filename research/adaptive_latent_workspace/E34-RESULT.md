# E34: retained function is real, but intervening updates degrade access to it

Read FALSIFICATION.md and E33-RESULT.md first. All540 conditions completed from
76fb017a9aeca69e02580960d4199f1f090e5089. Auditor75439 validates all source-state
choices and records, reexecutes216 adaptive conditions exactly, and compares324
frozen conditions with an independent dense float64 attention reference. Maximum
logit discrepancy6.2434e-6, below preregistered tolerance. Frozen fast weights,
initialization and source checkpoints remain exact. Scorer73.965s; audit42.676s.
This reuses E33 streams post hoc; it is not a new-task generalization study.

Disabling updates preserves most first-return-cycle performance. For carry-
trained E2E, frozen/adaptive correctness is74.5%/73.4% on A→B→A,67.2%/67.7% for
longer contexts and39.6%/40.1% after A→B→C→D→E. Thus early returns do use retained
function; they cannot be explained solely by updates during that cycle.
Causal token context remains available even in the frozen conditions.

Continued updates matter over the complete return: carry-trained frozen versus
adaptive accuracy is82.3%/93.1%,77.4%/97.7% and45.8%/80.1% in those regimes.
Source firstA acquired in all these streams. Restoring the earlier first-A
snapshot without updates instead gives96.1%,99.3%,95.6%. Holding initialization,
inputs and cleared KV fixed while restoring only fast weights recovers much
of the earlier function. This establishes functional interference under the
current readout, not information-theoretic erasure of every latent fact.

Archive selection is an oracle: the experimenter knows the return isA. No
learner chose an archive from observations, maintained a bounded archive bank
or paid its search/admission cost. Each extra fast snapshot adds24,576 tensor
bytes plus metadata; shared initialization, evidence, caches and process costs
remain additional. The diagnostic is not a free efficiency gain or a deployed
memory solution.

Even the correct frozen snapshot has only50–75% first-chunk accuracy across
E2E training/regimes, versus94.5–99.4% across the full return. Preserved weights
are therefore not an instant exact lookup from a cold context. The role of
initial-context features versus selection and stored associations needs testing;
this experiment does not isolate all those causes. Static-training arms did
not acquire short contexts, so their drops must not be called forgetting of
already mastered rules. All cells follow.

| Training | Regime | Return treatment | InitialA acquired | First4 accuracy | First16 accuracy | Full-return accuracy | Full-return NLL |
|---|---|---|---:|---:|---:|---:|---:|
|e2e_carry|extra_interference|archive_adaptive|12/12|60.4%|88.5%|97.0%|0.6635|
|e2e_carry|extra_interference|archive_frozen|12/12|60.4%|88.5%|95.6%|0.7288|
|e2e_carry|extra_interference|current_adaptive|12/12|20.8%|40.1%|80.1%|1.1043|
|e2e_carry|extra_interference|current_frozen|12/12|20.8%|39.6%|45.8%|1.9019|
|e2e_carry|extra_interference|reset_frozen|12/12|10.4%|9.9%|8.3%|2.8666|
|e2e_carry|familiar_schedule|archive_adaptive|12/12|54.2%|87.5%|96.7%|0.6971|
|e2e_carry|familiar_schedule|archive_frozen|12/12|54.2%|87.5%|96.1%|0.7731|
|e2e_carry|familiar_schedule|current_adaptive|12/12|43.8%|73.4%|93.1%|0.8383|
|e2e_carry|familiar_schedule|current_frozen|12/12|43.8%|74.5%|82.3%|1.2142|
|e2e_carry|familiar_schedule|reset_frozen|12/12|8.3%|8.3%|7.9%|3.0145|
|e2e_carry|long_life|archive_adaptive|12/12|54.2%|88.5%|99.3%|0.3855|
|e2e_carry|long_life|archive_frozen|12/12|54.2%|88.5%|99.3%|0.4128|
|e2e_carry|long_life|current_adaptive|12/12|31.2%|67.7%|97.7%|0.4548|
|e2e_carry|long_life|current_frozen|12/12|31.2%|67.2%|77.4%|1.0991|
|e2e_carry|long_life|reset_frozen|12/12|2.1%|9.9%|10.9%|2.9258|
|e2e_reset|extra_interference|archive_adaptive|12/12|75.0%|92.7%|97.8%|0.7129|
|e2e_reset|extra_interference|archive_frozen|12/12|75.0%|92.7%|97.4%|0.7798|
|e2e_reset|extra_interference|current_adaptive|12/12|31.2%|30.7%|76.0%|1.2716|
|e2e_reset|extra_interference|current_frozen|12/12|31.2%|32.8%|33.6%|2.1692|
|e2e_reset|extra_interference|reset_frozen|12/12|10.4%|8.3%|9.1%|2.7865|
|e2e_reset|familiar_schedule|archive_adaptive|12/12|50.0%|86.5%|96.4%|0.7426|
|e2e_reset|familiar_schedule|archive_frozen|12/12|50.0%|85.9%|94.5%|0.8148|
|e2e_reset|familiar_schedule|current_adaptive|12/12|37.5%|69.8%|91.7%|0.8925|
|e2e_reset|familiar_schedule|current_frozen|12/12|37.5%|67.2%|75.4%|1.2931|
|e2e_reset|familiar_schedule|reset_frozen|12/12|4.2%|5.2%|5.2%|2.9700|
|e2e_reset|long_life|archive_adaptive|12/12|62.5%|90.6%|99.4%|0.4547|
|e2e_reset|long_life|archive_frozen|12/12|62.5%|90.6%|99.4%|0.4887|
|e2e_reset|long_life|current_adaptive|12/12|35.4%|59.9%|97.1%|0.5533|
|e2e_reset|long_life|current_frozen|12/12|35.4%|62.0%|68.8%|1.4068|
|e2e_reset|long_life|reset_frozen|12/12|4.2%|5.2%|3.7%|3.0159|
|static|extra_interference|archive_adaptive|0/12|16.7%|37.5%|53.0%|2.1626|
|static|extra_interference|archive_frozen|0/12|16.7%|40.1%|48.7%|2.2136|
|static|extra_interference|current_adaptive|0/12|12.5%|16.1%|39.3%|2.2874|
|static|extra_interference|current_frozen|0/12|12.5%|15.6%|19.1%|2.4811|
|static|extra_interference|reset_frozen|0/12|8.3%|6.8%|7.9%|2.6977|
|static|familiar_schedule|archive_adaptive|0/12|4.2%|30.7%|53.9%|2.1386|
|static|familiar_schedule|archive_frozen|0/12|4.2%|34.4%|45.3%|2.1874|
|static|familiar_schedule|current_adaptive|0/12|2.1%|24.5%|48.3%|2.1787|
|static|familiar_schedule|current_frozen|0/12|2.1%|25.5%|34.5%|2.2936|
|static|familiar_schedule|reset_frozen|0/12|2.1%|7.8%|11.3%|2.6789|
|static|long_life|archive_adaptive|7/12|16.7%|55.7%|72.8%|1.9485|
|static|long_life|archive_frozen|7/12|16.7%|54.7%|71.3%|1.9753|
|static|long_life|current_adaptive|7/12|14.6%|33.9%|69.0%|1.9815|
|static|long_life|current_frozen|7/12|14.6%|33.3%|37.2%|2.2571|
|static|long_life|reset_frozen|7/12|6.2%|7.3%|5.8%|2.7336|

Per-seed carry-trained comparison, complete-return accuracy averaged over four
source streams. These are paired diagnostic conditions, not fresh replications.

| Seed | Regime | Current frozen | Current adaptive | Archive frozen | Archive adaptive |
|---|---|---:|---:|---:|---:|
|33101|familiar_schedule|84.0%|92.6%|96.9%|96.9%|
|33101|long_life|75.9%|97.4%|99.6%|99.6%|
|33101|extra_interference|55.1%|84.8%|97.3%|97.3%|
|33102|familiar_schedule|81.6%|92.6%|96.9%|96.9%|
|33102|long_life|78.9%|97.7%|99.0%|99.0%|
|33102|extra_interference|37.9%|76.6%|93.0%|96.1%|
|33103|familiar_schedule|81.2%|94.1%|94.5%|96.5%|
|33103|long_life|77.5%|97.9%|99.2%|99.2%|
|33103|extra_interference|44.5%|78.9%|96.5%|97.7%|

The candidate should now test preserving useful fast states and selecting them
from already observed evidence. Next comparisons must include unpredictable
recurrence, newly arriving rules, finite capacity and routing/update costs.
A fixed-prefix representation boundary can make shared evidence compatible with
fast-weight changes, but correct selection, cold-context access and bounded
memory still need demonstration. MSA-SOURCE-AUDIT.md constrains later encoder
changes. No learned archive/compression policy, general reasoning, autonomous
procedure self-revision, architectural novelty or full-goal success is claimed.
