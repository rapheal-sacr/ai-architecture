# E32: tutorial fitting improves; larger-world competence still fails

Read FALSIFICATION.md first. All12 cases/24 snapshots complete from frozen
7d4edc962214188b6d4537e8d261cdc8c3c7ad64. Full independent audit passes
(session63130): ordered examples/groups, optimizer/RNG/work, references,
actual trajectories and observed memory. All30 world specifications/labels
are disjoint. Historical256-update replay and independent joint-gradient
preflights remain separate implementation evidence; the full auditor does not
reexecute every scored gradient. All results, including deterioration, remain.

The8-example/.001 arm reaches the tutorial and initial-acquisition gates on
all three seeds at1024 examples. It nevertheless solves only1/12 larger-world
missions. No seed/arm/milestone reaches the larger-world3/4 gate. The observed-map
planner completes36/36 missions in206 actions with the same observation-only
world interface. Full observed memory was available to every neural arm.
Thus memory truncation and assumed tutorial non-acquisition cannot explain
away all execution failures. This does not yet isolate exploration, planning,
symbol binding and action sampling as causes.

All-seed comparison; tutorial/initial gates are counts of three seeds:

| Arm | Examples | Tutorial gates | Initial gates | Larger successes /12 | Deliveries /36 | Actions | Step-penalized return |
|---|---:|---:|---:|---:|---:|---:|---:|
|batch1_lr1e3|256|0/3|1/3|2|16|875|7.25|
|batch1_lr1e3|1024|1/3|1/3|0|11|987|1.13|
|batch1_lr1e4|256|0/3|2/3|2|21|773|13.27|
|batch1_lr1e4|1024|2/3|2/3|1|16|846|7.54|
|batch8_lr1e3|256|0/3|1/3|2|16|911|6.89|
|batch8_lr1e3|1024|3/3|3/3|1|19|822|10.78|
|batch8_lr1e4|256|0/3|1/3|1|18|942|8.58|
|batch8_lr1e4|1024|0/3|2/3|1|18|879|9.21|

Fourfold additional fitting is not monotonically beneficial. The single-example
higher-rate arm loses5 deliveries in aggregate; the lower-rate arm also loses5.
The accumulated higher-rate arm gains3; the accumulated lower-rate arm ties.
Per-seed reversals remain below. This is not a score-based choice of a winning
optimizer. Accumulation also changes optimizer-step frequency, so it is not a
pure gradient-noise intervention.

| Seed | Arm | Examples | Tutorial | Gate | Initial /4 | Larger /4 | Unchanged return /2 | Relocation | Deliveries /12 | Actions |
|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|
|32101|batch1_lr1e3|256|129/142|False|2|0|1|False|4|306|
|32101|batch1_lr1e3|1024|93/142|False|0|0|0|False|0|384|
|32101|batch1_lr1e4|256|119/142|False|2|0|1|False|4|316|
|32101|batch1_lr1e4|1024|140/142|True|3|0|1|False|5|285|
|32101|batch8_lr1e3|256|127/142|False|2|0|1|True|5|303|
|32101|batch8_lr1e3|1024|139/142|True|3|0|1|False|5|285|
|32101|batch8_lr1e4|256|110/142|False|2|0|1|True|5|332|
|32101|batch8_lr1e4|1024|132/142|False|2|0|1|False|4|316|
|32102|batch1_lr1e3|256|124/140|False|2|1|1|False|5|308|
|32102|batch1_lr1e3|1024|136/140|True|4|0|2|True|8|255|
|32102|batch1_lr1e4|256|111/140|False|3|0|2|True|7|269|
|32102|batch1_lr1e4|1024|135/140|True|2|0|1|False|3|328|
|32102|batch8_lr1e3|256|127/140|False|0|0|0|False|1|384|
|32102|batch8_lr1e3|1024|140/140|True|4|1|2|False|7|276|
|32102|batch8_lr1e4|256|99/140|False|3|1|2|True|8|262|
|32102|batch8_lr1e4|1024|122/140|False|3|1|2|True|8|273|
|32103|batch1_lr1e3|256|114/140|False|3|1|2|True|7|261|
|32103|batch1_lr1e3|1024|127/140|False|1|0|1|True|3|348|
|32103|batch1_lr1e4|256|114/140|False|4|2|2|True|10|188|
|32103|batch1_lr1e4|1024|134/140|False|4|1|2|True|8|233|
|32103|batch8_lr1e3|256|119/140|False|4|2|2|True|10|224|
|32103|batch8_lr1e3|1024|139/140|True|4|0|2|False|7|261|
|32103|batch8_lr1e4|256|99/140|False|2|0|1|True|5|348|
|32103|batch8_lr1e4|1024|128/140|False|4|0|2|False|6|290|

Interpret unchanged-return retention only for a source state that first acquired
its initial missions. Teacher examples include ties and supplied legal actions;
tutorial correctness is not arbitrary reasoning. The evaluation model is frozen,
so these mission sequences do not test online continual policy improvement.

Cumulative training seconds across three seeds at1024 examples: .001/batch1
740.887; .0001/batch1 758.330; .001/batch8 750.121; .0001/batch8 750.508.
Do not add the256 snapshot costs again: they are included in1024. All paired
arms consume the same ordered gradient-example tokens/attention work at each
milestone; batch8 performs8x fewer Adam steps. Diagnostic and acting work is
separately charged in results/e32_audit.json. Shared-machine wall times are not
isolated hardware latency measurements. Pretraining/energy/Python overhead are
not completely measured; tensor/observed-memory accounting is not total cost.

Conventional optimization can repair tutorial fitting in this setup, but not
reliable larger-world execution. This baseline result neither implements E2E's
outer objective nor establishes learned compression, a novel architecture,
general reasoning, efficient long-horizon learning or reliable RSI. E33 separately
implements and tests training through future fast-weight updates.
