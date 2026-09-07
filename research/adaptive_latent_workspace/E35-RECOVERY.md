# E35 post hoc recovery: retain censored contexts

This supplements E35-RESULT.md with an observation-delay assay, not a new
training run or a revised success gate. All90 audited prediction streams and
9,660 segment/learner occurrences are included. `recovery_e35.py` inherits
E35's75%-correct-of16 criterion and checks it at actual four-target observation
chunk boundaries. All16 targets must lie inside the current evaluator-only map
segment. The learner never sees those boundaries or the recovery calculation.

The first qualifying endpoint is recorded; a context ending without one is
right-censored at its actual end. The restricted delay is the observed endpoint
minus the context start, or its duration if censored. Dividing by context duration
gives the restricted lifetime fraction. Censoring does not mean eventual recovery,
and the fraction is not a prediction of recovery beyond the observed lifetime.

There is a16–19-observation floor even for an already competent learner because
the assay waits for a complete window and chunk boundary. Consequently this is
criterion-observation delay, not the earliest instant of acquisition or a minimum
number of samples needed to learn. No per-recovery wall time was recorded; E35's
whole-stream work and time remain separate. This is not a universal rate bound.

| E2E training | Regime | Active recovered/total | Bank8 recovered/total | Active mean restricted fraction | Bank8 mean restricted fraction |
|---|---|---:|---:|---:|---:|
|Reset|Recurring|276/278|183/278|.242|.497|
|Carry|Recurring|276/278|218/278|.238|.401|
|Reset|Expanding|380/407|129/407|.470|.807|
|Carry|Expanding|379/407|156/407|.444|.763|
|Reset|Drifting|280/281|81/281|.323|.798|
|Carry|Drifting|280/281|78/281|.315|.811|

Two recurring segments and one drifting segment per learner have no complete
evaluable window, due to their short ending. They are explicitly separated in
the JSON. Active learners recover in every evaluable recurring and drifting
segment. Bank8 leaves93/58 evaluable recurring segments unresolved for reset/carry
training, and199/202 under drift. In expansion all segments have evaluable windows;
active reset/carry fail to meet the criterion in27/28, versus278/251 for bank8.

Successful-only medians would mislead. Under carry-trained drift, bank8's median
is19 observations versus20 active, but the archive recovers in only78 segments
versus280. Under reset-trained expansion its successful median is18 versus20
active while only129 versus380 contexts recover. Dropping censored contexts
would turn widespread failure into an apparent adaptation-speed benefit.

The report also includes duration-weighted fractions, bank2, active-always,
the exact no-selection bank, previously-acquired subsets, and subsequent windows
falling below criterion after a first hit. The latter guards against interpreting
a transient threshold crossing as durable competence. Final original E35
acquisition counts can differ because that assay checks the segment's final16
targets even when the endpoint is not a chunk boundary.

Two independent implementations (prefix counts and direct window sums) agree
on3,060 development segment fixtures and every scored segment. Cases include
misaligned boundaries, short segments, never-recovered contexts and recovery
followed by loss. No scoring/model state changed. The first report omitted a
separate no-complete-window count; v2 adds that reporting field without changing
any per-segment result (cases SHA remains
`e2e027e3d33bbc1cca186bbe1b499a73fe9daa3d4f5cd5fe68c8a9d2974b849b`).

WD outputs are `alw-runs/e35-recovery-v2`; aggregate evidence and protocol are
copied to `results/e35_recovery.json`. The calculation took8.4301s. This supports
a large recoverable policy component in the operational bottleneck. It does not
identify an irreducible observation limit or prove an efficient general learner.
