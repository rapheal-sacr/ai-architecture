# Full-goal evidence map — completion remains unproven

Read `FALSIFICATION.md` before design prose. This map preserves the user's full
objective; component tests do not redefine success as a smaller graph or stream
problem. E1–E29 are complete and audited. Language-core and adapter checks establish
execution feasibility only. No experiment is currently running.

| Requirement | Authoritative evidence currently available | Status and missing evidence |
|---|---|---|
| Read the supplied corpus, prior falsification records, both historical branches and model gallery | Preserved WD `2026-09-05/PAPER-REVIEW.md`, `SOURCE-INDEX.md`, prior `FALSIFICATION.md`; current `SOURCES-AND-REPRODUCTION.md` | Prior source/corpus audit completed. It is not a numerical replication of every supplied paper. |
| Design a scientifically novel architecture | `DESIGN-REVISION.md`, `NOVELTY-AUDIT.md`, `REASONING-REVISION.md` | Proposed integrated architecture exists; only subsets are implemented. Novelty is not established and major components have identified prior art. |
| Persistent useful memory | E2/E5, E13, E24–E26 and their saved learner checkpoints | Scoped reuse/access gains and explicit failures. Integrated facts, skills, procedure state and query distribution changes are not validated together. |
| Efficient memory compression | E6/E13 total costs; E24 merge/rare-query audit; E25/E26 access interventions | Incomplete. Compression and access failures must remain separate. Lower complete amortized cost with rare/old query preservation has not been demonstrated. |
| Continual acquisition and retention | Persistent streams E17–E26; closed-loop E8/E10/E12; graph E27 | Incomplete. Several regimes improve; others fail acquisition, forget or reverse transfer. Graph task IDs explicitly request operations and do not solve hidden-context inference. |
| General reasoning | E14–E16 model/search separation; E27 trained graph results; E28 frozen-depth intervention; E29 variable-depth failure | Not established. Familiar-length chain competence fails larger-graph transfer; more recurrence can damage correct answers. The graph core is not a trained language or multimodal reasoning system. |
| Autonomous recursive self-improvement | E9/E11 reset-student tests; E18 actual online self-application; E19 procedure-only continuation intervention | Implemented narrowly, not reliably effective. Procedure gains fail distribution transfer and cost controls. Human-written experiment revisions and fixed randomized-depth training do not count as the system improving itself. |
| Efficient long-horizon autonomous tasks | E12 closed-loop failures; E13 long supervised streams; E14–E16 planning costs | Not established. Observation count is not autonomous task horizon, and better error without repaid work is not efficiency. |
| Identify the binding constraint and collapse problems into roots | `CONSTRAINTS.md`, E17 recovery assay, E20–E26 interventions, E28 depth attribution | Three roots are separable: identifying evidence, retained distinctions, useful computation. Recovery cost relative to useful context life is an operational candidate, not a measured universal rate bound. |
| Spend most effort challenging the design and report failures honestly | Frozen protocols, `E*-RESULT.md`, checkpoint audits and `FALSIFICATION.md` | Falsification work is preserved. No numerical fraction of total research effort is claimed. Failures and censored non-acquisitions remain in reports. |
| Use cloned source and WD for large work | `SOURCES-AND-REPRODUCTION.md`, source pins/manifests and WD run/checkpoint paths | Followed for the recorded experiments. No unobserved or unavailable benchmark result is represented as reproduced. |

Full confidence would require an integrated agent to acquire genuinely new
competencies, preserve useful prior knowledge, improve its own procedure over
multiple generations, and beat strong conventional controls on fresh,
substantially longer closed-loop tasks after charging all training, trials,
storage, routing and execution. The present record does not prove these claims.
An easier component pass must not be substituted for them.

The current numerical measurements are scoped to declared tasks, seeds,
hardware and costs. Unmeasured quantities include universal improvement rates,
complete process-energy cost, indefinite retention, and broad deployment
performance. Explicit mathematical bounds in `CONSTRAINTS.md` require their
stated information and memory assumptions; they do not convert finite tests
into a general convergence or self-improvement theorem.
