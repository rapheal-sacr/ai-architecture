# E31: frozen tutorial fidelity and address-binding diagnostic

Read FALSIFICATION.md first. E30 does not establish reliable acquisition even
with full memory. This experiment tests its frozen core before another learning
mechanism is added. Settings and gates are in protocol_e31.json; freeze these
files before any scored E31 logits. No gradient updates or candidate actions.

For both E30 source seeds, compare the pretrained base, tutorial-trained adapter,
and final adaptive adapter. Query all reconstructed tutorial observations and
observed-only teacher trajectories on four fresh worlds at sizes4,4,8,8. Every
model sees the same observations under original aliases, a fresh random label
bijection, and compact IDs. Option order, record order and facts remain fixed.
The mapping uses only labels already visible in the query; changing irrelevant
other-world memory cannot change it. Actual token lengths and costs are charged.
Query-local IDs are an encoding diagnostic, not a persistent address system.

Reference sets accept all shortest first steps over observed edges. Delivery,
pickup and carrying-home choices are distinguished from exploration agreement.
Nearest-frontier agreement is not globally optimal exploration. Report exact
teacher-label fidelity and cross entropy as well as allowed-set correctness and
probability. Repeated observations are correlated; reconstructed training
exposure counts include identical prompt/label duplicates. The tutorial gate
is95% overall rule fidelity and80% in each category with at least10 queries.
Only original-encoding tutorial queries directly test original training fit.
No passing diagnostic establishes autonomous task learning or the full goal.

This attacks useful computation/access and symbol binding. Available facts are
held fixed under renaming, so paired answer changes cannot be attributed to
new missing observations or a smaller memory. Token lengths are different;
this is not equal-compute evidence or a uniquely isolated tokenizer mechanism.
Failure on original tutorial states prevents assuming mastery before transfer.
There is no learned compression, procedure improvement or novelty claim here.

Structural preflight accepts both branches of a diamond route, verifies the
unique immediate-home choice and proves irrelevant-world noninterference for
the encoder. Seed1711 supplies a full CPU inventory and16 reduced neural queries
across three states and three encodings. All nine neural cells preserve base,
adapter, optimizer, memory and RNG state. An initial invocation without the
preflight flag was rejected by source-status validation before neural work;
that launch failure is preserved. No scored source logits informed the protocol.

Run all frozen cells; independently reconstruct queries, renamings, references,
exposure counts, work and metrics. Report every state/seed/category, including
correct-to-incorrect and incorrect-to-correct flips. This is a static diagnostic
on teacher-induced states, not evidence of autonomous behavior or retention.
The full architecture, efficient learned memory compression, reliable RSI,
broad reasoning and long-horizon efficiency remain unproven.
