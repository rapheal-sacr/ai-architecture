# Amendment · I2 and I4, before E0.1

E0.1 cannot be well posed against I4 as currently written. The review is right that
the tempting metric defines its answer into existence — but the deeper problem is
upstream of the experiment. **I4 is a bookkeeping claim wearing a competence
claim's name**, and no experiment design fixes that. This is the same shape as
E1.1: §A stated the subspace budget in the one form that does not work while
pointing at the form that does, and the fix was to the specification.

Three amendments, in dependency order. The second is a prerequisite for the first.

---

## A · I4 · restate as a two-sided, verified property

**Current (Part I, L7 card):**

> *Recompilability of competence. Every active adapter carries the ledger entry set
> it was compiled from and can be regenerated from it. No weight change commits with
> an empty provenance set.*

Everything in that sentence is about records. A system that records provenance
perfectly and recompiles to something useless satisfies it completely. And with a
deterministic recompiler it is true by construction, which is exactly the confident
PASS the review warns about.

**Amended:**

> **I4 · Verified recompilability.** For an adapter `A` promoted with provenance `P`,
> and `S(A)` the set of provenance-indexed suite items `A` passed at promotion:
> recompiling from `P`'s surviving image under the current ledger yields `A′`, and
> `A′` must pass exactly `S(A) \ D`, where `D` is the subset of `S(A)` whose
> supporting entries have been tombstoned.
>
> Two failure directions, both recorded:
> - items in `S(A) \ D` that `A′` **fails** → **over-forgetting.** The recompile lost
>   competence the ledger still supports.
> - items in `D` that `A′` still **passes** → **under-forgetting.** The cascade leaked.
>   E0.2b measures this at ~8.7% and proves no set-based closure closes it.
>
> I4 is asserted only up to verification and never inferred from bookkeeping. The
> discrepancy set is written to L1 as `refuted` entries scoped to the influence paths
> involved.

Four things this buys.

**It can fail in both directions.** The current form is one-sided — regeneration
either happens or does not. The amended form fails if too much competence survives
*or* too little, and those have opposite causes and opposite repairs. A one-sided
test cannot distinguish a leaky cascade from a lossy recompile, and E0.2b already
established that the leak is real and irreducible, so the two-sided form starts
from a measured fact rather than an assumption.

**It makes the tombstone delta the success criterion rather than an exception.**
After a deletion, some suite items *should* now fail. That is unlearning working.
Stating I4 as "all items still pass" makes the design's strongest safety claim
indistinguishable from its failure.

**It absorbs R7.** "Verify unlearning instead of inferring it" stops being a repair
bolted on after E0.2b and becomes what the invariant says. Which is correct: R7 was
never a repair to a mechanism, it was a correction to what the invariant claimed to
guarantee.

**It routes the provenance model's own errors into the gap set.** A discrepancy is
evidence that the influence model is incomplete along a specific path. Written as a
scoped `refuted` entry, it becomes an L8 gap source — the provenance model
improving from its own measured failures, using machinery already in the design.

### The new requirement this creates

`S(A)` requires **the sealed suite to be provenance-indexed** — each item carrying
the ledger entries its expected outcome depends on. Nothing in Parts I–III specifies
this, and it is the one genuinely new artifact here.

It is nearly free, because **probe harvesting and I4's testability are the same
mechanism.** A harvested probe is a real past interaction, so its supporting entries
are already in L1 with IDs; the link is a by-product of harvesting rather than an
annotation task. Part III §5's optimism about harvesting and E0.1's testability turn
out to be one thing.

One care point, because this puts a link between a sealed artifact and a mutable
store: the provenance is recorded **as of seal time and frozen**. A tombstone marks
an item *expected to fail now*; it never edits the item. Links read from the suite
into L1 and never the reverse, so the seal holds.

---

## B · I2 · the stamp is a four-tuple, not a pair

**Current:**

> *every latent structure is a pure function of (ledger entries, weight version)*

That was written before L9 existed. With a harness that L9 may edit, a recompile
runs through a retrieval policy, a compile schedule, and tool schemas that may all
have changed since the original compile. Two more dependencies were added to the
design and never added to the invariant that enumerates them.

**Amended:**

> **I2 · Derivability.** Every derived structure is a pure function of
> `(ledger cursor, weight version, harness version, signature-ontology version)`.
> Every derived artifact carries all four. A change in any one invalidates rather
> than migrates.

Without this, **E0.1's result is uninterpretable.** If `A′ ≠ A`, you cannot tell
whether the ledger changed or L9 changed the compiler, and the experiment reports a
number that confounds the property under test with drift in the machinery testing
it. E0.1 needs harness version pinned across the compile/recompile pair as a control
arm, and pinned separately as a treatment arm — harness drift is a second axis of
I4 failure and it is arguably the worse one, since it is the axis the design created
for itself.

The signature-ontology version belongs in the tuple for the same reason: Part III §3
made the ontology a recompilable L3 view, and adapter regions, τ rows and probe
buckets are all indexed by it.

This amendment also pays for R4's second component — *pin derived views by stamping
each Assay run with the versions it ran against* — because it is the same stamp.
One change, two repairs.

---

## C · State the cost bound, and name the dial it shares with cascade breadth

§G's kill criterion notes I4 is worthless if recompile is `O(ledger)` with an
unbounded ledger. That belongs in the design as a commitment, not only in the test
as a criterion, because the commitment implies a mechanism.

**The honest bound:** recompile reads the compiled view, not the raw ledger, so cost
is `O(|P| + |L3 slice|)` — which requires `|P|` bounded, and **nothing in the design
bounds it.** E0.2b measured cascade breadth (how many adapters one entry touches) at
69% of the fleet. The dual quantity — provenance-set *size* per adapter — sets
recompile cost and is unmeasured.

**Mechanism:** cap `|P|`; on exceeding it, distill to a `compiled` entry carrying its
own provenance.

**And the tension, which should be stated rather than discovered later.** Distilling
provenance coarsens deletion: tombstoning any one of the collapsed entries now
invalidates the whole compiled entry. So the cap is a dial between recompile cost
and deletion precision — **and it is a lever on cascade breadth, in the wrong
direction.** E0.2d found the design has no lever on breadth; this is one, and
tightening it for cost makes breadth worse. If no setting is acceptable on both,
that is a genuine architectural finding rather than a spec defect, and it is cheap
to check because E0.2c already has the recompile-cost machinery.

---

## What E0.1 looks like after this

Arms that can return the other answer:

| Arm | Holds fixed | Varies | Fails if |
|---|---|---|---|
| A0 control | ledger, weights, harness, ontology | nothing | `A′` ≠ `A` at all → the recompiler is not deterministic, and every other arm is uninterpretable |
| A1 ledger drift | harness, weights, ontology | decay + supersede between compile and recompile | over-forgetting on `S(A) \ D` |
| A2 tombstone | harness, weights, ontology | tombstone a supporting entry | under-forgetting on `D` — E0.2b predicts ~8.7% |
| A3 harness drift | ledger, weights, ontology | one L9 generation | over-forgetting attributable to harness alone |
| A4 ontology drift | ledger, weights, harness | one signature re-partition | any competence change at all — the ontology is not supposed to be load-bearing for adapter competence, and if it is, Root 3 reaches further than Part III claims |
| A5 cost | — | ledger length, `|P|` | recompile cost superlinear in `|P|`, or `|P|` unbounded |

A0 is the arm that catches the failure mode the review names: if A0 does not pass
trivially, the experiment is broken; if A0 is the *only* arm, the experiment is
tautological. Its job is to be a control, not a result.

A4 is the arm I would not have thought to include before this amendment, and it is
the one most likely to surprise. It is also the cheapest.

---

## What I would not fix yet

**Do not add a replay cursor that reconstructs pre-tombstone ledger state.** It is
tempting — it would make A2 cleanly auditable — and it directly contradicts real
deletion, which is the design's actual differentiator. I4 should hold *modulo
tombstones*, and the amended statement says so. Building machinery to see around
your own privacy guarantee in order to test an invariant is how a privacy guarantee
becomes decorative.

**Do not bound `|P|` yet.** State the bound as an open design parameter and measure
`|P|` first. Part C's tension means picking a cap before measuring both sides is
choosing a point on a tradeoff curve you have not drawn.
