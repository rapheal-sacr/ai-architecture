# Amendment · the mechanism declaration

**The meta-repair.** Every other amendment here fixes a mechanism. This one
changes the rate at which the remaining gaps get found, and it is the only item
in the record that does.

---

## The problem it solves

Eleven repairs have been proposed. R9 — provenance-aware admission, the repair
for the last architectural gap — arrived carrying **two unstated couplings** and
could not be evaluated at all:

- the breadth side assumes something about how the adapter fleet composes from
  the card bank as the bank shrinks
- the coverage side needs I11 stated as the rate it already is, not as a binary

That is the same defect the repairs exist to fix. Generalise it:

> **Every new mechanism carries its own unstated couplings, so repairs are
> gap-productive.** If each repair introduces roughly one new unspecified
> coupling, testing never converges — you run experiments to discover
> specification gaps you created in the previous round.

Item 1 of `wam_amendment_before_continuing.md` already caught the shape of this:
adding a `requires_tier` column would have found three of the programme's
findings by inspection, before any of them cost an experiment. This is that
column, extended to the full set.

---

## The declaration

Four columns, filled in **before a mechanism is written**, not in a review.

| Column | Question | Why this one |
|---|---|---|
| **requires tier** | what verifier tier does it need to *function*, as distinct from what it may promote? | E1.4 → R2, E2.1, E3.1c: three findings were the same error — a mechanism described as free while silently requiring verification the domain may not have |
| **consumes** | what resource, in the currency §0 says is binding? | Part III §0 says verification coverage and cost is the binding constraint. A mechanism that does not say what it spends cannot be priced against it |
| **trades against** | what quantity does it make worse — *and what is the status of the threshold on that quantity: measured, decided, or unset?* | Every mechanism specified without stating its tension has been found out by an experiment: the subspace budget, the blast-radius rule, three-λ release, the staged ladder, and now R9. The threshold clause is a correction — see below |
| **assumes** | what couplings does it assume about parts of the system it does not own? | R9's blocking defect. It would have been caught on paper, in the time it takes to fill a row |

---

## Worked example — R9 as it should have been declared

```
mechanism      R9 · provenance-aware admission
requires tier  T0 -- set overlap is computable, no judgement needed
consumes       nothing at evaluation time; card-bank size at steady state
trades against per-region card density (I11 coverage) -- a DISTRIBUTION over
               regions, worst-case binding -- and NOT content redundancy, which
               is what cos <= 0.93 already governs.
               THRESHOLD: the density floor. Status UNSET, and it is a DECIDE
               quantity, not a measured one. E0.2e sweeps it: at a floor of 1
               there is no tension at any tau; at 5 the tension is total even at
               the loosest.
assumes        (a) how many distinct cards an adapter's training draw uses
               (b) whether that is an absolute count or a fraction of the bank
               (c) whether fleet size tracks bank size
               -- none of which this mechanism owns or Parts I-III state
```

The `assumes` row alone would have stopped R9 before an experiment was built.
The `trades against` row would have forced I11's rate form, because a density
tension cannot be stated against a binary.

---

## What it would have caught, retrospectively

| Finding | Column that would have caught it | Cost saved |
|---|---|---|
| E1.4 → R2 relocating to Root 2 | requires tier | one experiment |
| E2.1 tier laundering | requires tier | one experiment |
| E3.1c compositional ceiling | requires tier | one experiment |
| E1.1c traffic-weighted aggregation | trades against | one experiment |
| E1.2 release rate vs allocation safety | trades against | one experiment |
| E2.3 mean filter vs coverage objective | trades against | one experiment |
| E0.2e R9's fleet coupling | assumes | one experiment, unresolved |
| E5.1 `H`, `L`, support redundancy | consumes | three constants, one still open |

Eight of the record's findings. Not all would have been *fully* resolved on
paper — a declared tension still has to be measured — but each would have been
**visible as a question before it cost a build**, which is the difference between
a specification and a discovery.

---

## The rule

> No mechanism enters Parts I–III without all four columns filled. A blank is
> not a default — it is an admission that the mechanism is not yet specified,
> and it blocks the mechanism rather than the reviewer.

And the corollary that matters most, because it is the one this record kept
demonstrating — in a form mechanical enough to check on one line of a table:

> **Declare each quantity's name, its shape (scalar / rate / distribution), and
> its scope — what population it ranges over. Never state a constraint against a
> coarser shape than the quantity it constrains.**

The **scope** term is the third one, added after `fleet` was found covering two
populations through twenty-four experiments. Three instances now, all one name
over two quantities: L7's "hours–days" (*disable* vs *recompile*, E0.2c), I11's
coverage (scalar vs rate, B17), and `fleet` (concurrent vs promoted). `fleet`
passes a prose review; **"promoted adapters, count, population-wide"** and
**"resident adapters, count, per-inference"** cannot silently be the same row.

B13 and B17 were the same error twice: a *distribution collapsed to a scalar*.
B13 reported a rate conditional on passing, so the denominator moved with `k`.
B17 reported a binary where the quantity was a density, and the tension vanished
from the instrument. In both, a constraint was stated against something coarser
than the thing it constrains.

This form is checkable. A `trades against` cell reading **"coverage"** passes a
prose review and cannot be verified; one reading **"per-region card density,
distribution over regions, worst-case binding"** cannot be satisfied by a min-1
test, and the mismatch is visible without running anything.

---

## Pre-registering this amendment

Eight findings surfaced retrospectively is a **fit to the data it was designed
from** — the columns were written by someone who had already found those eight,
so they necessarily catch them. The real test is prospective, and it is
registered here in the form the rest of the programme uses.

> **Pre-registration.** The next mechanism specified under the four columns is
> declared *before* build. The amendment **fails** if any subsequent experiment
> finds a coupling, resource, or tier requirement the declaration omitted.

### One correction found before the test starts

Auditing R9's own draft declaration against the five gating quantities turns up a
gap **in the amendment**, not in R9. The `trades against` row named per-region
card density — the right quantity, in the right shape — but said nothing about
**the floor**, the threshold that decides whether the trade is acceptable. And
the floor is a *decide* quantity: E0.2e shows the finding inverts completely
between a floor of 1 and a floor of 5.

So the four columns as first written name *what* a mechanism trades against and
not *what would settle whether the trade is tolerable*. `trades against` now
carries the threshold and its status — **measured, decided, or unset** — and an
`unset` threshold is as blocking as a blank column.

This was found by inspection rather than by an experiment, so it refines the
amendment rather than failing the pre-registration below. But the prospective
test should start from the corrected form, and it is worth recording that the
meta-repair needed a repair on first contact with its own worked example.

**R9 is the first case**, since its declaration already exists in draft above.
If the fleet coupling and the density floor turn out to be the complete set of
its unstated assumptions, that is the first evidence the process converges. If a
ninth thing shows up, the columns are incomplete and *which* one it is tells you
what to add.

This is the only claim in the record about the **method** rather than the design,
and it is the one that determines whether the remaining work is finite.
