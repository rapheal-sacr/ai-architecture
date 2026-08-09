# Asset B · specification, and the predicates that decide whether it is adequate

**Worklist v2 §4.** The constraint has moved from sequencing to specification, so
this is written before any acquisition rather than after one. The predicates are
executable — [`tools/validate_asset_b.py`](tools/validate_asset_b.py) — and they
run against the asset, not against the run.

> **EB.3 is why this exists.** It ran, the manipulation check fired twice, and the
> deliverable turned out to be a specification for what the run needed. That
> specification cost a run. This one costs nothing and is checkable before the
> acquisition is spent.

---

## What B unblocks — three, not four

**EB.6 took `r` off the blocked list** by measuring it with a generator and
predicates, neither of which is corpus-shaped. So asset C is already spent and B
carries three measurements:

| | Measurement | Decides |
|---|---|---|
| **M1** | checkability–difficulty correlation | where reality sits on E2.1's sweep — and therefore harvest yield, which §1.3's oracle line now turns on entirely |
| **M2** | certified fraction on real traffic | §2's re-registered kill (availability during a deletion window) |
| **M3** | entry-degree distribution of an unconstrained bank | §3's fan-out kill |

---

## The predicates, and what each one is protecting against

Full text and thresholds are in the validator. What matters here is *which failure
each one is named after* — all four come from this record rather than from
imagination.

### P0 · independent resolution — the cross-cutting one

At least a fifth of interactions must resolve by a mechanism that is **not the
model under test** (`executable`, `replay`, `human`).

This is **E2.1's laundering**, and it is the only predicate that contaminates
every measurement at once. An outcome resolved by the judge and then used to
validate the judge makes M1's checkability axis the judge's opinion of itself, and
leaves M2's outcomes untrustworthy for the same reason. **It fails silently** —
the log looks complete, every field is populated, and nothing about the file
announces it.

### P1b · difficulty independent of outcome — the E0.2 trap for this asset

Difficulty must be recorded from a source that is not the outcome.

E0.2's defect was that ground truth and mechanism shared a definition, so no
world could return anything but zero. The same shape here: if difficulty is read
off success, then checkability-vs-difficulty is a correlation between a variable
and itself. The predicate checks `difficulty_source != "outcome"` per row, which
is a fact about the asset and not a hope about the analysis.

### P2a · decomposable scorer — §2's stated tension, as a predicate

§2 already names it: *"a reranker in the retrieval path costs you certificates."*
That was written as a tension to be priced. It is better as an admission check —
if the log comes from a cross-encoder, **M2 cannot be run at all**, and that is
knowable from one field before anything is analysed.

### P2b · rival tail bounded — E0.5's decomposition, applied forward

**75.9% of uncertified selections are rival rise**, and a rival outside top-k
cannot be bounded at all. So the log must carry either the full candidate set or
the **(k+1)-th score** as an entry threshold. Without it M2 measures the best case
and silently omits its own dominant failure mode — which is exactly what E0.5's
rig did, since retrieval there was argmax over *all* cards.

### P3a · bank is unconstrained — §3's kill needs an unfiltered bank

If the source system already applied a cosine or overlap cap, the degree
distribution is **the filter's, not the structure's**, and M3 answers a question
about someone else's threshold. E0.2e is the precedent: R9 could not be evaluated
because a tighter cap shrinks the bank and moves the thing being measured.

---

## The cheapest adequate asset

Reading the predicates as requirements rather than checks, the minimum is smaller
than "a production traffic log":

- **~600+ interactions**, spanning ≥2 resolution mechanisms with ≥20% independent
- **≥500 distinct cited entries**, so a top-1% tail exists
- **retrieval logged with the full candidate set** (or the (k+1)-th score), from a
  decomposable scorer
- **difficulty stamped at request time**, before the outcome is known
- **the bank captured before any admission filter**, or the filter's parameters
  recorded so its effect can be undone

The two that are hardest to retrofit are P1b and P3a, because both require a
decision made *at capture time*. Difficulty stamped after the fact is not
independent, and a bank filtered before capture cannot be unfiltered. **Those two
are the ones to get right first**, and they are the reason this document exists
before the acquisition rather than after it.

---

## What this does not do

**It does not check that the asset is representative.** Every predicate here is
about whether a measurement is *computable*, not whether its answer generalises.
E2.1's difficulty bias is a representativeness problem and §10 already concedes it
has no repair in the design — a log that passes every predicate can still be one
deployment's traffic, and M1's correlation would then locate *that* deployment on
E2.1's sweep rather than locating reality.

**It does not cover asset A.** Form-diverse expository text for the spillover
differential (4.2) is a separate acquisition with a separate failure mode, already
specified by EB.3: domains must differ in **form as well as topic**, and the format
control is the check. A and B do not overlap and should not be conflated into "a
corpus" again.

**And two predicates fire in pairs**, which is correct rather than a defect:
breaking P0 makes resolution mechanisms constant, which is what P1c independently
checks; breaking P2b adds an unlogged candidate, which is what P2c independently
checks. A fired predicate names the one measurement it blocks — read them per
measurement, never as a total.
