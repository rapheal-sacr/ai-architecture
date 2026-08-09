# The §0 re-audit · does the thesis predict new deletions?

**Worklist v3 item 1.2, and it is the only free item that tests a *prediction*
rather than checking a mechanism.**

§0's table found eight sites where the system *estimates from traffic* a fact the
ledger *already records* — against the records that existed when it was written.
Four records exist now that did not: the **selection journal**, the **commitment
register**, the **fleet residual set**, and the **coverage cascade**.

> **The thesis makes a prediction: new records should reveal new estimates to
> delete.** That prediction has never been tested. If none of the candidates
> survives, that is informative about the thesis in a way nothing else on the free
> list is.

Each candidate has to clear two checks, and *both* are places the audit can fail:

| | |
|---|---|
| **Is the estimate real?** | the design must actually estimate it, not merely mention it |
| **Is the record sufficient?** | the record must determine the quantity — same referent, available at the same time, at the same grain |

---

## Result: one clean, two partial, one new — and the partials are the finding

| Site | Estimate today | Record now available | Verdict |
|---|---|---|---|
| **merge prior** | basis overlap between owners | the register's `basis: [direction_id]` | **clean** — overlap becomes a set operation |
| **deletion risk** | `F_max` assigned by entry *category* | tombstone events, recorded in L1 | **clean, and new** — not among the three proposed |
| **gap set** | posterior variance per signature over turns | coverage cascade's unowned fraction | **partial** — records the never-compiled half only |
| **routing** | which region a query is in | the journal's winning card | **partial** — right referent, wrong time |

---

## 1 · Merge prior — clean

§8 selects merge candidates on *"high transfer plus basis overlap."* Under the
spectrum, basis overlap between two adapters is estimated from their subspaces.
The register records `basis: [direction_id]` per owner, so overlap is
`|b_i ∩ b_j|` — a set intersection on records, exact, no estimator.

**Both checks pass.** The estimate is real (it gates a real operation) and the
record is sufficient (same referent, available at merge time, same grain).

**One condition.** The register's schema says `basis: [direction_id] | sketch`. The
substitution holds for the id form and *not* the sketch form — under a sketch,
overlap is estimated again. So this is an argument for recording bases as ids, and
that argument did not exist before the register did.

## 2 · Deletion risk — clean, and it was not on the list

§3 prices fan-out by deletion risk: *"entries likely to be tombstoned — personal
data, user-corrected facts, quarantined untrusted sources — get low fan-out."*
That is a **category judgment standing in for a rate**, and the rate is recorded:
every tombstone is a ledger event, so `P(tombstone | entry class)` is a count over
history rather than a taxonomy someone wrote down.

**Both checks pass**, and this is the audit's own yield — it was not among the
three candidates the review proposed, and it was found by asking the §0 question
of §3 rather than of the four new records.

**What it costs:** a cold ledger has no tombstone history, so the category
judgment is still needed as a prior. The substitution is *eventual*, not immediate
— which is a different shape from the original eight, all of which were immediate.

## 3 · Gap set — partial, and the boundary is the finding

Part I §3 / L8 integrates *"posterior variance per domain signature over turns"* to
find where competence is thin. The coverage cascade records what no live owner
covers.

**The estimate is real. The record is not sufficient**, and the gap is precise:

- **unowned** ⇒ nothing was ever compiled there. The cascade records this exactly,
  and it is I11's quantity — E0.6 measured 0.483 at thin traffic.
- **owned but weak** ⇒ something was compiled and is poor. No record covers this;
  it needs measurement, and the gap-set estimator is what measures it.

So the cascade **decomposes** the gap set into a recorded half and an estimated
half rather than deleting it. That is a smaller claim than the original eight
supported — and it is the useful one, because the two halves have different
repairs. The recorded half is closed by promoting *something*; the estimated half
is closed by improving what is already there.

## 4 · Routing — partial, and it fails on *time*, not on referent

§1.2 keeps `R_t` for routing and §1.5 keeps the ontology for it, so routing infers
which region a query belongs to. The journal records which card actually won.

**The referent is right and the timing is wrong.** Routing decides *whether to use
weights or go to retrieval* — a decision taken **before** retrieval runs. The
winning card is known only **after**. A record that postdates the decision it would
inform cannot replace the estimate for that decision.

Two things survive:

- **offline**, the journal fits the routing policy on recorded outcomes rather than
  on an inferred partition — the estimate stays online but stops being *trained*
  against an inferred label;
- **for any decision taken after retrieval** — which card to condition on, what to
  cite — the record replaces the estimate outright.

**This is the check the audit needed to be able to fail, and it half-failed.** A
record is not a substitute merely by naming the same quantity; it has to be
available when the decision is made. That criterion did not exist in §0's original
table because all eight of its sites happened to satisfy it silently.

---

## What the audit says about the thesis

**The prediction is confirmed, and weakly.** New records did reveal new deletions —
two clean ones, and one of those was not on the proposed list. So the thesis has
predictive content rather than being a description of eight findings already in
hand.

**But the yield has changed shape.** All eight original sites were *immediate,
total* substitutions. Of four candidates here, two are clean (one of them only
*eventually*, once history accumulates) and two are partial. That is consistent
with the easy sites having been found first, and it means the remaining yield of
"look for an estimate where a record exists" is smaller per site than §0's table
implies.

**And the audit produced a criterion §0 did not have.** *A record substitutes for
an estimate only if it is available at the time the decision is made.* All eight
original sites satisfied that silently, so nobody had to state it. Routing is the
first site where it binds, and it is the reason routing is partial rather than
clean.

**What would have falsified this:** all four candidates failing either check. Two
did fail the sufficiency check — which is why the result is "confirmed, weakly"
rather than "confirmed", and it is the more useful answer.
