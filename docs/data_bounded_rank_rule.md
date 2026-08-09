# R12 · Bound rank by data, not only by budget

**Worklist v2 item 1.2.** Derivation from EB.4's measured curve. No new data — `f`
is already measured, and this is the rule that falls out of it.

---

## The problem EB.4 created

EB.4 puts the split-half stability floor for a top-8 subspace at **≈0.65 × dim**.
Below it, a per-owner spectrum is sampling noise and any committed rank read off
it is a property of the draw rather than of the capability.

**That is a *data* floor, not a compute floor, and the distinction is the whole
point.** More background lane does not help. An owner compiled from 40 ledger
entries cannot produce 1,000 *independent* activation vectors at any budget —
tokens within one entry are not independent draws, so the entry is the unit of
independence and `n ≤ |provenance_o|` however long the texts are.

So the floor binds hardest on **exactly the rare, small-provenance owners the
register exists to protect.** The mechanism proposed to fix the tail has its own
tail problem, one level down.

---

## `f`, read off EB.4's curves

Stability rises about linearly in `log₂(n)`, so the largest rank whose subspace is
stable at a given `n` is a fit rather than an extrapolation past the data. Largest
`r` with fitted split-half stability ≥ 0.90:

| n (entries) | dim 896 | dim 1536 |
|---|---|---|
| 40 | **none** | **none** |
| 100 | **none** | **none** |
| 250 | **none** | **none** |
| 500 | none | 4 |
| 1000 | 8 | 4 |
| 2000 | 16 | 8 |
| 5000 | 16 | 16 |

*(`none` = no rank has a stable subspace at that provenance size. The 1536 column
is non-monotone at n=500–1000 — that is fit noise on three measured points, not
structure, and it is left visible rather than smoothed.)*

---

## The rule

> **R12 · `rank_o ≤ f(|provenance_o|)`.** An owner's basis size is bounded by what
> its own provenance can support, with `f` the curve above. A rare owner gets a
> small basis because that is all its data supports. **If that is not enough
> competence, the answer is more provenance, not more rank.**

Three things this changes:

**It makes §1.3's granularity argument conditional on a quantity it never named.**
"Prefer many small owners" assumed granularity is limited by wall-clock (EB.1) and
then by oracle cost (E0.7). It is limited before either by *whether a small owner's
provenance can support a basis estimate at all* — and below a few hundred entries,
at these dimensions, it cannot.

**It gives the register a rule where it currently has none.** Allocation is set
arithmetic over the complement of live owners' bases (§1.2), and nothing in that
says how large a request may be. `rank_o ≤ f(|P_o|)` is the missing bound, and it
is derived rather than chosen.

**And it converts a silent failure into a refusal.** Without it, a 40-entry owner
requests rank 8, gets it, and its basis is fitted to noise — with every protection
statistic computed against that basis reported as an equal draw, satisfying I8.
With it, the request is refused and the owner is told what it needs.

---

## What it does not settle

**`f` is measured on EB.2's eight short-text domains**, so the curve is that
corpus's. A more diverse buffer spans more directions and would move the floor
**up**, making this optimistic — the `none` rows would extend further right.

**The independence assumption is conservative and unverified.** `n ≤ |provenance|`
treats one entry as one independent draw. If entries within a domain are
themselves correlated, the effective `n` is lower still; if a single long entry
contains genuinely independent content, it is higher. Measuring the effective
sample size per entry is a corpus question and joins asset A.

**~~What a sub-threshold owner should do instead is unstated.~~ [CLOSED — there is
a third answer, and it is better than both candidates.]** The two I had were
inheriting a basis from a provenance-adjacent owner — which reintroduces the
neighbourhood clause E0.7 killed, in a different currency — and allocating from a
pooled estimate, which costs the owner its independent statistic. Both are worse
than a basis of its own.

> **A sub-threshold owner does not become an owner.** It stays an L3 skill card
> and is served by retrieval.

No similarity judgment, so the neighbourhood clause stays dead. No pooling, so
nothing loses an independent statistic. And `|provenance| ≥ threshold` becomes a
**promotion criterion** — a recorded quantity checkable *before* promotion, which
makes it a **sound screen** in §6's sense: one more hoisted conjunct of the
promotion conjunction, free, and it synthesises nothing.

**The cost is honest and it lands on the tail.** Rare capabilities stay
retrieval-served rather than compiled. That is not a new regime — it is the one
L4's gate `g` already handles, routing to retrieval where weights should not be
trusted. Whether it is *acceptable* is measurable rather than arguable: compiled
against retrieval-served on the same probes.

**One coupling to state, because the two mechanisms pull against each other.** R12
pushes a sub-threshold owner to broaden its provenance in order to qualify, and
§3's fan-out cap constrains which entries it may broaden *into*. A thin owner
seeking promotion and a saturated entry are in direct conflict, and neither
mechanism knows about the other.
