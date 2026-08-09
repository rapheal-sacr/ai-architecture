# E2.2 · Verifier synthesis, by hand, on one T3 domain

**The design's own declared highest-leverage item, and the one Part III §9 and rev 2
§11 both say to do by hand before building any machinery.** No rig, no card bank, no
traffic. This is that exercise, and it produced an architectural finding rather than
the label-agreement number it was registered for — because the label half is blocked
and the structural half turned out to be decidable without labels.

---

## 0 · What was registered, and what is actually reachable

> **Kill criterion (as registered).** If the decomposition into constraint-wise checks
> does not agree with held-out human labels, or does not reject known-bad cases the
> base judge accepted, the recipe is not real and none of L10 should be built.

Four things that criterion bundles, with different availability:

| | Question | Reachable by hand? |
|---|---|---|
| **D1** | Can the judgment be decomposed into constraint-wise checks at all? | **yes** |
| **D2** | Are the checks executable, deterministic, individually falsifiable? | **yes** |
| **D3** | Do they reject known-bad cases? | **partly** — only against bads I construct |
| **D4** | Do they agree with held-out human labels? | **no** — needs a labelled corpus |

D3 is worth naming as compromised rather than reporting: a decomposition tested
against cases its own author constructed is generator and evaluator sharing weights,
one level up, which is the failure mode the design's own §4.1 exists to prevent. It
is reported below and it is not evidence.

D1 and D2 are decidable now, and they turn out to decide more than expected.

---

## 1 · The domain

**L3 skill-card admission: "is this card a faithful, useful distillation of its
source entries?"**

Chosen because it is load-bearing rather than convenient — it is the judgment L3
makes on every admission, the design currently routes it to a judge model, and it is
squarely T3 by the design's own tier definitions (no executable ground truth, no
counterfactual replay that settles it).

---

## 2 · The decomposition — what constraint-wise checks reach

Every check below is executable, deterministic, and individually falsifiable: for
each one there is a card that fails it and passes all the others.

| # | Check | Executable by | Falsified by |
|---|---|---|---|
| C1 | every factual claim maps to ≥1 cited entry | claim extraction + entailment against cited text | a card asserting something no source contains |
| C2 | no claim contradicts a cited entry | NLI, or exact match on structured fields | a card that negates its own source |
| C3 | every cited entry id exists and is live | set membership in L1 | a citation to a tombstoned entry |
| C4 | content cosine ≤ admission threshold against the bank | vector arithmetic | a near-duplicate of an admitted card |
| C5 | fan-out: ∀e ∈ prov(c), F(e)+1 ≤ F_max(e) | degree count in the provenance hypergraph | a card citing an already-saturated entry |
| C6 | recompiles from its own provenance to itself, within ε | re-run the distillation (I4) | a card that cannot be regenerated from what it cites |
| C7 | schema and type validity | schema check | a malformed card |
| C8 | asserts nothing a `refuted` entry denies (I6) | set intersection against refutations | a card restating a refuted claim |

**D1 passes and D2 passes.** The decomposition exists, and it is not a trick: these are
the design's own invariants and mechanisms restated as predicates, which is why they
are executable — they were already recorded.

---

## 3 · What the decomposition does not reach, and this is the finding

Enumerating what the human judgment covers that no check above touches:

| | Not reachable, and why |
|---|---|
| **usefulness** | whether the card helps a downstream task. Only an *outcome* settles it |
| **salience** | of the many true distillations of these sources, whether this is the one worth keeping |
| **abstraction vs average** | §8's own test needs a *third region neither parent targeted* — an outcome test, T1/T2, not T0 |
| **calibration** | whether the hedging expressed matches the support that exists |

The split is clean and it is not a coincidence:

> **Constraint-wise checks decompose SOUNDNESS. They do not decompose VALUE.**

Every reachable check answers *is this card true, grounded, well-formed, permitted*.
Every unreachable one answers *is this card worth having*. And that is the same
boundary E2.3 already found from the other side, where the finding underneath the
finding was that **the design never says which objective promotion serves**. The
decomposable half is the constraint half. The non-decomposable half is the objective.

---

## 4 · The consequence: `A` has a computable ceiling, and it was assumed not to

Rev 2 §4.2 defines the design's headline metric:

```
A  =  candidates scored  /  oracle labels consumed
```

and argues that moving a domain T3 → T1 makes "its whole capability surface promotable
at once — a categorical gain."

**It is not categorical.** If synthesized checks decompose soundness only, then a
candidate that *fails* a screen is rejected for zero labels, and a candidate that
*passes* still needs a label for the value judgment. Over N candidates, with `r` the
fraction a sound screen rejects:

```
labels consumed  =  N (1 − r)
A                =  N / N(1 − r)   =   1 / (1 − r)
```

| screen rejection `r` | ceiling on `A` |
|---|---|
| 0.25 | 1.33 |
| 0.50 | 2 |
| 0.90 | 10 |
| 0.99 | 100 |

**And `r` is bounded by the fraction of candidates that are *unsound*, not the fraction
that are *bad*.** A sound screen cannot reject a candidate that is true, grounded,
well-formed and worthless — that is precisely what soundness means. So:

> `r ≤ P(candidate is unsound)`

### 4.1 · The uncomfortable dynamic

Improving the generator is the entire point of the system. A better generator emits
fewer unsound candidates. So `r` falls, and:

> **`A` degrades as the system improves.** L10's amplification is largest when the
> system is worst, and tends to 1 exactly as the generator gets good.

That is the opposite of §4.2's assumption that L10 "changes the exponent". It does not
change the exponent; it applies a factor that shrinks as it succeeds. The oracle floor
§10 already admits is not just real — it is *approached from above as the system
improves*, which is the least convenient possible shape.

---

## 5 · The tension this exposes between §4 and §6

The escape from §4.1 would be for a synthesized instrument to score *quality* and not
merely soundness — then it reaches the value half and `A` is not bounded by `r`.

But §6 forbids exactly that for a cheap rung:

> §6: *"Every cheap rung must be a **sound necessary condition** for promotion — something
> whose failure logically implies the full rung would fail. Then recall is 1.0 by
> construction."*

And E2.3 measured what happens when a cheap rung estimates instead: recall of the
expensive rung's top decile falls to **0.349**, with ρ(cheap, full) at **0.322**.

So the two sections pull opposite ways and the design does not say which governs L10:

| If L10's instruments are… | then | and the cost is |
|---|---|---|
| **sound screens** | `A ≤ 1/(1−r)`, self-limiting, degrades as the generator improves | bounded amplification |
| **estimates** | `A` unbounded in principle | E2.3's recall problem one level up — the tail is discarded, and the archive never sees the false negatives |

**This is E2.3's result arriving at L10, and neither document connects them.** §6 was
written about the promotion ladder; it applies verbatim to the verifier field, because
a synthesized instrument *is* a cheap rung by another name.

---

## 6 · What this makes runnable that was not

The registered kill criterion needs a labelled corpus and stays blocked. But §4's
viability now turns on a quantity that needs **no labels and no L10**:

> **Measure `r`: the fraction of generated candidates that fail a sound screen.**

Every check in §2 is executable today against any generator's output. Counting how
many candidates fail at least one gives the ceiling on `A` directly, before any
instrument is synthesized and before any oracle budget is spent. If `r` is small on a
real generator, L10 cannot pay for itself whatever the synthesis recipe turns out to
be — and that is decidable on the same Rig B trip as everything else.

This is the same move the record has made repeatedly: replace a quantity that needs an
expensive new measurement with one already implied by what is recorded.

---

## 7 · D3, reported and not counted as evidence

Constructed bads, one per check, each passing every other check: an uncited assertion
(C1), a self-negating card (C2), a citation to a tombstoned entry (C3), a
near-duplicate (C4), a card citing a saturated entry (C5), a card that does not
regenerate (C6), a malformed card (C7), a restatement of a refuted claim (C8). All
eight are rejected by their intended check and by no other.

**That is a statement about my ability to construct examples matching predicates I
wrote, and nothing else.** Generator and evaluator share weights. It is recorded so
the exercise is complete, and it is not evidence for D3.

---

## 8 · Verdict

- **D1, D2 pass.** The decomposition exists and its checks are executable,
  deterministic and individually falsifiable.
- **D3 compromised by construction**, reported and discounted.
- **D4 blocked** on a labelled corpus. The registered kill criterion cannot fire yet.
- **And the exercise produced a finding the criterion was not aimed at:** verifier
  synthesis decomposes soundness and not value, so `A ≤ 1/(1−r)`, `r` is bounded by
  the unsound fraction, and `A` therefore *degrades as the generator improves*. §4's
  "categorical gain" is withdrawn. §4 and §6 disagree about what an instrument may be,
  and the design does not adjudicate.

**Recommendation.** Do not build L10 on the strength of §4.2's amplification argument.
Measure `r` first — it is executable now, it needs nothing that does not already
exist, and a small `r` kills the mechanism more cheaply than a synthesis attempt would.
