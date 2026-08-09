# Amendment · two fleets, and the bound nobody stated

`fleet` is one name covering two quantities, and every experiment in this record
has used one number for both. It is not a constant to reconcile — the two
constraints ask about different populations.

---

## A · Split the name

| Quantity | What it counts | Bounded by |
|---|---|---|
| **`concurrent_fleet`** | adapters occupying subspace *simultaneously* — resident at inference | the subspace budget: `free_rank` |
| **`promoted_fleet`** | every adapter that has been promoted and carries provenance — the population a tombstone can invalidate | promotion history, throttled by the gate and by merge/reclamation |

`promoted_fleet ≥ concurrent_fleet`, usually by a lot: a deployment with many
promoted adapters does not hold them all resident, it loads per domain.

**The constraints, restated:**

```
C1      free_rank  >=  concurrent_fleet × RANK_REQUEST
C3/C4   drain and union computed over  promoted_fleet
```

E5.1 asked C1 as `free >= RANK_REQUEST` — whether the budget fits **one**
rank-8 adapter — while C3 and C4 used `fleet = 64`. Same symbol, 64× apart.

**Their ratio is a deployment policy**, so it joins `L` and the I11 density floor
on the *decide* list — or goes on a swept axis until someone commits. It is
swept in the E5.1 re-run rather than chosen, for the same reason the floor was.

---

## B · State the bound §A implies and never wrote down

Part II §A requires adapters to occupy **disjoint bases** — that is the whole
basis of its interference argument. Disjointness gives, immediately:

> **`concurrent_fleet × RANK_REQUEST ≤ dim`**

At rank 8 in 128 dimensions that is **16 adapters, hard**. E5.1's `fleet = 64`
was never coherent under §A: 64 basis-disjoint rank-8 adapters need 512
dimensions in a 128-dimensional space. Not approximately impossible — exactly.

The design states neither the bound nor its alternative. The alternative is that
**adapters serving different regions share basis**, in which case §A's
interference argument has to cover the shared case, and it currently does not:
§A's whole claim is that basis-disjoint composition carries no interference
risk, and it says nothing about what overlapping composition costs beyond
"a measured penalty."

---

## C · Why this must precede the Rig B activation pass

**Basis sharing and per-region feature overlap are the same question.**

- If regions overlap heavily in feature space, adapters serving them can share
  directions, the disjointness requirement relaxes, and `concurrent_fleet` is
  not capped at 16.
- If they do not, `concurrent_fleet ≤ dim / RANK_REQUEST` is hard arithmetic and
  C1 is decided before any simulation runs.

The Rig B pass was scoped to spectrum shape variance and per-region overlap.
With the split it must also answer:

> **Can two adapters share basis, and at what measured interference cost?**

Same activations, same pass — but you have to know to look before you run it,
which is the entire reason this amendment comes first.

---

## D · The declaration gains a third term

`fleet` passes a prose review. `promoted adapters, count, population-wide` and
`resident adapters, count, per-inference` cannot silently be the same row.

This is the **third** instance of one name covering two quantities:

| Name | Two quantities | Found by |
|---|---|---|
| L7's "hours–days" | *disable* (soundness) vs *recompile* (competence) | E0.2c |
| I11's coverage | scalar (binary) vs rate (density) | B17 |
| `fleet` | concurrent vs promoted | this amendment |

So `wam_amendment_mechanism_declaration.md`'s corollary takes one more term:

> **Declare a quantity's name, its shape, and its scope — what population it
> ranges over.**

Name alone is what let one symbol carry two populations through twenty-four
experiments.
