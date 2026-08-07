# Procedure · measuring support redundancy in harvested probes

The second of E5.1's four gating quantities. Unlike `H`, it **cannot be measured
on any rig here** — it is a property of real harvested-probe provenance, and this
project has no interaction corpus. So this is a sampling procedure with a
decision rule attached, waiting on data.

Writing it down rather than simulating it is deliberate. E5.2's decision table
shows the parameter spans most of C6's operating range, which is exactly the
condition under which a synthetic value gets quoted as a measurement — the
failure B9 and B11 already committed twice.

---

## What to measure

For each harvested probe *i*:

| Symbol | Definition |
|---|---|
| `k` | number of distinct ledger entries the probe's **expected outcome** depends on |
| `m` | number of those that must survive for the probe to still resolve the same way |

Report **`m` of `k`**, not a fraction. Fractions are ambiguous at small `k`: at
`k = 3`, thresholds 0.50 and 0.34 are the same constraint, and an earlier version
of the decision table listed them as separate rows with identical numbers.

`m` is not a modelling choice — it is a property of the probe. A probe whose
answer is stated identically in four retrieved documents has `m = 1`; a probe
requiring a join across three facts has `m = 3`.

---

## How to sample

**Use the same interaction sample as the tier audit (E2.1).** The two questions
read the same field of the same records, and drawing them together avoids two
sampling passes with two different biases.

1. **Draw** resolved interactions at the rate the sealing policy would harvest
   them, from the traffic distribution as it actually is — not a balanced sample.
   Coverage bias is a *finding* here, not something to design away.
2. **Filter by verification tier first.** E2.1 measured 29–63% of an unfiltered
   harvest as laundered T3 judge opinion, and **a laundered probe carries a
   laundered support set** — its `k` is whatever the judge happened to look at.
   Apply the T0/T1 filter before measuring `k`, or the distribution is
   contaminated by exactly the tier the design forbids from promoting anything.
3. **Attribute support by ablation, not by citation.** What the trajectory cited
   is not what the outcome depended on. For each candidate supporting entry,
   remove it and re-resolve; `k` is the count of entries whose removal changes
   the outcome, and `m` follows from how many can be removed jointly before it
   changes. This is E0.2b's functional method, and it is used for the same
   reason: a citation list is the mechanism's own account of itself.
4. **Report per region, unweighted**, with the worst region beside the pooled
   figure and the ratio as a blindness factor. `k` will plausibly be *smaller* in
   rare regions — less traffic means less redundant coverage of the same fact —
   which is the direction that makes C6 bind, and a pooled mean would hide it.

**Sample size.** Enough that the worst region has ≥30 probes after the tier
filter. At E2.1's measured 33–68% strict yield and a Zipfian region
distribution, that is a few thousand resolved interactions.

---

## The decision rule

From E5.2, at full draw and per-region decay, worst-region over-forgetting
against C6's 0.20 tolerance:

```
       k   m      d=0.05      d=0.15      d=0.25
       3   2       0.025       0.100       0.250   <- fails at high decay
       5   2       0.000       0.006       0.038
       5   3       0.000       0.062       0.156
       8   2       0.000       0.000       0.000
       8   3       0.000       0.000       0.006
      10   3       0.000       0.000       0.000
      10   4       0.000       0.000       0.013
```

> **If real harvested probes rest on 5 or more supporting entries, C6 stops
> binding.** Only the 3-entry shape fails, and only at high decay.

That shape — 2 of 3 — was the value pinned in every run before E5.1 swept it, so
the constant carrying the constraint was the pessimistic end of the range. The
same shape of problem as `H`, and resolved the same way: measure the thing.

### And the second table matters more than it looks

Compile adequacy (**I7**) — the fraction of probes with enough support actually
*drawn* to pass at all — binds harder than C6 at low draw fractions:

```
       k   m     f=0.25     f=0.50     f=0.75     f=1.00
       3   2      0.056      0.319      0.763      1.000
       8   2      0.481      0.919      1.000      1.000
      10   3      0.331      0.913      1.000      1.000
```

**No shape clears the 0.80 floor at a 25% draw.** So the draw cap and support
redundancy trade against each other, and at aggressive caps the binding
constraint is not over-forgetting but whether the compile covered the region at
all — which is I7's whole point, and why it needed to be an invariant rather than
a test note.

---

## What this resolves, and what it does not

**Resolves:** whether C6 binds, and therefore how aggressively the L3 draw can be
capped — which feeds directly back into recompile cost and E0.1's A6.

**Does not resolve:** cascade breadth `β`, which is independent and remains the
architectural gap (R9). Nor latency tolerance `L`, which is a product
requirement to be written down rather than measured.

**Watch for:** `k` correlating with region frequency. If rare regions have
systematically smaller support sets, then C6, I7 and every finding in the
weighting-rule enumeration all bind hardest in the same place, and the four
symptoms of the anisotropy become five.
