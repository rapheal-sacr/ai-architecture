# E31 result: tutorial mastery fails before transfer can be claimed

Read FALSIFICATION.md first. All18 frozen cells completed from `ff78ad7`.
The audit reconstructs all4,293 queries and exposure counts, verifies reference
sets with an independent Floyd–Warshall distance calculation, recomputes metrics
from recorded logits and verifies source adapters and work. All648 repeated
identical-prompt queries at identical weights give exactly identical logits.
Scorer checks preserve base, adapter, optimizer, memory and RNG state.

## All registered cells

Entries are tutorial / fresh observed-reference correctness (%). Tutorial sets
contain153 queries each; fresh sets contain81/90 queries for30101/30102. These
are static queries on teacher-induced states, not autonomous candidate episodes.

| Seed | Frozen state | Original names | Fresh random names | Compact IDs |
|---|---|---:|---:|---:|
| 30101 | base | 56.9 / 49.4 | 58.8 / 53.1 | 52.9 / 45.7 |
| 30101 | bootstrap | 58.2 / 55.6 | 55.6 / 50.6 | 64.7 / 58.0 |
| 30101 | adaptive_final | 56.2 / 58.0 | 60.1 / 55.6 | 52.3 / 40.7 |
| 30102 | base | 58.8 / 45.6 | 54.2 / 43.3 | 52.9 / 40.0 |
| 30102 | bootstrap | 76.5 / 58.9 | 71.9 / 58.9 | 72.5 / 58.9 |
| 30102 | adaptive_final | 77.8 / 61.1 | 78.4 / 61.1 | 77.8 / 61.1 |

Every cell fails the registered tutorial gate:95% overall, and80% in categories
with at least10 examples. Only original tutorial encoding directly measures
original training fit. The E30 bootstrap states reach89/153 and117/153 allowed
decisions, or58.2% and76.5%. They do not establish tutorial mastery followed by
a transfer failure. Final online adaptation does not repair this gap.

This is not just missing training exposure. After pooling identical prompt/label
duplicates,142/153 tutorial queries on30101 and133/153 on30102 were actually
sampled during bootstrap. Bootstrap correctness on those exposed queries is
85/142 (59.9%) and102/133 (76.7%). Across116 unique tutorial prompt/label pairs
per source, original bootstrap correctness is56.9% and74.1%. Exact-label
accuracy and cross entropy are preserved separately from tie-aware scores.

## Learning can damage a simple existing decision

| Original tutorial category | 30101 base | 30101 bootstrap | 30101 final | 30102 base | 30102 bootstrap | 30102 final |
|---|---:|---:|---:|---:|---:|---:|
| deliver | 15/32 | 32/32 | 32/32 | 15/32 | 30/32 | 32/32 |
| pick_target | 22/32 | 20/32 | 12/32 | 22/32 | 32/32 | 32/32 |
| home_one_edge | 16/32 | 2/32 | 6/32 | 18/32 | 20/32 | 20/32 |
| home_multi_edge | 8/8 | 2/8 | 7/8 | 8/8 | 8/8 | 8/8 |
| stock_one_edge | 10/22 | 12/22 | 7/22 | 10/22 | 10/22 | 10/22 |
| stock_multi_edge | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| explore_one_edge | 10/21 | 15/21 | 16/21 | 11/21 | 11/21 | 11/21 |

On30101, bootstrap improves delivery from15/32 to32/32 while reducing the
unique immediate-home choice from16/32 to2/32. Original tutorial cross entropy
worsens from1.095 to1.572; on30102 it improves from1.094 to0.950. These are
scoped interference/optimization symptoms, not proof of one unique cause.
Sampling order, adapter initialization, learning rate, examples and representation
remain confounded. Gradient replay in E30 already reproduces the updates exactly.

All eight tutorial home_multi_edge queries have two valid first steps. Their
100% scores in some cells do not demonstrate difficult multi-step reasoning;
uniform legal choice already has50% allowed probability on that category. The
audit reports ambiguity and uniform-reference probability alongside accuracy.
Legal delivery choices also supply substantial task structure. Do not inflate
high scores on these categories into general reasoning.

## Renaming sensitivity is real, but compact names are not a repair

On bootstrap30101 tutorial states, fresh nonce renaming turns9 correct answers
wrong and repairs5; compact IDs turn9 wrong and repair19. On bootstrap30102,
fresh names lose8 and repair1, while compact IDs lose8 and repair2. No facts,
record order, option order or weights change. This establishes answer sensitivity
to a pure label transformation on the measured queries. It does not uniquely
isolate tokenization from learned representation or other language-model effects.

Compact IDs do not meet the tutorial gate in any state. On the first final
adaptive state they reduce tutorial correctness from56.2% to52.3% and fresh
correctness from58.0% to40.7%. Thus even this plausible encoding change has a
counterexample. No encoder was selected or trained using these outcomes. IDs
are query-local diagnostic mappings, not a persistent learned address store or
an implementation of memory compression. All paired flips remain in the audit.

## Correlation and cost accounting

The seeds are not independent corpora. Their tutorials share7/8 world
specifications and103 of116 unique prompt/label pairs. Fresh sets share2/4
exact (seed,size) world specifications. Tutorial and fresh world specifications
are disjoint. E30 operational test worlds also differ. The overlap was found
during reconstruction; the frozen run was preserved, and E30-RESULT.md now
contains the correction. No confidence interval treats these repeated queries
or source seeds as independent environments. Future generators must use
disjoint streams rather than adjacent seed ranges.

The diagnostic executes4,293 neural queries,1,167,552 input tokens and
108,707,308,416 forward attention-score elements in501.49 query seconds. Model
loading takes4.71 seconds once per seed. CPU reconstruction takes1.85 seconds.
Peak CUDA allocation ranges1.047–1.054 GB. Attention elements are not FLOPs.
The two query/view artifacts occupy1,342,703 and1,448,329 bytes on WD; they
include audit-only original memories, references and mappings. Only rendered
query text enters the neural model. Each mapping is serialized and charged.

The base holds988,065,536 parameter bytes and adapters1,081,344 bytes. There
are no new gradient updates here. Bootstrap/adaptive states still inherit
their E30 teaching, training and actual interaction costs; diagnostic reuse does
not erase them. Pretraining, energy, complete Python memory overhead and all
process setup/hash overhead are not fully measured. Exact original/renamed
token counts, query times, source hashes and every logit are preserved.

## Root collapse and next work

The dominant located failure is useful learning/computation over already
available facts. Missing observations and memory eviction cannot explain a
wrong answer on an unchanged, exposed tutorial state. Name sensitivity is
another measured limit, but shorter names do not reliably remove it. Retained
distinctions and long-term compression remain separate unresolved roots.

The binding-constraint hypothesis remains recovery cost relative to useful
context life. Here successful acquisition has not been demonstrated, so a
retention/recovery rate cannot be inferred from later failures. A universal
rate bound, indefinite retention and efficient long-horizon behavior remain
unmeasured. No new architecture novelty or reliable RSI is established.

Before adding another memory or procedure module, establish a competent learned
execution baseline. A next controlled study should compare conventional
gradient accumulation and learning-rate controls with equal observed examples,
preserve every failed fit, and test resulting policies on disjoint fresh worlds.
Use an exact replay of the existing bootstrap as an implementation preflight.
Do not silently spend more data/compute, call a better optimizer a novel
architecture, or stop the full goal at a tutorial pass. No E32 protocol is frozen
or running. The full integrated architecture remains incomplete.
