# Asset B2 · candidate datasets, checked against R13's predicates

**B2 is the scarce half** — items carrying *difficulty independent of outcome*
(P1b). Everything else about asset B is a property of a pipeline I build; this is
the one part that has to come from somewhere else.

Checked against the predicates rather than read from papers, per R13's own model:
**a fired predicate names the one measurement it blocks.**

---

## The rejection first, because it is the field everyone reaches for

**HotpotQA's `level` fails P1b by construction.** The dataset carries an explicit
`level ∈ {easy, medium, hard}`, which is exactly the difficulty axis M1 needs — and
the distinction between medium and hard is *"determined by training multiple
baselines and testing the answerability of the questions."*

That is **difficulty read off model outcomes**. Using it makes
checkability-vs-difficulty a correlation between a variable and a function of the
thing it is being correlated with, and no world returns anything else. It is E0.2's
shape, and it would have been invisible without checking how the label was made.

HotpotQA's `type ∈ {comparison, bridge}` is capture-time and fine — but it is two
values and a question *kind*, not a difficulty.

---

## The two that survive

| | MuSiQue | 2WikiMultihopQA |
|---|---|---|
| **difficulty (P1b)** | hop count **in the id** — `2hop__482757_12019` — and `question_decomposition` gives the sub-question chain. Composed *from* single-hop questions, so hop count is structural by construction | derivable from `evidences`, a list of `[subject, relation, object]` triples forming the reasoning path |
| **spread (P1c)** | 2–4 hops | 2-hop and 4-hop |
| **domain (P1d)** | none | `type ∈ {comparison, inference, compositional, bridge_comparison}`, template-generated at capture. Plus `entity_ids` → Wikidata |
| **outcome (P0/P0a)** | gold `answer` + `answer_aliases` | gold `answer` |
| **sources (P4)** | composed from **five**: SQuAD, T-REx, Natural Questions, MLQA, Zero Shot RE | one construction pipeline, Wikidata + Wikipedia |
| **licence** | not stated on the mirror checked — **verify before use** | **Apache-2.0**, 192,606 rows, on HuggingFace |

**P0 and P0a are satisfied by the shape of the task, not by luck.** A gold answer
means the outcome resolves by string match — `executable`, and *disjoint from any
retrieval scorer*. QA-with-gold-answers is exactly "outcomes that resolve
independently of the judge", which is the predicate that fails silently everywhere
else.

**2WikiMultihopQA is the better fit**, on the licence and on P1d.

---

## The correction this search produced, and it is worth more than the shortlist

Applying P1d to 2Wiki forced a distinction I had collapsed.

P1d said the domain label must be **capture-time**, on the grounds that a label
applied afterwards is an *inferred partition* — E0.1-K4 and E3.3's territory.
2Wiki carries `entity_ids` (Wikidata), and joining those against Wikidata's
`instance of` yields a subject-domain partition *after* the dataset was built. By
the letter of P1d that fails.

It should not. Wikidata's `instance of` is authored by an authority that **predates
the questions and does not depend on any model in this system**. Nothing about it is
derived from the traffic being measured.

> **P1d corrected: the requirement is independence from the system being measured,
> not literal capture time.** Capture-time was a *proxy* for it. A partition is
> **inferred** when the system derives it from its own traffic; it is **recorded**
> when it comes from an authority independent of that traffic. Both pass.

`domain_source` now takes `capture | independent_authority | inferred`, and only the
last fires. **This is a defect in my own predicate found by trying to apply it** —
the same way EB.3's specification was produced by a run rather than by thought.

---

## What this does *not* unblock, and the split it sharpens

**B1's hard half was never the corpus.** None of these datasets carries retrieval
logs at all, so every pipeline-side predicate is still mine to satisfy:

| Predicate | Who supplies it |
|---|---|
| P2a decomposable scorer | a scorer I choose |
| P2b (k+1)-th score | a field I log |
| P2c candidate provenance | my compiled bank |
| P3a unfiltered bank | compiling it myself, unfiltered |
| P3b ≥1600 distinct entries | a property of the bank I build |
| P3c bank/fleet/rollout ratios | my pipeline's own configuration |

So the reclassification holds and gets sharper: **B2's scarce half is resolved and
B1's hard half is a build, not an acquisition.** The corpus was always the easy part.

**Two things to verify before spending anything**, both cheap and both capable of
killing a candidate:

1. **MuSiQue's licence.** Not stated on the mirror checked. It composes from five
   upstream datasets, so the effective licence is the union of theirs.
2. **Run the candidate through `validate_asset_b.py`.** Not the paper — the
   predicates. P1a/P1c/P4 are countable directly; P1b and P1d are answered above but
   should be re-derived from the actual files, since both rest on documentation
   rather than on the data.

**And the honest limit, unchanged.** These are Wikipedia-based multi-hop QA sets.
M1 would locate *that* distribution on E2.1's sweep — not "reality". R13's P4 bounds
representativeness by requiring two independent sources; MuSiQue's five upstream
seeds help, and neither dataset closes it. §10 already concedes representativeness
has no repair in the design.

---

**Sources**

- [HotpotQA dataset fields and level construction](https://deepwiki.com/hotpotqa/hotpot/1.2-hotpotqa-dataset)
- [HotpotQA paper](https://arxiv.org/abs/1809.09600)
- [MuSiQue repository](https://github.com/StonyBrookNLP/musique)
- [MuSiQue fields and id format](https://huggingface.co/datasets/dgslibisey/MuSiQue)
- [2WikiMultihopQA repository](https://github.com/Alab-NII/2wikimultihop)
- [2WikiMultihopQA on HuggingFace](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)
