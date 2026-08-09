# Asset B2 · candidate datasets, checked against R13's predicates

**B2 is the scarce half** — items carrying *difficulty independent of outcome*
(P1b). Everything else about asset B is a property of a pipeline I build; this is
the one part that has to come from somewhere else.

Checked against the predicates rather than read from papers, per R13's own model:
**a fired predicate names the one measurement it blocks.**

---

## HotpotQA first — one label, two boundaries of different kinds

Not "rejected" and not "fine". Checked against the paper rather than either
summary, and it is worse than a two-labels-under-one-name problem.

From HotpotQA §4 (arXiv 1809.09600), on how the split was made:

> *"This train-easy set contains 18,089 mostly single-hop examples. We implemented
> a question answering model based on the current state-of-the-art architectures…
> Based on this model, we performed a three-fold cross validation on the remaining
> multi-hop examples. Among these examples, the models were able to correctly
> answer 60% of the questions with high confidence (determined by thresholding the
> model loss). These correctly-answered questions … are split out and marked as the
> train-medium subset… After splitting out train-easy and train-medium, we are left
> with hard examples."*

So `level` has **two boundaries and they are not the same kind of thing**:

| boundary | how it is drawn | P1b |
|---|---|---|
| easy \| rest | single-hop vs multi-hop — **structural** | **passes** |
| medium \| hard | a SOTA model answered it correctly at high confidence, by thresholding model loss | **fails** |

And the paper confirms the second boundary carries no structural signal: the
multi-hop ratio is **93.3% in train-medium against 92.0% in hard**. The two
subsets are structurally indistinguishable; only model performance separates them.

**This is not fixable by choosing which label to take**, which is what a
two-labels-under-one-name problem would allow. The consequence is narrower and
usable:

> **HotpotQA's binary reading survives — `easy` vs `medium ∪ hard` is single-hop vs
> multi-hop, and satisfies P1b. Its three-way reading does not, at any level of
> care.**

**And there is a third quantity called difficulty on top of this.** The percentile
thresholds on retriever scores widely used in the literature are a *different*
label again, and outcome-derived. So `difficulty_source` must record not merely
*which* difficulty was taken but **which boundary** — and a dataset that ships one
field with two boundaries of different kinds is the sixth instance of one name
covering two quantities in this record, and the first found in someone else's
artifact rather than in this one.

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

## A second difficulty axis, which makes the pre-registered caveat testable

R12's caveat was *"structural difficulty may not be the difficulty that matters"* —
recorded, and not checkable with one proxy.

**FRAMES supplies a second one.** 824 multi-hop questions requiring **2–15
Wikipedia articles** each, ~36% involving reasoning across multiple constraints,
each carrying a reasoning-type label (numerical, tabular, multiple constraints,
temporal, post-processing). **Article count is a structural difficulty proxy
independent of hop count.**

That converts the caveat into a check: **if hop count and source count disagree
about which items are hard, M1's answer is proxy-dependent and the choice of proxy
has to be reported alongside it.** 824 is too small to be primary and exactly right
for that. 2WikiMultihopQA's four reasoning types give a third axis.

This is the first time a pre-registered caveat in this record has become
measurable rather than staying a caveat.

---

## Asset B1 · BEIR supplies the field that cannot be retrofitted

18 retrieval datasets in one corpus/queries/qrels format, spanning scientific,
news, biomedical and finance collections.

**The dataset identity *is* the recorded domain label.** That is P1d, satisfied by
construction rather than by classifying queries afterwards — which would be exactly
the inferred partition E0.1-K4 is about. Under the corrected P1d it is
`independent_authority`: the partition predates any question I ask of it.

**P0 is satisfied at 100%** by human relevance judgments — `resolved_by: human`,
disjoint from any scorer I build, which is P0a for free.

Everything else remains a build, which is the point: **P2a** (a decomposable scorer
I choose), **P2b** (the (k+1)-th score I log), **P3a** (an unfiltered bank I
compile), **P3c** (my own ratios).

First pass, three that fit on a laptop and differ genuinely: **SciFact** (5K docs /
300 queries), **NFCorpus** (3.6K / 323), **FiQA** (57K / 648). **CQADupStack** adds
a finer second axis — twelve StackExchange sub-forums, so within-dataset domain
structure as well as between-dataset.

**One caution that arrives with it:** HotpotQA is inside BEIR, so the `level`
problem above comes along. Under the binary reading it is usable; under the
three-way reading it is not.

---

## Asset A · form diversity is the requirement, and most corpora fail it

EB.3's finding was that **78% of raw spillover was shared form**. Nearly every
"multi-domain" corpus varies topic while holding form constant — which is precisely
the confound, so "multi-domain" is the wrong search term.

**Source-labelled pretraining corpora are the right shape**, because their sources
differ in form by construction: **Dolma** (peS2o academic papers, Project Gutenberg
books, Wikipedia/Wikibooks, C4 web text), or **Pile** subsets (PubMed Central prose,
FreeLaw opinions, USPTO patents, StackExchange Q&A, GitHub code, Ubuntu IRC
transcripts, arXiv LaTeX).

**Avoid Books3.** A DMCA takedown in August 2023 was followed by copies persisting
anyway, and RedPajama removed it from its own corpus over copyright.
**Gutenberg/PG19 is the clean substitute.**

---

**Sources**

- [HotpotQA dataset fields and level construction](https://deepwiki.com/hotpotqa/hotpot/1.2-hotpotqa-dataset)
- [HotpotQA paper](https://arxiv.org/abs/1809.09600)
- [MuSiQue repository](https://github.com/StonyBrookNLP/musique)
- [MuSiQue fields and id format](https://huggingface.co/datasets/dgslibisey/MuSiQue)
- [2WikiMultihopQA repository](https://github.com/Alab-NII/2wikimultihop)
- [2WikiMultihopQA on HuggingFace](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)
- [FRAMES dataset](https://hyper.ai/en/datasets/34835)
- [BEIR benchmark](https://github.com/beir-cellar/beir)
- [Dolma](https://arxiv.org/pdf/2402.00159) · [Pile subsets and Books3 removal](https://www.unfragile.ai/the-pile)
