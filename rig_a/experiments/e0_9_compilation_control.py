"""E0.9 -- the compilation control. What M3's number is a property of.

B1's build found entry degree = 1 for every one of 43,332 entries, so section 3's
fan-out kill was unevaluable. The compiler is NOT what this decides -- that is
settled by the design:

    I1 FORCES DEGREE > 1 UNDER ANY NON-DUPLICATING COMPILER. L1 is the single
    authoritative store, so an entry cannot be copied. Two capabilities resting on
    the same observed fact must both cite it; there is no second copy to give one
    of them. Sharing is architectural. A compiler decides HOW MUCH, not WHETHER.

    DEGREE = 1 IS THE SIGNATURE OF A RELABELLING. `doc = card` partitions entries
    along a boundary the corpus already had. L3's cards are procedures distilled
    from recurring patterns, and distillation over recurrences produces overlap by
    construction. M3 was never answerable from a schematic compilation, on any
    corpus.

So `demand` -- a card is the entry set jointly supporting one query -- is the
compiler. What E0.9 decides is WHAT THE RESULTING NUMBER MEANS.

DESIGN: 2 compilers x 2 ledgers x real/null, REPORTED PER LEDGER, NEVER POOLED.

    compiler  schematic (doc = card, as built) | demand (card = query-support set)
    ledger    SciFact | NFCorpus, separately, per the weighting rule
    null      real assignment | rewired at matched marginals

THE NULL, AND A CORRECTION TO ITS SPECIFICATION. E0.9 as proposed said "rewire
entry->card citations at random while holding EACH ENTRY'S citation count fixed."
That null cannot fail: entry citation count IS entry degree, so holding it fixed
preserves the degree distribution exactly and real-versus-null is identically zero.
It is an operation that cannot report failure, which this record has now found
seven times.

The informative null holds CARD SIZES fixed -- each card cites as many entries as
it really does, drawn uniformly from the entry population -- so total citation
volume is controlled and entry degree is free to vary. That isolates STRUCTURE
from HOW MANY CITATIONS EXIST, which is what the proposal asked for. It is E0.7's
birthday null one level up, and E0.7 is why it is here.

MEASURED PER CELL:
    1. entry-degree distribution, and top-1% degree over median degree
    2. distinct entry count, against MIN_ENTRIES = 1600 from R13
    3. certified fraction, single-tombstone -- the same certificate, reusing
       SelectionJournal rather than reimplementing its bound

Item 3 is why M2 is not deferred. Certified fraction depends on margins, margins on
card vectors, card vectors on which entries a card is a mean over -- so it is a
property of the BANK. Running it in both compiler arms costs nothing extra and turns
"a number for a bank about to be replaced" into a sensitivity measurement on
section 2.

PRE-REGISTERED READINGS:
    degree differs BETWEEN LEDGERS under one compiler
        -> ledger information. Section 3's kill is about structure and stands.
    degree differs BETWEEN COMPILERS but not between ledgers
        -> policy information. F_max is a compiler parameter, not a ledger
           property, and section 3's kill must say so.
    demand real matches the rewired null
        -> no structure to cap. Section 3 is a non-mechanism on this ledger.

KILL: if `demand` also yields degree = 1 -- no two queries sharing an evidence
entry -- BEIR cannot support M3 at any compilation, and the acquisition needs a
corpus with genuine evidence reuse. SciFact and NFCorpus plausibly differ on
exactly this, which is why both run.

TENSION, STATED RATHER THAN DISCOVERED: query-support compilation trades entry
count for degree. The population shrinks to evidence entries only -- closer to the
design, since the ledger records what was USED -- but MIN_ENTRIES may then bind.
Measured below, not assumed.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.register import CardBounds, Selection, SelectionJournal  # noqa: E402

MIN_ENTRIES = 1600                 # R13's derived floor
N_QUERIES = 400
N_DELETIONS = 24                   # single-tombstone draws for certified fraction
CERT_SELECTIONS = 80               # selections scored per deletion, for runtime
SEED = 20260806
TOKEN = re.compile(r"[a-z0-9]+")


def sentences(text: str) -> list:
    p = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return p or [text.strip() or "empty"]


def load(d: pathlib.Path) -> dict:
    corpus, queries, qrels = {}, {}, defaultdict(set)
    for line in (d / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
    for line in (d / "queries.jsonl").read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            queries[o["_id"]] = o["text"]
    for f in sorted((d / "qrels").glob("*.tsv")):
        for i, line in enumerate(f.read_text().splitlines()):
            if i and line.strip():
                q, doc, s = line.split("\t")[:3]
                if int(s) > 0:
                    qrels[q].add(doc)
    return {"name": d.name, "corpus": corpus, "queries": queries, "qrels": qrels}


def entries_of(ds: dict):
    """L1: every sentence of every document, indexed once per ledger."""
    texts, by_doc = [], {}
    for cid, t in ds["corpus"].items():
        ids = []
        for s in sentences(t):
            ids.append(len(texts))
            texts.append(s)
        by_doc[cid] = ids
    return texts, by_doc


def compile_bank(ds: dict, by_doc: dict, mode: str, qids: list) -> dict:
    """schematic: doc = card. demand: card = the entry set supporting one query."""
    if mode == "schematic":
        return {f"doc:{c}": list(e) for c, e in by_doc.items()}
    bank = {}
    for q in qids:
        ents = []
        for doc in sorted(ds["qrels"][q]):
            ents += by_doc.get(doc, [])
        if ents:
            bank[f"q:{q}"] = sorted(set(ents))
    return bank


def rewire(bank: dict, n_entries: int, rng) -> dict:
    """The null: CARD SIZES preserved, entries drawn uniformly.

    Not entry-degree-preserving -- that would fix the quantity being measured and
    the comparison would be identically zero. See the module docstring.
    """
    return {c: sorted(rng.choice(n_entries, size=min(len(e), n_entries),
                                 replace=False).tolist())
            for c, e in bank.items()}


def degree_stats(bank: dict) -> dict:
    deg = Counter()
    for ents in bank.values():
        for e in ents:
            deg[e] += 1
    if not deg:
        return {"n_entries": 0, "max": 0, "median": 0, "top1pct": 0, "ratio": 0.0}
    v = np.array(sorted(deg.values())[::-1])
    k = max(1, len(v) // 100)
    med = float(np.median(v))
    top = float(v[:k].mean())
    return {"n_entries": len(deg), "n_cards": len(bank),
            "max": int(v[0]), "median": med, "top1pct": round(top, 3),
            "ratio": round(top / max(med, 1e-9), 2),
            "share_deg1": round(float((v == 1).mean()), 3)}


def tfidf(texts: list):
    toks = [TOKEN.findall(t.lower()) for t in texts]
    df = Counter()
    for t in toks:
        df.update(set(t))
    vocab = {w: i for i, w in enumerate(w for w, c in df.items() if c >= 2)}
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = np.log(len(texts) / (1 + df[w]))
    r, c, v = [], [], []
    for e, t in enumerate(toks):
        for w, f in Counter(w for w in t if w in vocab).items():
            r.append(e); c.append(vocab[w]); v.append((1 + np.log(f)) * idf[vocab[w]])
    return sparse.csr_matrix((v, (r, c)), shape=(len(texts), len(vocab))), vocab, idf


def certified_fraction(bank: dict, E, Q, qids: list, rng) -> float:
    """Reuses SelectionJournal's bound rather than reimplementing it.

    A card vector is the MEAN over its source entries and there is no distillation
    rotation here, so proj_norm = 1. Everything else is the same algebra E0.5
    audited against a world that really performs the deletion.
    """
    cids = list(bank)
    dim = E.shape[1]
    rows = []
    bounds = []
    for cid in cids:
        ents = bank[cid]
        sub = E[ents]
        mean = np.asarray(sub.mean(axis=0)).ravel()
        dev = np.sqrt(np.maximum(
            np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
            - 2 * (sub @ mean) + float(mean @ mean), 0.0))
        bounds.append(CardBounds(k=len(ents), norm=float(np.linalg.norm(mean)),
                                 proj_norm=1.0,
                                 src_dev={int(e): float(d) for e, d in zip(ents, dev)}))
        rows.append(mean)
    C = np.vstack(rows)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)

    sel_idx = rng.choice(Q.shape[0], size=min(CERT_SELECTIONS, Q.shape[0]), replace=False)
    Qd = np.asarray(Q[sel_idx].todense())
    S = Qd @ Cn.T
    log = [Selection(query_id=int(i), scores=S[i].copy(), chosen=int(np.argmax(S[i])),
                     query_norm=float(np.linalg.norm(Qd[i]))) for i in range(len(sel_idx))]
    j = SelectionJournal(bounds=bounds, log=log)

    pop = sorted({e for ents in bank.values() for e in ents})
    fracs = []
    for _ in range(N_DELETIONS):
        e = int(rng.choice(pop))
        fracs.append(float(j.certified([e]).mean()))
    return float(np.mean(fracs))


def main() -> int:
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return 2
    print("\nE0.9 -- the compilation control\n")
    print("  Reported PER LEDGER, never pooled. The null holds CARD SIZES fixed,")
    print("  not entry degree -- fixing entry degree would preserve the quantity")
    print("  being measured and the comparison would be identically zero.\n")

    out = {}
    for d in dirs:
        ds = load(pathlib.Path(d))
        rng = np.random.default_rng(SEED)
        texts, by_doc = entries_of(ds)
        E, _, _ = tfidf(texts)
        qids = [q for q in ds["queries"] if ds["qrels"].get(q)]
        rng.shuffle(qids)
        qids = qids[:N_QUERIES]

        # query vectors, over the same vocabulary
        qtexts = [ds["queries"][q] for q in qids]
        allE, vocab, idf = tfidf(texts + qtexts)
        E, Q = allE[:len(texts)], allE[len(texts):]

        cells = {}
        for mode in ("schematic", "demand"):
            bank = compile_bank(ds, by_doc, mode, qids)
            null = rewire(bank, len(texts), np.random.default_rng(SEED + 1))
            cells[mode] = {
                "real": degree_stats(bank), "null": degree_stats(null),
                "certified": round(certified_fraction(bank, E, Q, qids, np.random.default_rng(SEED + 2)), 4),
            }
        out[ds["name"]] = cells
        out[ds["name"]]["qrels_density"] = round(
            float(np.mean([len(v) for v in ds["qrels"].values()])), 2)

    print(f"  {'ledger':>10}{'compiler':>11}{'arm':>7}{'cards':>8}{'entries':>9}"
          f"{'max':>6}{'med':>6}{'top1%':>8}{'ratio':>7}{'deg=1':>8}{'cert':>7}")
    for name, cells in out.items():
        for mode in ("schematic", "demand"):
            for arm in ("real", "null"):
                s = cells[mode][arm]
                cert = f"{cells[mode]['certified']:.3f}" if arm == "real" else ""
                print(f"  {name if mode=='schematic' and arm=='real' else '':>10}"
                      f"{mode if arm=='real' else '':>11}{arm:>7}"
                      f"{s['n_cards']:>8}{s['n_entries']:>9}{s['max']:>6}"
                      f"{s['median']:>6.0f}{s['top1pct']:>8.2f}{s['ratio']:>7.2f}"
                      f"{s['share_deg1']:>8.2f}{cert:>7}")
        print()

    names = list(out)
    dem = {n: out[n]["demand"]["real"] for n in names}
    sch = {n: out[n]["schematic"]["real"] for n in names}
    kill = all(v["max"] <= 1 for v in dem.values())
    dens = {n: out[n]["qrels_density"] for n in names}

    print("  READINGS, pre-registered:\n")
    print(f"  KILL demand still gives degree = 1:        {'FIRED' if kill else 'no'}")
    if not kill:
        between_ledger = (max(v["ratio"] for v in dem.values())
                          / max(min(v["ratio"] for v in dem.values()), 1e-9))
        between_compiler = {n: dem[n]["ratio"] / max(sch[n]["ratio"], 1e-9) for n in names}
        vs_null = {n: dem[n]["ratio"] / max(out[n]["demand"]["null"]["ratio"], 1e-9)
                   for n in names}
        print(f"       degree ratio, demand: " +
              ", ".join(f"{n} {dem[n]['ratio']:.2f}" for n in names))
        print(f"  between LEDGERS   (demand):  {between_ledger:.2f}x")
        print(f"  between COMPILERS (per ledger): " +
              ", ".join(f"{n} {v:.2f}x" for n, v in between_compiler.items()))
        print(f"  real vs NULL      (demand):     " +
              ", ".join(f"{n} {v:.2f}x" for n, v in vs_null.items()))
        print()
        # THE NULL COMPARISON IS PRIMARY, because it is the only one that
        # controls card size. The between-ledger comparison does not, and these
        # two ledgers differ by 38x in relevant-docs-per-query.
        dratio = max(dens.values()) / max(min(dens.values()), 1e-9)
        print(f"  qrels density (relevant docs/query): "
              + ", ".join(f"{n} {dens[n]:.1f}" for n in names)
              + f"   -> {dratio:.0f}x apart")
        print()
        if all(v > 1.3 for v in vs_null.values()):
            print("  -> REAL EXCEEDS THE REWIRED NULL IN EVERY LEDGER. There is")
            print("     structure beyond citation volume, so section 3's kill is")
            print("     about structure and stands as registered.")
        elif all(abs(v - 1) < 0.15 for v in vs_null.values()):
            print("  -> demand matches the rewired null: NO STRUCTURE TO CAP.")
            print("     Section 3 is a non-mechanism on these ledgers.")
        else:
            print("  -> the null comparison splits between ledgers. Read per")
            print("     ledger; there is no single verdict.")

        if dratio > 2.0:
            print()
            print(f"  AND THE BETWEEN-LEDGER FIGURE ({between_ledger:.2f}x) IS NOT READABLE.")
            print(f"  These ledgers differ by {dratio:.0f}x in relevant-docs-per-query, so a")
            print("  demand card in one is ~38x the size of a demand card in the")
            print("  other. That is a comparison across arms differing in more than")
            print("  the manipulation -- B19's defect, and EB.2's. The null")
            print("  comparison survives it because the null preserves card sizes")
            print("  WITHIN each ledger, which is what it was built to control.")

        print()
        print("  SCHEMATIC SITS BELOW ITS OWN NULL, which is worth stating: real")
        print(f"  ratio 1.00 against a null of "
              + ", ".join(f"{out[n]['schematic']['null']['ratio']:.2f}" for n in names)
              + ". A partition is not merely")
        print("  uninformative about reuse -- it is ANTI-concentrated relative to")
        print("  chance, because every entry has degree exactly 1 by construction.")
        print("  That is the quantitative form of `doc = card is a relabelling`.")

    print("\n  MIN_ENTRIES, and the tension stated rather than discovered:")
    for n in names:
        d_, s_ = dem[n], sch[n]
        print(f"    {n:>10}  schematic {s_['n_entries']:>6} entries,"
              f" demand {d_['n_entries']:>6}"
              f"   {'clears' if d_['n_entries'] >= MIN_ENTRIES else 'BINDS'} the"
              f" {MIN_ENTRIES} floor")

    print("\n  CERTIFIED FRACTION, single tombstone, both banks -- M2a is not")
    print("  deferred, because it is a property of the BANK: margins depend on")
    print("  card vectors and card vectors are means over source entries.")
    for n in names:
        print(f"    {n:>10}  schematic {out[n]['schematic']['certified']:.3f}"
              f"   demand {out[n]['demand']['certified']:.3f}")

    p = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_9_compilation_control.json"
    p.write_text(json.dumps({"seed": SEED, "min_entries": MIN_ENTRIES,
                             "by_ledger": out, "kill_degree_one": bool(kill)}, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
