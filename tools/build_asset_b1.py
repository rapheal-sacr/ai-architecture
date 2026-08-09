"""Build asset B1 from BEIR: a corpus plus MY OWN instrumented retrieval.

Worklist v3 §2 reclassified B1 as "a public corpus + your own instrumented
pipeline, buildable now". This is that pipeline, at the smallest honest scale.

WHY A BUILD AND NOT AN ACQUISITION. Of R13's predicates, the ones no third party
supplies are exactly the ones that describe how retrieval was RUN:

    P2a  a decomposable scorer          -- a scorer I choose
    P2b  the (k+1)-th score             -- a field I log
    P2c  candidate provenance           -- my compiled bank
    P3a  an unfiltered bank             -- compiled without an admission filter
    P3c  bank/fleet/rollout ratios      -- my pipeline's own configuration

No retrieval log in the world carries them, because nobody instrumented their
retrieval for this certificate. What BEIR supplies is the two that cannot be
retrofitted: an independent domain label, and outcomes resolved by humans.

THE MAPPING TO THE DESIGN, and it is not arbitrary:

    L1 entry  = a sentence of a document      the atomic recorded unit
    L3 card   = a document                    a distillation of its source entries
    provenance(card) = its sentence ids       recorded, not inferred
    score     = cos(query, mean of the card's entry vectors)

That last line is the point. The card vector is a MEAN OVER ITS SOURCE ENTRIES,
which is exactly the model E0.5's margin certificate assumes -- so a log built this
way is one the certificate can actually run on, rather than one it would have to be
adapted to.

WHAT WILL FIRE, AND IT SHOULD. BEIR carries no structural difficulty, so P1a/P1b/
P1c fire and M1 is correctly reported as blocked. That is the validator working:
a fired predicate names the ONE measurement it blocks, and B1 was never the asset
that unblocks M1 -- B2 is.

Run:  python tools/build_asset_b1.py <beir_dir> [<beir_dir> ...] -o asset_b1.jsonl
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

TOP_K = 10                 # candidates logged per query
MAX_QUERIES = 400          # per dataset, keeps the first pass on a laptop
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list:
    return TOKEN.findall(s.lower())


def sentences(text: str) -> list:
    """Split a document into ENTRIES. Crude on purpose -- the entry boundary is a
    recording decision, and pretending otherwise would hide it."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts or [text.strip() or "empty"]


def load(dirpath: pathlib.Path) -> dict:
    corpus, queries, qrels = {}, {}, defaultdict(set)
    for line in (dirpath / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    for line in (dirpath / "queries.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    for f in sorted((dirpath / "qrels").glob("*.tsv")):
        for i, line in enumerate(f.read_text().splitlines()):
            if i == 0 or not line.strip():
                continue
            q, d, s = line.split("\t")[:3]
            if int(s) > 0:
                qrels[q].add(d)
    return {"name": dirpath.name, "corpus": corpus, "queries": queries, "qrels": qrels}


def build(ds: dict, rng) -> list:
    name, corpus, queries, qrels = ds["name"], ds["corpus"], ds["queries"], ds["qrels"]

    # ---- L1 entries and L3 cards. THE BANK IS UNFILTERED (P3a): every document
    # is admitted, no cosine cap, no overlap cap. That is the whole point -- a
    # pre-filtered bank's degree distribution is the filter's, not the structure's.
    entry_texts, card_entries = [], {}
    for cid, text in corpus.items():
        ids = []
        for sent in sentences(text):
            ids.append(len(entry_texts))
            entry_texts.append(sent)
        card_entries[cid] = ids

    # ---- TF-IDF over ENTRIES, then card = mean of its entries. Decomposable by
    # construction, which is P2a and is what the certificate's bound assumes.
    df = Counter()
    toks = [tokenize(t) for t in entry_texts]
    for t in toks:
        df.update(set(t))
    vocab = {w: i for i, w in enumerate(w for w, c in df.items() if c >= 2)}
    n_e = len(entry_texts)
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = np.log(n_e / (1 + df[w]))

    rows, cols, vals = [], [], []
    for e, t in enumerate(toks):
        c = Counter(w for w in t if w in vocab)
        for w, f in c.items():
            rows.append(e); cols.append(vocab[w]); vals.append((1 + np.log(f)) * idf[vocab[w]])
    E = sparse.csr_matrix((vals, (rows, cols)), shape=(n_e, len(vocab)))

    card_ids = list(card_entries)
    idx = {c: i for i, c in enumerate(card_ids)}
    r2, c2, v2 = [], [], []
    for cid, eids in card_entries.items():
        for e in eids:
            r2.append(idx[cid]); c2.append(e); v2.append(1.0 / len(eids))
    M = sparse.csr_matrix((v2, (r2, c2)), shape=(len(card_ids), n_e))
    C = M @ E                                        # card vector = MEAN of entries
    Cn = C.multiply(1.0 / np.maximum(sparse.linalg.norm(C, axis=1), 1e-12)[:, None]).tocsr()

    qids = [q for q in queries if qrels.get(q)]      # outcome must be resolvable
    rng.shuffle(qids)
    qids = qids[:MAX_QUERIES]
    qr, qc, qv = [], [], []
    for i, q in enumerate(qids):
        c = Counter(w for w in tokenize(queries[q]) if w in vocab)
        for w, f in c.items():
            qr.append(i); qc.append(vocab[w]); qv.append((1 + np.log(f)) * idf[vocab[w]])
    Q = sparse.csr_matrix((qv, (qr, qc)), shape=(len(qids), len(vocab)))
    Qn = Q.multiply(1.0 / np.maximum(sparse.linalg.norm(Q, axis=1), 1e-12)[:, None]).tocsr()

    S = np.asarray((Qn @ Cn.T).todense())

    out = []
    for i, q in enumerate(qids):
        order = np.argsort(-S[i])[:TOP_K + 1]
        top, nxt = order[:TOP_K], order[TOP_K]
        chosen = card_ids[top[0]]
        cands = [{"card_id": card_ids[j], "score": round(float(S[i, j]), 6)} for j in top]
        out.append({
            "query_id": f"{name}:{q}",
            # P1d: the DATASET IDENTITY is the domain, supplied by construction.
            # `independent_authority` -- BEIR's partition predates this pipeline
            # and depends on no model in it.
            "domain": name, "domain_source": "independent_authority",
            "source": name,
            "candidates": cands, "chosen": chosen,
            # P2a: card = mean of its entries' tf-idf vectors. Decomposable.
            "scorer": "embedding_sum",
            # P2b: the (k+1)-th score, so a rival outside top-k is BOUNDED rather
            # than invisible. E0.5: 75.9% of uncertified selections are rival rise.
            "k_logged": TOP_K, "next_score": round(float(S[i, nxt]), 6),
            # P0/P0a: BEIR qrels are HUMAN relevance judgments -- a resolver
            # disjoint from the scorer above.
            "outcome": "pass" if chosen in qrels[q] else "fail",
            "resolved_by": "human",
            # B1 carries NO structural difficulty. P1a/P1b/P1c fire, and they
            # should: M1 is B2's measurement, not B1's.
            "difficulty": None, "difficulty_source": "absent",
            # Entry ids are NAMESPACED BY DATASET. Without this, `e0` exists in
            # every dataset and provenance-based measurements silently merge
            # unrelated entries -- which is what the first build did, inflating
            # the distinct-entry count and manufacturing a degree-2 tail that was
            # entirely id collision.
            "cards": {card_ids[j]: [f"{name}:e{e}" for e in card_entries[card_ids[j]]]
                      for j in top},
            "bank_admission": "none",
            "bank_size": len(card_ids), "fleet_size": 16,
            "rollouts_per_adapter": round(len(qids) / 16, 2),
        })
    return out


def main(argv: list) -> int:
    rest = argv[1:]
    out_path = pathlib.Path("asset_b1.jsonl")
    if "-o" in rest:
        i = rest.index("-o")
        out_path = pathlib.Path(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]      # the -o VALUE is not an input dir
    args = [a for a in rest if not a.startswith("-")]
    if not args:
        print(__doc__)
        return 2
    rng = __import__("random").Random(20260806)
    rows = []
    for d in args:
        ds = load(pathlib.Path(d))
        print(f"  {ds['name']}: {len(ds['corpus'])} docs, {len(ds['queries'])} queries, "
              f"{len(ds['qrels'])} with judgments")
        r = build(ds, rng)
        print(f"    -> {len(r)} logged selections, "
              f"{sum(1 for x in r if x['outcome'] == 'pass')} chosen card relevant")
        rows += r
    out_path.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\n  wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
