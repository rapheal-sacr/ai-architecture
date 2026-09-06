# MSA source audit: compressed evidence has a representation lifetime

Read FALSIFICATION.md first. Source cloned with history from
https://github.com/EverMind-AI/MSA at
`77fbdfde88e150cd91307fd710067cf06828cfdc`, WD `repos/MSA`.
GitHub metadata identified the repository; no rendered page was fetched. README
identifies the maintainers as the paper's authors. The supplied msa.pdf and
prior review motivated this refresh. Inspected README/QUICK_START, attention
dispatch, pooling/routing, source-text injection, and cache admission/metadata.
This is a targeted audit, not a full source or benchmark reproduction.

The paper reports158.95B continual-pretraining tokens with routing supervision
and subsequent SFT. No4B weights, Flash Attention GPU run, training corpus or
LLM judge was executed locally. Those training costs are not a free module swap.

## Evidence path

`sequence_pooling_kv` averages already contextualized K/V via float32 cumulative
sums, then casts back. Routing compares query projections with the bank's keys.
Top-k bounds selected content, not the existence or scoring cost of all routing
keys. This is not constant storage for an indefinitely growing corpus.

`src/msa/generate.py:191` also turns retrieved IDs into original document text
through `idx_to_doc`, then tokenizes that text for generation. The engine's
`get_idx_to_doc` returns IDs mapped to `doc.doc`. This source's evidence path
therefore retains text and performs another read, beyond compressed KV. That
useful choice must remain in storage, transfer and inference cost comparisons.

## Executed probes and limitations

`audit_msa_source.py` executes unchanged AST method bodies on CPU fixtures.
Full-file/body hashes and outputs are in `results/msa_source_audit.json`.
These isolated calls are not a complete upstream model execution.

1. **Pooling loses an association.** Inputs have keys(1,0),(-1,0), with swapped
   values(1,0),(0,1). Upstream pooling returns identical key means and value
   means(.5,.5). Unpooled attention with query(10,0) returns approximately(1,0)
   versus(0,1). This proves noninjectivity at the KV interface. It does not
   establish reachability from text in the trained model: earlier contextual
   encoding can carry associations and source-text reinjection can restore detail.

2. **Fresh-equivalent coordinates break old-cache access.** Upstream routing
   gives two unit keys scores(1,-1). Negating both query/key projections preserves
   fresh scores exactly. A new query against old cached keys gives(-1,1), reversing
   selection. The fresh routing function lost no semantics; the versions are
   incompatible. This is a constructed intervention, not a measured checkpoint
   update or estimate of trained-model failure frequency.

3. **Document count does not validate content.** With optional MEMORY_DATA_PATH,
   unchanged `generate_blocks` loads an existing cache, checks only the number
   of documents, postprocesses and returns. A fixture requesting `alpha has code
   NEW` accepts cached `alpha has code OLD` at the same count; changing the count
   is rejected. Loader/postprocessor are explicit fixtures isolating admission,
   not executed GPU workers. The examined metadata/admission path does not
   verify a content/encoder fingerprint. Default operation without this optional
   cache path is not implicated. No downstream wrong-answer rate was measured.

These distinguish root2 (lost distinctions), root3 (access through incompatible
coordinates) and stale source validity. They do not refute the published scores
or establish a superior architecture.

## E2E integration consequence

Do not attach a long-lived KV bank to a changing encoder and assume compatibility.
A concrete candidate boundary is to encode persistent evidence at a fixed prefix
before any fast-adaptive layer and inject it at that boundary. Inner updates then
cannot change the stored representation's producer. Later-layer cache features
may depend on earlier fast weights and do not inherit this property. Stable
coordinates still do not guarantee correct routing or sufficient features.

This is a proposed integration constraint, not an implemented memory module or
novelty claim. A fixed producer limits representation learning. Slow encoder
updates require re-encoding, proven transport or explicitly tolerated stale/lost
entries. Version metadata should cover content, tokenizer, encoder and pooling
layout. A hash detects incompatibility; it does not repair it. Exact arbitrary
re-encoding needs original evidence or a sufficient lossless representation.
Learned lossy summaries cannot generally reconstruct removed distinctions.

E33 remains unchanged: persistent gradient learning without an external KV bank.
Its acquisition/return results are required before crediting extra memory with
repairing an unacquired capability. CONSTRAINTS.md adds the conditional cost of
representation maintenance; no universal improvement-rate bound is asserted.
