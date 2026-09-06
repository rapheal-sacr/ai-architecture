"""Execute unchanged, isolated upstream MSA method bodies on CPU fixtures.

Not a 4B model run, benchmark replication or upstream training reproduction.
"""
import __future__
import argparse
import ast
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def method(path, name):
    tree = ast.parse(path.read_text())
    nodes = [n for c in tree.body if isinstance(c, ast.ClassDef)
             for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    node = nodes[0]
    assert not node.decorator_list
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "F": F, "os": os}
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name], {"file": str(path), "line": node.lineno,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "body_ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest()}


def main(repo, work, out):
    start = time.perf_counter()
    torch.set_num_threads(1)
    repo, work = Path(repo), Path(work)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert commit == "77fbdfde88e150cd91307fd710067cf06828cfdc"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    pooling, pool_src = method(repo / "src/msa/memory_sparse_attention.py", "sequence_pooling_kv")
    routing, route_src = method(repo / "src/msa/memory_sparse_attention.py", "_calculate_routing_scores_adaptive")
    reuse, reuse_src = method(repo / "src/msa_service.py", "generate_blocks")

    # Identical pooled state, different key/value associations. These are
    # permissible attention-layer inputs, not asserted reachable text encodings.
    keys = torch.tensor([[[[1., 0.], [-1., 0.]]]])
    va = torch.tensor([[[[1., 0.], [0., 1.]]]])
    vb = va.flip(2)
    indices = torch.tensor([[0, 0], [0, 1]])
    chunk_ids = torch.tensor([0, 0])
    ka, pa = pooling(None, keys, va, indices, chunk_ids)
    kb, pb = pooling(None, keys, vb, indices, chunk_ids)
    assert torch.equal(ka, kb) and torch.equal(pa, pb)
    q = torch.tensor([10., 0.])
    probs = (keys[0, 0] @ q).softmax(-1)
    answer_a, answer_b = probs @ va[0, 0], probs @ vb[0, 0]
    assert (answer_a - answer_b).abs().max() > .99
    collision = {"keys": keys.tolist(), "values_a": va.tolist(), "values_b": vb.tolist(),
        "pooled_keys_equal": True, "pooled_values_equal": True,
        "pooled_values": pa.tolist(), "unpooled_answer_a": answer_a.tolist(),
        "unpooled_answer_b": answer_b.tolist(),
        "scope": "Noninjective pooling at the KV interface. Does not measure trained MSA text accuracy or rule out upstream contextual encoding of associations."}

    # A common sign change of both router projections preserves every fresh
    # cosine similarity, but mixing new queries with old cached keys reverses it.
    router = SimpleNamespace(decouple_router=True, aux_loss_method="INFONCE",
                              head_reduce_method="mean", query_reduce_method="max", scaling=1.0)
    qr = torch.tensor([[[[1., 0.]]]])
    kr = torch.tensor([[[[1., 0.]], [[-1., 0.]]]])
    qm, km = torch.ones((1, 1), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool)
    original = routing(router, qr, kr, qm, km)
    refreshed = routing(router, -qr, -kr, qm, km)
    stale = routing(router, -qr, kr, qm, km)
    assert torch.equal(original, refreshed)
    assert original.argmax(-1).item() == 0 and stale.argmax(-1).item() == 1
    coordinate = {"old_scores": original.tolist(), "fresh_both_changed_scores": refreshed.tolist(),
        "new_query_old_cache_scores": stale.tolist(), "new_and_old_fresh_functions_identical": True,
        "scope": "Constructed representation-coordinate change executed through upstream routing. Not an observed checkpoint update or learned-model failure rate."}

    # Only the unchanged cache-admission control flow is executed. Deserialization
    # and postprocessing are fixtures, not the full GPU worker implementation.
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="msa-cache-audit-", dir=work) as tmp:
        path = Path(tmp) / "gpu1/0"
        path.mkdir(parents=True)
        old_docs = ["alpha has code OLD", "beta has code KEEP"]
        new_docs = ["alpha has code NEW", "beta has code KEEP"]
        (path / "fixture.json").write_text(json.dumps({"docs": old_docs, "encoder": "old"}))
        fixture = SimpleNamespace(generate_config=SimpleNamespace(world=1), gpu_id=0,
            model_config=SimpleNamespace(model_path="new-encoder"), rebuilt=False, postprocessed=False)
        def deserialize(p):
            saved = json.loads((Path(p) / "fixture.json").read_text())
            fixture.block_desc = SimpleNamespace(nr_docs=len(saved["docs"]), docs=saved["docs"])
            fixture.loaded_encoder = saved["encoder"]
        def postprocess():
            fixture.postprocessed = True
        def rebuild():
            fixture.rebuilt = True
            raise AssertionError("Unexpected rebuild in same-count reuse branch")
        fixture.deserialize, fixture._post_process, fixture._start_worker = deserialize, postprocess, rebuild
        old_env = os.environ.get("MEMORY_DATA_PATH")
        os.environ["MEMORY_DATA_PATH"] = tmp
        try:
            reuse(fixture, new_docs)
            assert fixture.postprocessed and not fixture.rebuilt
            assert fixture.block_desc.docs == old_docs and fixture.block_desc.docs != new_docs
            mismatch_rejected = False
            try:
                reuse(fixture, new_docs + ["extra document"])
            except AssertionError:
                mismatch_rejected = True
            assert mismatch_rejected
        finally:
            if old_env is None:
                del os.environ["MEMORY_DATA_PATH"]
            else:
                os.environ["MEMORY_DATA_PATH"] = old_env
        admission = {"requested_docs": new_docs, "loaded_docs": fixture.block_desc.docs,
            "requested_encoder": "new-encoder", "loaded_encoder": fixture.loaded_encoder,
            "same_document_count_accepted_without_rebuild": True, "different_count_rejected": True,
            "scope": "Opt-in MEMORY_DATA_PATH admission branch; fake loader/postprocessor isolate its validation. No GPU/model inference or wrong-answer rate measured."}
    report = {"event": "MSA_SOURCE_COUNTEREXAMPLES_EXECUTED", "commit": commit,
        "sources": {"pooling": pool_src, "routing": route_src, "reuse_gate": reuse_src},
        "pooling_collision": collision, "coordinate_staleness": coordinate, "cache_admission": admission,
        "runtime": {"torch": torch.__version__, "device": "cpu", "threads": 1},
        "seconds": time.perf_counter() - start,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.repo, args.work, args.out)
