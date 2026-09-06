"""Narrow source fixtures and analytic resource accounting, no model benchmark.

Execute unchanged AST function bodies from pinned clones; omit only optional
kernel-dispatch decorators. Do not import/download giant model weights.
"""
import argparse
import ast
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(path, name):
    source = Path(path).read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    original = ast.get_source_segment(source, node)
    node.decorator_list = []
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name], {"path": str(path), "file_sha256": sha(path), "function": name,
                             "line": node.lineno, "body_sha256": hashlib.sha256(original.encode()).hexdigest()}


def main(repos, out):
    start = time.perf_counter()
    torch.set_num_threads(1)
    repos = Path(repos)
    tr = repos/"transformers-2026-09-06"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tr, text=True).strip() == "c93057d4835cd31752bb56f59989dd27696eb45b"
    gdn, gs = function(tr/"src/transformers/models/qwen3_5/modeling_qwen3_5.py", "torch_recurrent_gated_delta_rule")
    kda, ks = function(tr/"src/transformers/models/glm5_next/modeling_glm5_next.py", "recurrent_kimi_delta_attention")
    cache_idx, ns = function(repos/"Nanbeige4.2-3B/modeling_nanbeige.py", "_get_loop_cache_layer_idx")
    rows = []
    # Begin with a known distinction in state row0. Only orthogonal row1 is
    # subsequently written. This is a recurrence-interface fixture, not learned
    # tokens or prior acquisition in a released model.
    for n in (1, 32, 256, 1024):
        q = torch.zeros(1, n, 1, 2)
        q[..., 0] = math.sqrt(2)
        key = torch.zeros_like(q)
        key[..., 1] = 1
        values = torch.zeros(1, n, 1, 1)
        beta = torch.full((1, n, 1), .5)
        initial = torch.zeros(1, 1, 2, 1)
        initial[..., 0, 0] = 1
        scalar_decay = torch.full((1, n, 1), math.log(.99))
        channel_decay = torch.empty_like(q)
        channel_decay[..., 0] = math.log(.999)
        channel_decay[..., 1] = math.log(.99)
        out_gdn, state_gdn = gdn(q, key, values, scalar_decay, beta,
                                initial_state=initial.clone(), output_final_state=True)
        out_kda, state_kda = kda(q, key, values, channel_decay, beta,
                                initial_state=initial.clone(), output_final_state=True)
        for value, expected in ((state_gdn[0, 0, 0, 0], .99**n), (state_kda[0, 0, 0, 0], .999**n)):
            assert math.isclose(float(value), expected, rel_tol=3e-5, abs_tol=1e-7)
        rows.append({"orthogonal_writes": n, "scalar_gate_old_component": float(state_gdn[0, 0, 0, 0]),
                     "channel_gate_old_component": float(state_kda[0, 0, 0, 0]),
                     "scalar_read": float(out_gdn[0, -1, 0, 0]), "channel_read": float(out_kda[0, -1, 0, 0]),
                     "state_tensor_bytes_each": state_gdn.numel()*state_gdn.element_size()})
    nc = json.loads((repos/"Nanbeige4.2-3B/config.json").read_text())
    indices = [cache_idx(layer, loop, nc["num_hidden_layers"]) for loop in range(nc["num_loops"])
               for layer in range(nc["num_hidden_layers"])]
    assert indices == list(range(44))
    nc_bytes = len(indices)*2*nc["num_key_value_heads"]*nc["head_dim"]*2
    assert nc_bytes == 180224
    qc = json.loads((repos/"Qwen3.8-27B/config.json").read_text())["text_config"]
    qmix = Counter(qc["layer_types"])
    q_state = qmix["linear_attention"]*qc["linear_num_value_heads"]*qc["linear_key_head_dim"]*qc["linear_value_head_dim"]*4
    q_slope = qmix["full_attention"]*2*qc["num_key_value_heads"]*qc["head_dim"]*2
    gc = json.loads((repos/"GLM-5.3-Flash/config.json").read_text())["text_config"]
    gmix = Counter(gc["layer_types"])
    linear = gc["linear_attn_config"]
    g_state = gmix["linear_attention"]*linear["num_heads"]*linear["head_dim"]**2*4
    g_slope = gmix["deepseek_sparse_attention"]*gc["kv_lora_rank"]*2
    dc = json.loads((repos/"DeepSeek-V4-Pro/config.json").read_text())
    ratios = dc["compress_ratios"][:dc["num_hidden_layers"]]
    assert len(ratios) == 61 and set(ratios) == {4, 128}
    d_main = sum(dc["head_dim"]*2/r for r in ratios)
    d_index = sum(dc["index_head_dim"]*2/r for r in ratios if r == 4)
    mc = json.loads((repos/"Muse-Glimmer-30B/config.json").read_text())["text_config"]
    mmix = Counter(mc["layer_types"])
    m_per_layer = 2*mc["num_key_value_heads"]*mc["head_dim"]*2
    assert mmix == {"sliding_attention": 39, "full_attention": 13}
    assert all(bool(theta) == (kind == "sliding_attention") for theta, kind in zip(mc["layer_rope_theta"], mc["layer_types"]))
    m_global = mmix["full_attention"]*m_per_layer
    # Current DynamicSlidingWindowLayer retains window-1 past entries. The
    # current chunk and attention intermediates are extra, transient storage.
    m_local = mmix["sliding_attention"]*(mc["sliding_window"]-1)*m_per_layer
    configs = {}
    for repo in ("GLM-5.3-Flash", "DeepSeek-V4-Pro", "Nanbeige4.2-3B", "Qwen3.8-27B", "Kimi-K3", "Muse-Glimmer-30B"):
        path = repos/repo
        configs[repo] = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip(),
                         "files": {name: sha(path/name) for name in ("config.json", "README.md", "k3_tech_report.pdf") if (path/name).exists()}}
    result = {"event": "FRONTIER_SOURCE_PROBES_COMPLETE", "sources": [gs, ks, ns], "repositories": configs,
              "additional_implementation_hashes": {str(tr/name): sha(tr/name) for name in (
                  "src/transformers/models/muse_glimmer/modeling_muse_glimmer.py", "src/transformers/cache_utils.py")},
              "recurrence_fixture": rows, "nanbeige_loop_cache_indices": indices,
              "derived_bytes_batch1_main_decoder_no_MTP": {
                  "nanbeige_bf16_KV_per_token": nc_bytes,
                  "qwen_fp32_recurrent_state_only": q_state, "qwen_bf16_attention_KV_per_token": q_slope,
                  "glm_fp32_recurrent_state_only": g_state, "glm_bf16_MLA_latents_per_token": g_slope,
                  "deepseek_bf16_compressed_attention_KV_per_token_asymptotic": d_main,
                  "deepseek_bf16_indexer_KV_per_token_asymptotic_extra": d_index,
                  "deepseek_compression_layer_counts": dict(Counter(ratios)),
                  "muse_bf16_global_KV_per_token": m_global,
                  "muse_bf16_local_persistent_KV_at_window_minus_one": m_local,
                  "muse_naive_all_layers_unbounded_KV_per_token": mc["num_hidden_layers"]*m_per_layer},
              "scope": "Unchanged PyTorch recurrence/helper bodies plus configuration-derived logical storage; no released weights, trained hidden-state reachability, accuracy, full runtime peak memory or throughput measured. Dispatch decorators removed; optional L2 branch unused. Convolution/state buffers, model weights, indexes other than explicitly counted, vision and MTP additional.",
              "seconds": time.perf_counter()-start}
    Path(out).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.repos, args.out)
