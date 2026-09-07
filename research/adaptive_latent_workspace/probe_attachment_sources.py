"""Unchanged-source fixtures; neither a trained CTM nor Metis replication."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def extract(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ctm_probe(repo):
    ns = dict(torch=torch, F=F, np=np)
    extract(repo/'models/utils.py', ['compute_normalized_entropy'], ns)
    extract(repo/'models/ctm.py', ['forward', 'compute_certainty'], ns, 'ContinuousThoughtMachine')
    extract(repo/'tasks/parity/analysis/run.py', ['calculate_thinking_time'], ns)

    class Fixture:
        def __init__(self, ticks):
            self.iterations, self.out_dims = ticks, 2
            self.prediction_reshaper = [2]
            self.start_trace = torch.zeros(4,3)
            self.start_activated_state = torch.zeros(4)
            self.decay_params_action = torch.zeros(4)
            self.decay_params_out = torch.zeros(4)
            self.calls = Counter()
            self.compute_certainty = types.MethodType(ns['compute_certainty'], self)

        def compute_features(self, x):
            self.calls['features'] += 1
            return torch.zeros(len(x),1,4)

        def compute_synchronisation(self, state, a, b, r, synch_type):
            self.calls['sync_'+synch_type] += 1
            return state, a, b

        def q_proj(self, x):
            self.calls['query'] += 1
            return x

        def attention(self, q, k, v, **kw):
            self.calls['attention'] += 1
            return torch.zeros_like(q), torch.zeros(len(q),1,1,1)

        def synapses(self, x):
            self.calls['synapses'] += 1
            return x[:,:4]

        def trace_processor(self, x):
            self.calls['neuron_models'] += 1
            return x[:,:,-1]

        def output_projector(self, x):
            self.calls['output'] += 1
            return torch.tensor([20.,-20.]).expand(len(x),-1)

    cases = []
    for ticks in [1,4,16]:
        fixture = Fixture(ticks)
        logits, certainties, _ = ns['forward'](fixture, torch.zeros(3,1))
        entropy = certainties[:,0,:].T.numpy()[:,:,None]
        selected = ns['calculate_thinking_time'](entropy)
        first_threshold = ns['calculate_thinking_time'](entropy, finish_type='threshold')
        assert fixture.calls['attention'] == fixture.calls['output'] == ticks
        assert (selected == 0).all() and (first_threshold == 0).all()
        cases.append(dict(ticks=ticks, calls=dict(fixture.calls),
                          retrospective_index=selected.tolist(), threshold_index=first_threshold.tolist(),
                          first_certainty=certainties[0,1,0].item(),
                          output_tensor_bytes=(logits.numel()+certainties.numel())*4))
    wrong = ns['compute_certainty'](Fixture(1), torch.tensor([[20.,-20.]]))
    return dict(cases=cases, confidently_wrong_fixture=dict(target=1, prediction=0,
                reported_certainty=wrong[0,1].item()), scope='counted fixture modules in unchanged forward')


def metis_probe(repo):
    hyper = module(repo/'metis/dev_beta/metis_hyper_memory.py', 'metis_hyper_probe')
    local = module(repo/'metis/dev_beta/metis_local_memory.py', 'metis_local_probe')
    cfg = types.SimpleNamespace(backbone_configs=types.SimpleNamespace(hidden_size=2,
                num_attention_heads=1, num_key_value_heads=1, head_dim=2), memory_configs={})
    mix = hyper.GatedDeltaRuleMixin()
    mix.kv_dim, mix.update_ratio = 2, .9
    mix.W_k, mix.W_v = torch.nn.Identity(), torch.nn.Identity()
    mix.gated_delta_alpha = torch.nn.Linear(2,1).double()
    mix.gated_delta_beta = torch.nn.Linear(2,1).double()
    with torch.no_grad():
        for layer, value in [(mix.gated_delta_alpha,.9),(mix.gated_delta_beta,.8)]:
            layer.weight.zero_()
            layer.bias.fill_(math.log(value/(1-value)))
    memory = local.NormalizedDeltaNetMetisLocalMemory(cfg)
    start = torch.tensor([[[1.,0.],[0.,0.]]], dtype=torch.float64)
    mass = torch.tensor([[[1.],[0.]]], dtype=torch.float64)
    query = torch.tensor([[[[1.,0.]]]], dtype=torch.float64)
    memory.write(start.clone(), mass.clone())
    reads = [memory.read(query)[0,0,0].item()]
    write = torch.tensor([[[0.,1.]]], dtype=torch.float64)
    weights = torch.ones(1,1,dtype=torch.float64)
    errors = []
    independent, independent_mass = start[0].clone(), mass[0].clone()
    key = write[0,0]/math.sqrt(2)
    eye = torch.eye(2,dtype=torch.float64)
    with torch.no_grad():
        for _ in range(32):
            mix._apply_gated_delta_rule_update(write, weights, memory)
            independent = .9*(eye-.72*torch.outer(key,key))@independent + .72*torch.outer(key,write[0,0])
            independent_mass = .9*(eye-.72*torch.outer(key,key))@independent_mass + .72*key[:,None]
            errors += [(memory.state[0]-independent).abs().max().item(),
                       (memory.key_state[0]-independent_mass).abs().max().item()]
            reads.append(memory.read(query)[0,0,0].item())
    assert max(errors) < 1e-12 and reads[-1] < reads[0]/10
    # Paired-token order is invisible to this parallel recurrence when hidden
    # vectors/weights are held fixed. A contextual backbone may encode order.
    tokens = torch.tensor([[[1.,0.],[1.,1.]]],dtype=torch.float64)
    token_weights = torch.tensor([[.5,.5]],dtype=torch.float64)
    states, sequential = [], []
    for rev in [False,True]:
        ts = tokens.flip(1) if rev else tokens
        mem = local.NormalizedDeltaNetMetisLocalMemory(cfg)
        mem.write(start.clone(),mass.clone())
        with torch.no_grad():
            mix._apply_gated_delta_rule_update(ts, token_weights, mem)
        states.append(mem.state.clone())
        reference = start[0].clone()
        for token in ts[0]:
            key = F.normalize(token,dim=0)/math.sqrt(2)
            reference = .9*(eye-.36*torch.outer(key,key))@reference + .36*torch.outer(key,token)
        sequential.append(reference)
    parallel_gap = (states[0]-states[1]).abs().max().item()
    sequential_gap = (sequential[0]-sequential[1]).abs().max().item()
    assert parallel_gap < 1e-12 and sequential_gap > 1e-3
    return dict(independent_recurrence_max_error=max(errors), old_normalized_read_by_write=reads,
                selected_gates=dict(alpha=.9,beta=.8,update_ratio=.9),
                parallel_order_gap=parallel_gap, sequential_order_gap=sequential_gap,
                scope='shared mixin and normalized local read; fixed supplied hidden vectors and gates')


def utility_probe():
    rng = np.random.default_rng(37201)
    n, actions, outcomes, drives = 10000, 5, 3, 4
    probabilities = rng.dirichlet(np.ones(outcomes), size=(n,actions))
    desire = rng.normal(size=(n,actions,outcomes,drives))
    weights = rng.dirichlet(np.ones(drives), size=n)
    old_weights = rng.dirichlet(np.ones(drives), size=n)
    vector_expected = np.sum(probabilities[:,:,:,None]*desire, axis=2)
    vector_score = np.sum(vector_expected*weights[:,None,:], axis=2)
    scalar_outcomes = np.sum(desire*weights[:,None,None,:], axis=3)
    scalar_score = np.sum(probabilities*scalar_outcomes, axis=2)
    old_score = np.sum(vector_expected*old_weights[:,None,:], axis=2)
    exact = vector_score.argmax(1) == scalar_score.argmax(1)
    assert exact.all()
    return dict(cases=n, actions=actions, outcomes=outcomes, drives=drives,
                choice_equivalences=int(exact.sum()), max_score_error=float(abs(vector_score-scalar_score).max()),
                stale_scalar_choice_changes=int((old_score.argmax(1)!=vector_score.argmax(1)).sum()),
                scope='known model algebra; no learned representation comparison')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repos', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    pins = {'Metis':'22f7aabc8d3e9c39fcea676b11a10007bc8b3748',
            'continuous-thought-machines':'4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466'}
    for name, sha in pins.items():
        assert subprocess.check_output(['git','-C',str(args.repos/name),'rev-parse','HEAD'],text=True).strip() == sha
        assert not subprocess.check_output(['git','-C',str(args.repos/name),'diff','--name-only'],text=True).strip()
    result = dict(pins=pins, own_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ctm=ctm_probe(args.repos/'continuous-thought-machines'),
                  metis=metis_probe(args.repos/'Metis'), utility=utility_probe(), status='complete')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
