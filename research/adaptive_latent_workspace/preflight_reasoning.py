from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch,networkx as nx
from clrs_reference import Reference,ALGORITHMS,validate_tree
from reasoning_processor import ReasoningProcessor

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    ref=Reference(a.repo);rng=np.random.default_rng(117);checks=0
    for n in (6,8,16):
        for disconnected in (False,True):
            for _ in range(8):
                w=rng.uniform(.1,1.,(n,n));w=np.triu(w,1);w=w+w.T;mask=np.triu(rng.random((n,n))<.35,1);mask=mask|mask.T;weights=w*mask
                if disconnected:weights[:n//2,n//2:]=0;weights[n//2:,:n//2]=0
                source=int(rng.integers(n))
                for alg in ALGORITHMS:
                    pi,record=ref(alg,weights,source);assert record['hint_count']>0;validate_tree(weights,source,pi,alg);checks+=1
    torch.manual_seed(118);model=ReasoningProcessor();n=16;chain=np.zeros((n,n),dtype=np.float32)
    for i in range(n-1):chain[i,i+1]=chain[i+1,i]=float(rng.uniform(.1,1.))
    weights=torch.tensor(chain)[None];source=torch.tensor([0]);task=torch.tensor([0]);logits=model(weights,source,task)
    pi,_=ref('dijkstra',chain,0);loss=torch.nn.functional.cross_entropy(logits.reshape(n,n),torch.tensor(pi));loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    perm=torch.randperm(n);inverse=torch.argsort(perm);permuted=model(weights[:,perm][:,:,perm],inverse[source],task)
    assert torch.allclose(permuted,logits[:,perm][:,:,perm],atol=2e-6,rtol=1e-6)
    short_left=model(weights,torch.tensor([0]),task,steps=2);short_right=model(weights,torch.tensor([15]),task,steps=2)
    assert torch.equal(short_left[:,7],short_right[:,7]),'Outside-receptive-field sources must be indistinguishable'
    left,_=ref('dijkstra',chain,0);right,_=ref('dijkstra',chain,15);assert left[7]==6 and right[7]==8
    # Two executions with the same explicit budget; no reference hint data enters forward.
    repeat=model(weights,source,task,steps=16);assert torch.equal(logits,repeat)
    a.out.mkdir(parents=True,exist_ok=True);files=('clrs_reference.py','reasoning_processor.py','preflight_reasoning.py')
    result=dict(source=ref.identity,code_identity={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in files},
        networkx_version=nx.__version__,torch_version=torch.__version__,reference_cases=checks,independent_tree_validation=True,
        finite_graph_gradient=True,permutation_equivariance=True,short_depth_indistinguishability_counterexample=dict(node=7,left_parent=int(left[7]),right_parent=int(right[7]),steps=2),
        repeated_public_size_budget_exact=True,model_counts=model.counts(1,16),status='PREFLIGHT_ONLY_NO_TRAINED_REASONING_RESULT')
    (a.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
