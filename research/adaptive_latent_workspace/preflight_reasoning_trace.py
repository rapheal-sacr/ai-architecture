from pathlib import Path
import argparse,json,hashlib
import torch
from reasoning_processor import ReasoningProcessor
from reasoning_trace import trace,prepare,advance,decode

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1471);model=ReasoningProcessor();checks=[]
    for n in (8,16,32):
        w=torch.rand(2,n,n);w=torch.triu(w,1);w=w+w.transpose(1,2);chain=torch.zeros(n,n);chain[torch.arange(n-1),torch.arange(1,n)]=1;chain=chain+chain.T;w[1]*=chain;s=torch.tensor([0,n-1]);t=torch.tensor([0,1]);before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();depths=(0,2,8,16,32);result=trace(model,w,s,t,depths)
        for d in depths:
            native=model(w,s,t,d);assert torch.equal(result['logits'][d],native);checks.append(dict(nodes=n,depth=d,all_logits_bitwise_equal=True))
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());assert torch.equal(rng,torch.get_rng_state());assert len(result['residuals'])==max(depths);assert result['executed_message_candidates']==2*n*n*max(depths);assert result['executed_decoder_candidates']==2*n*n*len(depths)
    # Independent residual arithmetic on a single transition.
    p=prepare(model,w,s,t);h=p['original'];new=advance(model,h,p);got=trace(model,w,s,t,(1,));expected=(new-h).flatten(1).norm(dim=1)/(new.flatten(1).norm(dim=1)+1e-9);assert got['residuals'][0]['relative']==expected.tolist()
    a.out.mkdir(parents=True,exist_ok=True);files=['reasoning_processor.py','reasoning_trace.py','preflight_reasoning_trace.py'];result=dict(status='PREFLIGHT_ONLY_NO_TRAINED_STABILITY_RESULT',device='cpu',torch=torch.__version__,code_identity={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in files},checks=checks,model_rng_unchanged=True,residual_arithmetic_exact=True,reused_prefix_costs_verified=True)
    (a.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
