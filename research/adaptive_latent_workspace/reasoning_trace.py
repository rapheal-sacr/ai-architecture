"""Read-only unroll instrumentation; unchanged trained ReasoningProcessor math.

No solver or halting policy is implemented. Residuals describe state motion,
not correctness. Query predictions and costs are separately recorded.
"""
import torch

def prepare(model,weights,source,task):
    b,n,_=weights.shape
    flag=torch.nn.functional.one_hot(source,n).to(weights.dtype)[...,None]
    original=torch.tanh(model.node(torch.cat([flag,model.task(task)[:,None,:].expand(-1,n,-1)],-1)))
    present=weights>0;allowed=present|torch.eye(n,device=weights.device,dtype=torch.bool)[None]
    edge=torch.stack([weights,present.to(weights.dtype)],-1)
    return dict(original=original,allowed=allowed,edge=edge,encoded_edge=model.edge(edge))

def advance(model,h,p):
    b,n,_=h.shape
    msg=model.message(torch.relu(model.send(h)[:,:,None,:]+model.receive(h)[:,None,:,:]+p['encoded_edge']))
    aggregate=msg.masked_fill(~p['allowed'][...,None],-1e9).max(1).values
    return model.norm(model.update(torch.cat([aggregate,p['original']],-1).reshape(b*n,-1),h.reshape(b*n,-1))).reshape(b,n,-1)

def decode(model,h,p):
    scores=model.score(torch.relu(model.parent(h)[:,:,None,:]+model.child(h)[:,None,:,:]+model.decode_edge(p['edge']))).squeeze(-1)
    return scores.transpose(1,2).masked_fill(~p['allowed'].transpose(1,2),-1e9)

@torch.no_grad()
def trace(model,weights,source,task,depths=(0,2,8,16,32,64,128)):
    depths=tuple(sorted(set(map(int,depths))));assert depths and depths[0]>=0
    p=prepare(model,weights,source,task);h=p['original'];selected={};residuals=[]
    if 0 in depths:selected[0]=decode(model,h,p)
    for step in range(1,max(depths)+1):
        new=advance(model,h,p);delta=(new-h).flatten(1).norm(dim=1);relative=delta/(new.flatten(1).norm(dim=1)+1e-9)
        residuals.append(dict(step=step,absolute=delta.cpu().tolist(),relative=relative.cpu().tolist()));h=new
        if step in depths:selected[step]=decode(model,h,p)
    b,n,_=weights.shape
    return dict(logits=selected,residuals=residuals,executed_message_candidates=b*n*n*max(depths),executed_decoder_candidates=b*n*n*len(depths),semantics='State motion only; no convergence or correctness guarantee')
