"""Untrained recurrent graph-reasoning component; no result claim.

Uses a shared message-passing/GRU core with input reinjection. This is established
neural algorithmic-reasoning machinery, not a novel architecture by itself.
Forward inputs exclude intermediate hints and label-derived stopping times.
"""
import torch
from torch import nn
class ReasoningProcessor(nn.Module):
    def __init__(self,hidden=32,tasks=2):
        super().__init__();self.hidden=hidden;self.task=nn.Embedding(tasks,4);self.node=nn.Linear(5,hidden)
        self.send=nn.Linear(hidden,hidden);self.receive=nn.Linear(hidden,hidden);self.edge=nn.Linear(2,hidden);self.message=nn.Linear(hidden,hidden)
        self.update=nn.GRUCell(2*hidden,hidden);self.norm=nn.LayerNorm(hidden)
        self.parent=nn.Linear(hidden,hidden);self.child=nn.Linear(hidden,hidden);self.decode_edge=nn.Linear(2,hidden);self.score=nn.Linear(hidden,1)
    def forward(self,weights,source,task,steps=None):
        b,n,_=weights.shape;steps=n if steps is None else int(steps);assert steps>=0
        flag=torch.nn.functional.one_hot(source,n).to(weights.dtype)[...,None]
        original=torch.tanh(self.node(torch.cat([flag,self.task(task)[:,None,:].expand(-1,n,-1)],-1)))
        h=original;present=weights>0;allowed=present|torch.eye(n,device=weights.device,dtype=torch.bool)[None]
        edge=torch.stack([weights,present.to(weights.dtype)],-1);encoded_edge=self.edge(edge)
        for _ in range(steps):
            msg=self.message(torch.relu(self.send(h)[:,:,None,:]+self.receive(h)[:,None,:,:]+encoded_edge))
            aggregate=msg.masked_fill(~allowed[...,None],-1e9).max(1).values
            h=self.norm(self.update(torch.cat([aggregate,original],-1).reshape(b*n,-1),h.reshape(b*n,-1))).reshape(b,n,-1)
        scores=self.score(torch.relu(self.parent(h)[:,:,None,:]+self.child(h)[:,None,:,:]+self.decode_edge(edge))).squeeze(-1)
        return scores.transpose(1,2).masked_fill(~allowed.transpose(1,2),-1e9)
    def counts(self,batch,n,steps=None):
        steps=n if steps is None else steps
        return dict(processor_steps=steps,dense_message_candidates=batch*n*n*steps,decoder_candidates=batch*n*n,
            parameters=sum(p.numel() for p in self.parameters()),input_weight_bytes=batch*n*n*4,latent_bytes=batch*n*self.hidden*4)
