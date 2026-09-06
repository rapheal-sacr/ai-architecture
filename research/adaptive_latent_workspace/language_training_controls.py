"""Conventional gradient accumulation; no change to the learning procedure online."""
import math
import numpy as np,torch

def train_group(actor,records):
    assert records and actor.opt is not None
    if len(records)==1:
        result=actor.supervised_update(records[0]);return dict(**result,example_losses=[result['loss']],examples=1)
    actor.opt.zero_grad(set_to_none=True);losses=[]
    for record in records:
        logits,_=actor.logits(record['prompt'],record['actions'],'update');loss=-torch.log_softmax(logits,-1)[record['index']];assert torch.isfinite(loss);losses.append(float(loss.detach().cpu()));(loss/len(records)).backward()
    norm=torch.nn.utils.clip_grad_norm_(actor.params,actor.cfg['gradient_clip']);assert torch.isfinite(norm);actor.opt.step();return dict(loss=float(np.mean(losses)),grad_norm=float(norm),example_losses=losses,examples=len(records))
