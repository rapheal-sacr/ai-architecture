"""Historical bootstrap replay and independent mean-gradient preflight."""
from pathlib import Path
import argparse,copy,hashlib,json,time,math
import numpy as np,torch
from language_delivery_agent import LanguageActor
from language_training_controls import train_group
from preflight_language_adapter import base_hash,adapter_state
from e28_depth import state_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();source=json.loads((a.source/'complete.json').read_text());assert source['status']=='E30_COMPLETE';boot=next(b for b in source['bootstrap'] if b['seed']==30101);cfg=source['manifest']['config'];assert sha(boot['checkpoint'])==boot['checkpoint_sha256'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);actor=LanguageActor(a.model,cfg,30101);initial=actor.state();base=base_hash(actor.model);assert base==boot['base_hash'];actor.start_optimizer(cfg['bootstrap_learning_rate']);rng=np.random.default_rng(30101+800000);trace=[];start=time.perf_counter()
    for _ in range(cfg['bootstrap_updates']):
        r=train_group(actor,[boot['examples'][int(rng.integers(len(boot['examples'])))]]);trace.append({k:r[k] for k in ('loss','grad_norm')})
    torch.cuda.synchronize();replay_seconds=time.perf_counter()-start;saved=torch.load(boot['checkpoint'],map_location='cpu',weights_only=False);assert trace==boot['training_trace'];assert state_hash(actor.state())==state_hash(saved);print(json.dumps(dict(event='E32_HISTORICAL_BOOTSTRAP_REPLAY',updates=len(trace),exact=True,seconds=replay_seconds)),flush=True)
    records=boot['examples'][:4];actor.load_state(initial);actor.start_optimizer(cfg['bootstrap_learning_rate']);accum=train_group(actor,records);accum_grad=[p.grad.detach().cpu().clone() for p in actor.params];accum_params=[p.detach().cpu().clone() for p in actor.params]
    actor.load_state(initial);actor.start_optimizer(cfg['bootstrap_learning_rate']);actor.opt.zero_grad(set_to_none=True);losses=[];torch.cuda.reset_peak_memory_stats()
    for record in records:
        logits,_=actor.logits(record['prompt'],record['actions'],'update');losses.append(-torch.log_softmax(logits,-1)[record['index']])
    joint=actor.finish_update(torch.stack(losses).mean());grad=[p.grad.detach().cpu().clone() for p in actor.params];max_grad=max(float((x-y).abs().max()) for x,y in zip(accum_grad,grad));max_parameter=max(float((x-p.detach().cpu()).abs().max()) for x,p in zip(accum_params,actor.params));assert all(torch.allclose(x,y,atol=1e-6,rtol=1e-4) for x,y in zip(accum_grad,grad));assert math.isclose(accum['loss'],joint['loss'],abs_tol=1e-5,rel_tol=1e-5);assert math.isclose(accum['grad_norm'],joint['grad_norm'],abs_tol=1e-5,rel_tol=1e-4);assert base_hash(actor.model)==base and sha(boot['checkpoint'])==boot['checkpoint_sha256']
    result=dict(status='E32_TRAINING_PREFLIGHT_ONLY',historical_seed=30101,source_checkpoint_sha256=boot['checkpoint_sha256'],historical_updates=len(trace),historical_losses_gradients_actor_optimizer_rng_work_exact=True,historical_replay_seconds=replay_seconds,mean_gradient_examples=len(records),mean_loss_and_clipped_gradients_within_tolerance=True,gradient_tolerance=dict(atol=1e-6,rtol=1e-4),max_clipped_gradient_absolute_difference=max_grad,max_post_adam_parameter_absolute_difference=max_parameter,accumulation=accum,joint_loss=joint,joint_backward_cuda_peak_bytes=torch.cuda.max_memory_allocated(),base_and_source_unchanged=True,gradient_example_presentations=cfg['bootstrap_updates']+2*len(records),source_hashes={f:sha(f) for f in ('language_training_controls.py','preflight_language_training_controls.py')},scope='Historical compatibility and mean-gradient arithmetic only. No E32 scored fitting, held-out task result or self-improvement claim.');a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
