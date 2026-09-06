"""GPU/low-rank update feasibility only; no continual-learning result."""
from pathlib import Path
import argparse,hashlib,json,time,resource
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM

class LowRankLinear(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__();self.base=base;self.a=nn.Parameter(torch.randn(rank,base.in_features,device=base.weight.device,dtype=torch.float32)*.01);self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32));self.scale=1.
    def forward(self,x):return self.base(x)+(torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)*self.scale).to(x.dtype)

def base_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def adapter_state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def sync():torch.cuda.synchronize()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1532);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available();a.out.mkdir(parents=True,exist_ok=True)
    source=json.loads((a.model/'source_identity.json').read_text());tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='eager').eval().requires_grad_(False).to('cuda');sync();load_seconds=time.perf_counter()-start
    text=tokenizer.apply_chat_template([dict(role='user',content='Return only the codeword heliotrope.')],tokenize=False,add_generation_prompt=True);prompt=tokenizer(text,return_tensors='pt').to('cuda');target=tokenizer('heliotrope'+tokenizer.eos_token,add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda');ids=torch.cat([prompt['input_ids'],target],1);labels=ids.clone();labels[:,:prompt['input_ids'].shape[1]]=-100
    with torch.no_grad():original=model(input_ids=ids,use_cache=False).logits.detach().clone()
    for layer in model.model.layers:
        layer.self_attn.q_proj=LowRankLinear(layer.self_attn.q_proj);layer.self_attn.v_proj=LowRankLinear(layer.self_attn.v_proj)
    with torch.no_grad():initial=model(input_ids=ids,use_cache=False).logits.detach().clone()
    assert torch.equal(original,initial);before=base_hash(model);params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001);losses=[];norms=[];torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss=model(input_ids=ids,labels=labels,use_cache=False).loss;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params);opt.step();losses.append(float(loss.detach().cpu()));norms.append(float(norm))
    sync();update_seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();assert before==base_hash(model)
    with torch.no_grad():trained=model(input_ids=ids,use_cache=False).logits.detach().clone();final_loss=float(model(input_ids=ids,labels=labels,use_cache=False).loss)
    assert not torch.equal(initial,trained);state=adapter_state(model);cp=a.out/'adapter.pt';torch.save(state,cp)
    with torch.no_grad():
        for p in params:p.zero_()
        saved=torch.load(cp,weights_only=True)
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(saved[name])
        restored=model(input_ids=ids,use_cache=False).logits;assert torch.equal(restored,trained)
    result=dict(status='GPU_ADAPTER_PREFLIGHT_ONLY_NO_CONTINUAL_LEARNING_RESULT',model_source=source,device=torch.cuda.get_device_name(),torch=torch.__version__,base_dtype='float16',adapter_dtype='float32',rank=4,adapted_projections='q_proj and v_proj in each of 24 layers',trainable_parameters=sum(p.numel() for p in params),adapter_parameter_bytes=sum(p.numel()*p.element_size() for p in params),base_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),base_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if not p.requires_grad),load_seconds=load_seconds,update_seconds=update_seconds,updates=2,sequence_tokens=ids.shape[1],target_tokens=target.shape[1],losses=losses,final_same_example_loss=final_loss,gradient_norms=norms,zero_adapter_exact=True,base_weights_unchanged=True,base_state_sha256=before,restored_logits_exact=True,cuda_peak_training_allocated_bytes=peak,max_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,adapter_checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Two repeated-example updates establish execution and state isolation only. No old-task retention, novel-task transfer, memory compression, procedure improvement or efficiency claim.')
    (a.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
