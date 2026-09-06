"""Pinned language actor, conventional LoRA, and fixed observed-feedback updates."""
from pathlib import Path
import copy,json,time
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from preflight_language_adapter import LowRankLinear,base_hash,adapter_state
from delivery_world import RULES,render_prompt,canonical

def tensor_bytes(x):
    if torch.is_tensor(x):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(tensor_bytes(v) for v in x)
    return 0

class LanguageActor:
    def __init__(self,model_path,cfg,seed):
        self.cfg=cfg;torch.manual_seed(seed);start=time.perf_counter();self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='eager').eval().requires_grad_(False).to('cuda')
        torch.manual_seed(seed+1000)
        for layer in self.model.model.layers:
            layer.self_attn.q_proj=LowRankLinear(layer.self_attn.q_proj,cfg['rank']);layer.self_attn.v_proj=LowRankLinear(layer.self_attn.v_proj,cfg['rank'])
        self.params=[p for p in self.model.parameters() if p.requires_grad];self.letters=[]
        for letter in 'ABCDEFGH':
            token=self.tokenizer.encode(letter,add_special_tokens=False);assert len(token)==1;self.letters.append(token[0])
        self.action_rng=np.random.default_rng(seed+500000);self.learning_rng=np.random.default_rng(seed+600000);self.opt=None;self.replay=[];self.seen=0
        self.work=dict(inference_calls=0,inference_tokens=0,inference_attention_elements=0,update_calls=0,update_tokens=0,update_attention_elements=0,audit_calls=0,audit_tokens=0)
        torch.cuda.synchronize();self.load_seconds=time.perf_counter()-start
    def encode(self,prompt):
        text=self.tokenizer.apply_chat_template([dict(role='system',content=RULES),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        ids=self.tokenizer(text,return_tensors='pt');assert ids['input_ids'].shape[1]<=self.cfg['max_prompt_tokens'];return {k:v.to('cuda') for k,v in ids.items()}
    def logits(self,prompt,actions,kind):
        inp=self.encode(prompt);n=inp['input_ids'].shape[1];out=self.model(**inp,use_cache=False,num_logits_to_keep=1).logits[0,-1,self.letters[:len(actions)]].float();assert torch.isfinite(out).all()
        if kind=='audit':self.work['audit_calls']+=1;self.work['audit_tokens']+=n
        else:self.work[kind+'_calls']+=1;self.work[kind+'_tokens']+=n;self.work[kind+'_attention_elements']+=24*14*n*n
        return out,n
    @torch.no_grad()
    def act(self,obs,memory):
        prompt,actions=render_prompt(obs,memory);logits,tokens=self.logits(prompt,actions,'inference');p=torch.softmax(logits/self.cfg['temperature'],-1);p=(1-self.cfg['uniform_exploration'])*p+self.cfg['uniform_exploration']/len(actions);probs=p.cpu().numpy().astype(np.float64);probs/=probs.sum();index=int(self.action_rng.choice(len(actions),p=probs));return actions[index],dict(prompt=prompt,actions=actions,index=index,behavior_logp=float(np.log(probs[index])),tokens=tokens)
    def start_optimizer(self,learning_rate,state=None):
        self.opt=torch.optim.Adam(self.params,lr=learning_rate)
        if state is not None:
            self.opt.load_state_dict(copy.deepcopy(state))
            for group in self.opt.param_groups:group['lr']=learning_rate
    def supervised_update(self,record):
        assert self.opt is not None;self.opt.zero_grad(set_to_none=True);logits,_=self.logits(record['prompt'],record['actions'],'update');loss=-torch.log_softmax(logits,-1)[record['index']];return self.finish_update(loss)
    def finish_update(self,loss):
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.params,self.cfg['gradient_clip']);assert torch.isfinite(norm);self.opt.step();return dict(loss=float(loss.detach().cpu()),grad_norm=float(norm))
    def observe_return(self,trajectory,success):
        advantage=float(success)-self.cfg['reward_step_penalty']*len(trajectory)-self.cfg['return_baseline'];current=[]
        for item in trajectory:
            record=copy.deepcopy(item);record['advantage']=advantage;current.append(record);self.seen+=1
            index=len(self.replay) if len(self.replay)<self.cfg['experience_capacity'] else int(self.learning_rng.integers(self.seen))
            if index<self.cfg['experience_capacity']:
                if index==len(self.replay):self.replay.append(record)
                else:self.replay[index]=record
        records=[]
        for i in range(self.cfg['online_updates_per_mission']):
            pool=current if i<self.cfg['online_updates_per_mission']//2 else self.replay;record=pool[int(self.learning_rng.integers(len(pool)))];self.opt.zero_grad(set_to_none=True);logits,_=self.logits(record['prompt'],record['actions'],'update');p=torch.softmax(logits/self.cfg['temperature'],-1);p=(1-self.cfg['uniform_exploration'])*p+self.cfg['uniform_exploration']/len(record['actions']);ratio=torch.exp(torch.log(p[record['index']])-record['behavior_logp']);adv=record['advantage'];loss=-torch.minimum(ratio*adv,torch.clamp(ratio,1-self.cfg['ppo_clip'],1+self.cfg['ppo_clip'])*adv);records.append(self.finish_update(loss))
        return records
    def load_adapter(self,state):
        with torch.no_grad():
            for name,p in self.model.named_parameters():
                if p.requires_grad:p.copy_(state[name])
    def state(self):return dict(adapter=adapter_state(self.model),optimizer=copy.deepcopy(self.opt.state_dict()) if self.opt else None,replay=copy.deepcopy(self.replay),seen=self.seen,action_rng=copy.deepcopy(self.action_rng.bit_generator.state),learning_rng=copy.deepcopy(self.learning_rng.bit_generator.state),work=copy.deepcopy(self.work),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    def costs(self):
        return dict(base_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters() if not p.requires_grad),adapter_parameter_bytes=sum(p.numel()*p.element_size() for p in self.params),optimizer_tensor_bytes=tensor_bytes(self.opt.state_dict()) if self.opt else 0,gradient_tensor_bytes=sum(p.grad.numel()*p.grad.element_size() for p in self.params if p.grad is not None),rng_tensor_bytes=tensor_bytes(torch.get_rng_state())+tensor_bytes(torch.cuda.get_rng_state_all()),replay_utf8_bytes=len(canonical(self.replay).encode()),replay_records=len(self.replay),**self.work)
    def load_state(self,state):
        self.load_adapter(state['adapter'])
        if state['optimizer'] is not None:
            if self.opt is None:self.start_optimizer(self.cfg['online_learning_rate'])
            self.opt.load_state_dict(copy.deepcopy(state['optimizer']))
        else:self.opt=None
        self.replay=copy.deepcopy(state['replay']);self.seen=state['seen'];self.action_rng.bit_generator.state=copy.deepcopy(state['action_rng']);self.learning_rng.bit_generator.state=copy.deepcopy(state['learning_rng']);self.work=copy.deepcopy(state['work']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
