"""Execute upstream collector method bodies with a minimal deterministic vec env."""
from pathlib import Path
import ast,hashlib,json,subprocess,types,copy
import numpy as np
ROOT=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research')
REPO=ROOT/'repos/imitation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Env:
    def __init__(self):self.actual=None
    def step_async(self,acts):self.actual=np.array(acts,copy=True)
    def step_wait(self):return self.actual[:,None].copy(),self.actual.astype(float),np.array([False]),[{}]
class Accumulator:
    def __init__(self):self.saved=[]
    def add_steps_and_auto_finish(self,**kw):self.saved.append(copy.deepcopy(kw));return []
def main():
    path=REPO/'src/imitation/algorithms/dagger.py';head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert head=='e5ef18806c449ca47153b494a02471c5e2ae3a14';assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True);tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='InteractiveTrajectoryCollector');methods=[copy.deepcopy(x) for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('step_async','step_wait')];assert len(methods)==2;module=ast.Module(body=methods,type_ignores=[]);ast.fix_missing_locations(module);namespace={'np':np,'VecEnvStepReturn':tuple,'_save_dagger_demo':lambda *args:None};exec(compile(module,str(path),'exec'),namespace);cases=[]
    for beta in (0.,1.):
        env=Env();accum=Accumulator();collector=types.SimpleNamespace(_is_reset=True,_last_obs=np.array([[0]]),num_envs=1,rng=np.random.default_rng(1981),beta=beta,get_robot_acts=lambda obs:np.ones(len(obs),dtype=int),venv=env,traj_accum=accum,_last_user_actions=None,save_dir=None);expert=np.array([2]);namespace['step_async'](collector,expert);namespace['step_wait'](collector);saved=accum.saved[0];actual=int(env.actual[0]);label=int(saved['acts'][0]);next_obs=int(saved['obs'][0,0]);assert label==2 and next_obs==actual;assert actual==(1 if beta==0 else 2);cases.append(dict(beta=beta,initial_observation=0,expert_label=label,executed_action=actual,saved_next_observation=next_obs,using_saved_label_as_executed_action_would_misidentify_transition=(next_obs!=label)))
    result=dict(status='IMITATION_SOURCE_COLLECTOR_AUDIT_ONLY',repo_commit=head,source_hashes={str(p.relative_to(REPO)):sha(p) for p in (path,REPO/'src/imitation/algorithms/bc.py',REPO/'setup.py')},unmodified_method_bodies_executed=[x.name for x in methods],cases=cases,scope='Minimal deterministic environment executes the upstream collector method bodies. It proves the stored action is an expert label even when the learner action caused the next observation. This is valid behavior-cloning data, not a bug claim. Reusing that label as the executed action in a world-model transition would be incorrect. No upstream policy training, benchmark reproduction or autonomous-learning result.',script_sha256=sha(__file__));(ROOT/'alw-runs/imitation-source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
