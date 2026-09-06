"""Small adapter for pinned CLRS reference bodies, not a CLRS benchmark replica.

Original graph algorithm bodies execute unchanged from their Apache-2.0 source.
Only rank assertions and probe recording are supplied locally, avoiding a full
JAX/TensorFlow installation. Hint values are discarded and never learner inputs.
"""
from pathlib import Path
import ast,hashlib,subprocess,types
import numpy as np
PIN='d33c3cfc765a18950194205a1ddb92a0981a355e'
ALGORITHMS=('dijkstra','mst_prim')
class Recorder:
    @staticmethod
    def initialize(_):return dict(inputs={},outputs={},hint_count=0)
    @staticmethod
    def push(record,stage,next_probe):
        if stage=='hint':record['hint_count']+=1
        else:record['inputs' if stage=='input' else 'outputs'].update(next_probe)
    @staticmethod
    def finalize(record):pass
    @staticmethod
    def mask_one(i,n):
        assert 0<=i<n;x=np.zeros(n);x[i]=1;return x
    @staticmethod
    def graph(a):return ((a+np.eye(len(a)))!=0)*1.
def rank(a,n):assert np.ndim(a)==n
class Reference:
    def __init__(self,repo):
        repo=Path(repo);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert head==PIN
        path=repo/'clrs/_src/algorithms/graphs.py';raw=path.read_bytes();tree=ast.parse(raw)
        defs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ALGORITHMS];assert len(defs)==2
        env=dict(np=np,_Array=np.ndarray,_Out=tuple,chex=types.SimpleNamespace(assert_rank=rank),probing=Recorder,
            specs=types.SimpleNamespace(SPECS={name:name for name in ALGORITHMS},Stage=types.SimpleNamespace(INPUT='input',HINT='hint',OUTPUT='output')))
        exec(compile(ast.Module(body=defs,type_ignores=[]),str(path),'exec'),env)
        self.functions={name:env[name] for name in ALGORITHMS};self.identity=dict(repo_commit=head,graphs_sha256=hashlib.sha256(raw).hexdigest(),functions=list(self.functions))
    def __call__(self,algorithm,a,source):
        assert algorithm in self.functions and a.ndim==2 and a.shape[0]==a.shape[1] and np.all(a>=0)
        parents,record=self.functions[algorithm](a,source)
        assert np.array_equal(parents,record['outputs']['pi'])
        # Public inputs and hint count are audit metadata, not passed to the model.
        return parents.astype(np.int64),record

def validate_tree(a,source,parents,algorithm):
    """Independent NetworkX functional check; no intermediate reference hints."""
    import networkx as nx
    graph=nx.from_numpy_array(a,create_using=nx.DiGraph if not np.array_equal(a,a.T) else nx.Graph)
    reachable=set(nx.node_connected_component(graph,source)) if not graph.is_directed() else set(nx.descendants(graph,source))|{source}
    assert parents[source]==source
    path_costs={}
    for v in range(len(a)):
        if v not in reachable:assert parents[v]==v;continue
        seen=set();node=v;cost=0.
        while node!=source:
            assert node not in seen,'Predicted cycle';seen.add(node);parent=int(parents[node]);assert 0<=parent<len(a) and a[parent,node]>0
            cost+=a[parent,node];node=parent
        path_costs[v]=cost
    if algorithm=='dijkstra':
        distance=nx.single_source_dijkstra_path_length(graph,source)
        assert all(abs(path_costs[v]-distance[v])<1e-6+1e-6*abs(distance[v]) for v in reachable)
    else:
        assert not graph.is_directed();tree=nx.minimum_spanning_tree(graph.subgraph(reachable));expected=sum(d['weight'] for _,_,d in tree.edges(data=True))
        actual=sum(a[int(parents[v]),v] for v in reachable if v!=source)
        assert abs(actual-expected)<1e-6+1e-6*abs(expected)
    return True
