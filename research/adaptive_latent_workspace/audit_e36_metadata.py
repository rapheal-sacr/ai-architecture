"""Independent arithmetic checks of reported E36 totals and storage.

Complements gradient/state replay: no neural execution, no new model scoring.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def fold_work(rows):
    result={}
    for row in rows:
        for k,v in row['work'].items():
            result[k]=result.get(k,0)+v
    return result


def run(folder,out):
    complete=json.loads((folder/'complete.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    assert complete['status']=='complete'
    assert complete['manifest']==manifest
    protocol=manifest['protocol']
    cfg=protocol['config']
    data=json.loads((folder/'data.json').read_text())
    assert hashlib.sha256((folder/'data.json').read_bytes()).hexdigest()==manifest['data_sha256']
    expected_training=len(protocol['seeds'])*len(protocol['training_arms'])
    assert len(complete['training'])==expected_training
    training_files={}
    updates=0
    for entry in complete['training']:
        path=Path(entry['training_file'])
        rows=json.loads(path.read_text())
        training_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(rows)==protocol['outer_steps']
        for t,row in enumerate(rows):
            length=len(data['training'][str(entry['seed'])][t]['records'])
            assert row['step']==t+1
            inner=int(entry['arm']!='static')*length
            attention=sum(cfg['layers']*cfg['heads']*2*(min(2*i,cfg['window'])+2) for i in range(length))
            expected=dict(forward_tokens=2*length,attention_score_elements=attention,forwards=length,
                          inner_updates=inner,gradient_tokens=2*inner)
            assert row['work']==expected,(entry['seed'],entry['arm'],t)
            assert math.isfinite(row['loss']) and math.isfinite(row['outer_grad_norm'])
            assert row['seconds']>=0
            updates+=1
        assert entry['work']==fold_work(rows)
        assert entry['final_loss']==rows[-1]['loss']
        assert entry['seconds']==sum(r['seconds'] for r in rows)
    w,h=cfg['width'],cfg['hidden']
    base_parameters=cfg['vocab']*w+w+cfg['layers']*(4*w+2*(w//cfg['heads'])+4*w*w+3*w*h)+cfg['suffix']*(2*w+3*w*h)
    fast_parameters=cfg['suffix']*3*w*h
    persistent_bytes=4*(fast_parameters+2*cfg['layers']*min(2*protocol['evaluation_steps'],cfg['window'])*w)
    n,a=protocol['states'],protocol['actions']
    steps=protocol['evaluation_steps']
    expected_attention=(n*a+1)*sum(cfg['layers']*cfg['heads']*2*(min(2*t,cfg['window'])+2) for t in range(steps))
    checked=neural=controls=0
    result_hashes={}
    for entry in complete['evaluation']:
        path=Path(entry['result_file'])
        raw=json.loads(path.read_text())
        result_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key,value in entry.items():
            if key not in ['seed','result_file']:
                assert value==raw[key],(str(path),key)
        rows=raw['rows']
        assert len(rows)==raw['actions']==steps
        for t,row in enumerate(rows):
            assert row['step']==t+1
            assert row['correct']==int(row['prediction']==row['outcome'])
            assert row['reward']==int(row['goal']==row['outcome'])
            assert row['nll'] is None or (math.isfinite(row['nll']) and row['nll']>=0)
        assert raw['goals']==sum(r['reward'] for r in rows)
        assert raw['correct']==sum(r['correct'] for r in rows)
        assert raw['work']==fold_work(rows)
        if raw['mean_nll'] is not None:
            assert raw['mean_nll']==sum(r['nll'] for r in rows)/steps
        if isinstance(raw['condition'],dict):
            inner=int(raw['condition']['update']!='frozen')*steps
            expected=dict(hypothetical_queries=n*a*steps,forward_tokens=2*(n*a+1)*steps,
                          attention_score_elements=expected_attention,inner_updates=inner,gradient_tokens=2*inner,
                          vectorized_batches=steps,bellman_iterations=protocol['planning']['horizon']*steps,
                          bellman_transition_terms=protocol['planning']['horizon']*n*n*a*steps,forwards=steps)
            assert raw['work']==expected
            assert raw['storage']==dict(base_parameter_bytes=4*base_parameters,
                                        persistent_tensor_bytes=persistent_bytes,base_parameters=base_parameters)
            assert raw['base_parameters_unchanged']
            neural+=1
        else:
            controls+=1
            if raw['condition']=='counts':
                assert raw['work']==dict(bellman_iterations=protocol['planning']['horizon']*steps,
                                        bellman_transition_terms=protocol['planning']['horizon']*n*n*a*steps)
                assert raw['storage']['count_tensor_bytes']==4*n*a*n
                assert raw['storage']['logical_uint8_history_bytes']<=n*a*protocol['count_control_window']
            elif raw['condition']=='bfs':
                assert raw['storage']['logical_int64_record_bytes']==raw['storage']['entries']*24
                assert raw['storage']['entries']<=n*a
                assert raw['storage']['writes']==steps
            else:
                assert raw['condition']=='random' and raw['storage']=={}
        checked+=len(rows)
    assert neural==len(protocol['seeds'])*len(protocol['evaluation_arms'])*len(data['evaluation'])
    assert controls==3*len(data['evaluation'])
    result=dict(status='complete',training_updates_checked=updates,neural_cases=neural,control_cases=controls,
                evaluation_rows_checked=checked,base_parameters=base_parameters,
                base_parameter_bytes=4*base_parameters,persistent_tensor_bytes=persistent_bytes,
                main_complete_sha256=hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest(),
                training_log_sha256=training_files,evaluation_result_sha256=result_hashes,
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['training_log_sha256','evaluation_result_sha256']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.input,a.out)
