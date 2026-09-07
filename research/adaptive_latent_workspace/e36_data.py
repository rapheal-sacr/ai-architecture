"""E36 evaluator-owned environments and actual uniform-action training data."""
import random

from action_navigation import NavigationWorld, domain_seed, digest, graph


def make_world(spec):
    return NavigationWorld(spec['maps'],spec['durations'],spec['slots'],
                           goal_seed=spec['goal_seed'],noise_rate=spec.get('noise_rate',0.),
                           noise_seed=spec['noise_seed'])


def prepare(protocol):
    used=set()
    train_maps=set()
    eval_maps=set()
    states,actions=protocol['states'],protocol['actions']
    def fresh(rng):
        while True:
            mapping=graph(rng.getrandbits(64),states,actions)
            hashed=digest(mapping)
            if hashed not in used:
                used.add(hashed)
                return mapping
    training={}
    for seed in protocol['seeds']:
        training[str(seed)]=[]
        for index in range(protocol['outer_steps']):
            rng=random.Random(domain_seed(seed,'e36_train',index))
            maps=[fresh(rng),fresh(rng)]
            train_maps.update(digest(m) for m in maps)
            spec=dict(maps=maps,durations=protocol['train_durations'],slots=protocol['train_slots'],
                      goal_seed=rng.getrandbits(64),noise_seed=rng.getrandbits(64),noise_rate=0.)
            world=make_world(spec)
            actions_used=[rng.randrange(actions) for _ in range(sum(spec['durations']))]
            records=[]
            for action in actions_used:
                observation,reward,receipt=world.step(action)
                records.append(receipt)
            training[str(seed)].append(dict(world=spec,records=records))
    evaluation=[]
    for regime in protocol['evaluation_regimes']:
        for index in range(protocol['evaluation_worlds_per_regime']):
            rng=random.Random(domain_seed(38401,'e36_'+regime,index))
            maps=[fresh(rng)]
            duration,slots=[],[]
            total=protocol['evaluation_steps']
            if regime in ['stationary','noisy']:
                duration,slots=[total],[0]
            elif regime=='recurring':
                maps.append(fresh(rng))
                while sum(duration)<total:
                    duration.append(min(rng.randint(*protocol['evaluation_duration_range']),total-sum(duration)))
                    slots.append(len(slots)%2)
            elif regime=='drifting':
                while sum(duration)<total:
                    if duration:
                        while True:
                            old=maps[-1]
                            perm=list(range(states))
                            x,y=rng.sample(range(states),2)
                            perm[x],perm[y]=perm[y],perm[x]
                            action=rng.randrange(actions)
                            new=[row[:] for row in old]
                            for s in range(states):
                                new[perm[s]][action]=perm[old[s][action]]
                            if digest(new) not in used:
                                used.add(digest(new))
                                maps.append(new)
                                break
                    duration.append(min(rng.randint(*protocol['evaluation_duration_range']),total-sum(duration)))
                    slots.append(len(maps)-1)
            else:
                raise ValueError(regime)
            eval_maps.update(digest(m) for m in maps)
            evaluation.append(dict(name=f'{regime}-{index}',regime=regime,maps=maps,durations=duration,
                                   slots=slots,goal_seed=rng.getrandbits(64),noise_seed=rng.getrandbits(64),
                                   policy_seed=rng.getrandbits(64),noise_rate=protocol['evaluation_noise_rate'] if regime=='noisy' else 0.))
    assert not train_maps.intersection(eval_maps)
    return dict(training=training,evaluation=evaluation,
                audit=dict(distinct_train_maps=len(train_maps),distinct_eval_maps=len(eval_maps),
                           no_train_eval_overlap=True,training_action_records=sum(len(e['records']) for es in training.values() for e in es)))


def private_schedule(spec):
    return [slot for slot,n in zip(spec['slots'],spec['durations']) for _ in range(n)]
