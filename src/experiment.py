"""Select on train/validation first. Evaluate only after frozen.json exists."""
import argparse
import json
import platform
import sys
from time import perf_counter
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from src.data import ROOT,prepare,occlude,fingerprint
from src.models import Bagging,SAMME,samme_round

A=ROOT/'artifacts'
CFG=json.loads((ROOT/'configs/protocol.json').read_text())
SEED=CFG['model_seed']

def save(name,value):
    A.mkdir(parents=True, exist_ok=True)
    (A/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def metrics(p,y,classes):
    pred=classes[np.argmax(p,axis=1)]
    return dict(correct=int(np.sum(pred==y)),n=len(y),accuracy=float(np.mean(pred==y)),confidence=float(np.max(p,axis=1).mean()))

def occupancy(tree):
    counts=tree.tree_.n_node_samples[tree.tree_.children_left==-1]
    return dict(leaves=len(counts),mean_samples=float(counts.mean()),singleton_fraction=float(np.mean(counts==1)))

def disagreement(predictions):
    # Number of unordered disagreeing pairs, averaged over inputs and pairs.
    b,n=predictions.shape
    if b<2: return None
    same=sum(np.sum(np.bincount(predictions[:,i],minlength=40)**2)-b for i in range(n))
    return float(1-same/(n*b*(b-1)))

def bag_diagnostics(m,x,y,z,n):
    votes=np.zeros((len(x),len(m.classes_))); count=np.zeros(len(x),int)
    for tree,ix in zip(m.estimators_[:n],m.bootstrap_indices_[:n]):
        oob=np.ones(len(x),bool); oob[np.unique(ix)]=False
        rows=np.flatnonzero(oob)
        votes[rows,np.searchsorted(m.classes_,tree.predict(x[rows]))]+=1
        count[rows]+=1
    eligible=count>0
    p=m.predict_proba(z,n); top=np.sort(p,axis=1)[:,-2:]
    return dict(oob_accuracy=float(np.mean(m.classes_[votes[eligible].argmax(axis=1)]==y[eligible])),oob_coverage=int(eligible.sum()),disagreement=disagreement(np.array([t.predict(z) for t in m.estimators_[:n]])),margin=float(np.mean(top[:,1]-top[:,0])),distinct=float(np.mean([len(np.unique(ix)) for ix in m.bootstrap_indices_[:n]])),distinct_expectation=float(len(x)*(1-(1-1/len(x))**len(x))))

def fit_tree(x,y,c):
    return DecisionTreeClassifier(**c,random_state=SEED).fit(x,y)

def select():
    save('toy.json',toy())
    x,y,s=prepare(); tr=s['train']; va=s['validation']
    # This function never indexes test labels or pixels.
    xt,yt,xv,yv=x[tr],y[tr],x[va],y[va]
    candidates={'tree':[],'bagging':[],'boosting':[]}
    start=perf_counter()
    for d in CFG['tree_depths']:
        for leaf in CFG['leaves']:
            c=dict(max_depth=d,min_samples_leaf=leaf)
            t0=perf_counter(); m=fit_tree(xt,yt,c)
            candidates['tree'].append(dict(config=c,validation_correct=int(np.sum(m.predict(xv)==yv)),train_accuracy=float(np.mean(m.predict(xt)==yt)),fit_seconds=perf_counter()-t0,**occupancy(m)))
    best=max(candidates['tree'],key=lambda r:r['validation_correct'])
    tc=best['config']
    bag=Bagging(max(CFG['bag_sizes']),**tc,random_state=SEED).fit(xt,yt)
    for b in CFG['bag_sizes']:
        candidates['bagging'].append(dict(n=b,validation_correct=int(np.sum(bag.predict(xv,b)==yv)),fit_seconds=float(sum(bag.fit_seconds_[:b]))))
    for d in CFG['boost_depths']:
        boost=SAMME(max(CFG['boost_rounds']),max_depth=d,random_state=SEED).fit(xt,yt)
        for n in CFG['boost_rounds']:
            actual=min(n,len(boost.estimators_))
            candidates['boosting'].append(dict(depth=d,n=n,actual=actual,validation_correct=int(np.sum(boost.predict(xv,actual)==yv)),fit_seconds=float(sum(boost.fit_seconds_[:actual])),stop_reason=boost.stop_reason_))
    bb=max(candidates['bagging'],key=lambda r:r['validation_correct'])
    bs=max(candidates['boosting'],key=lambda r:r['validation_correct'])
    frozen=dict(tree=tc,bagging=dict(n_estimators=bb['n'],**tc),boosting=dict(n_estimators=bs['n'],max_depth=bs['depth']),model_seed=SEED,selection=CFG['selection'],split_sha256=fingerprint(np.array(tr+va+s['test'],dtype=np.int64)))
    save('candidates.json',candidates); save('frozen.json',frozen)
    save('selection_log.json',dict(seconds=perf_counter()-start,command='python -m src.experiment select',test_used=False))
    print('Frozen:',frozen,flush=True)

def fit_selected(xt,yt,f):
    return dict(tree=fit_tree(xt,yt,f['tree']),bagging=Bagging(**f['bagging'],random_state=SEED).fit(xt,yt),boosting=SAMME(**f['boosting'],random_state=SEED).fit(xt,yt))

def tree_structure(tree):
    features=tree.tree_.feature
    features=features[features>=0]
    inside=((features//64>=24)&(features//64<40)&(features%64>=24)&(features%64<40))
    return dict(achieved_depth=int(tree.get_depth()),internal_nodes=len(features),masked_region_nodes=int(inside.sum()),split_features=features.tolist())

def evaluate():
    f=json.loads((A/'frozen.json').read_text()); x,y,s=prepare()
    xt,yt=x[s['train']],y[s['train']]; xv,yv=x[s['validation']],y[s['validation']]
    xe,ye=x[s['test']],y[s['test']]; xm=occlude(xe)
    t0=perf_counter(); tree=fit_tree(xt,yt,f['tree']); tree_time=perf_counter()-t0
    bag=Bagging(200,**f['tree'],random_state=SEED).fit(xt,yt)
    boost=SAMME(100,max_depth=f['boosting']['max_depth'],random_state=SEED).fit(xt,yt)
    nb=f['bagging']['n_estimators']; ns=min(f['boosting']['n_estimators'],len(boost.estimators_))
    results=dict(bagging=[],boosting=[],selected={},tree_candidates_test=[])
    predsave={'labels':ye,'indices':np.array(s['test'])}
    for n in CFG['bag_sizes']:
        row=dict(n=n,fit_seconds=float(sum(bag.fit_seconds_[:n])),**bag_diagnostics(bag,xt,yt,xv,n))
        for name,z,l in [('train',xt,yt),('validation',xv,yv),('clean',xe,ye)]: row[name]=metrics(bag.predict_proba(z,n),l,bag.classes_)
        results['bagging'].append(row)
    for n in CFG['boost_rounds']:
        actual=min(n,len(boost.estimators_)); row=dict(n=n,actual=actual,fit_seconds=float(sum(boost.fit_seconds_[:actual])))
        for name,z,l in [('train',xt,yt),('validation',xv,yv),('clean',xe,ye)]: row[name]=metrics(boost.predict_proba(z,actual),l,boost.classes_)
        results['boosting'].append(row)
    for name,m,n in [('tree',tree,None),('bagging',bag,nb),('boosting',boost,ns)]:
        row={}
        for split,z,l in [('train',xt,yt),('validation',xv,yv),('clean',xe,ye),('masked',xm,ye)]:
            t0=perf_counter(); p=m.predict_proba(z) if n is None else m.predict_proba(z,n)
            row[split]=metrics(p,l,m.classes_); row[split]['inference_seconds']=perf_counter()-t0
            if split in ('clean','masked'): predsave[name+'_'+split]=p
        row['drop_pp']=100*(row['clean']['accuracy']-row['masked']['accuracy'])
        row['fit_seconds']=tree_time if name=='tree' else float(sum(m.fit_seconds_[:n]))
        results['selected'][name]=row
    results['tree_occupancy']=occupancy(tree)
    save('tree_structure.json',tree_structure(tree))
    # All requested depth/leaf train and validation effects; test never used to rank.
    for c in json.loads((A/'candidates.json').read_text())['tree']:
        m=fit_tree(xt,yt,c['config']); results['tree_candidates_test'].append(dict(config=c['config'],test_accuracy=float(np.mean(m.predict(xe)==ye))))
    np.savez_compressed(A/'predictions.npz',**predsave)
    save('results.json',results)
    save('trace.json',boost.trace_[:ns])
    weights=np.array(boost.sample_weights_[:ns]); hist=np.array(boost.training_predictions_[:ns])
    heavy=int(np.argmax(weights.max(axis=0)))
    save('heavy.json',dict(train_position=heavy,dataset_index=s['train'][heavy],label=int(yt[heavy]),weights=weights[:,heavy].tolist(),predictions=hist[:,heavy].tolist(),incorrect=(hist[:,heavy]!=yt[heavy]).tolist()))
    np.savez_compressed(A/'boost_training.npz',weights=weights,predictions=hist,labels=yt,indices=np.array(s['train']))
    # Case indices fixed by ascending test order; never select a model from these.
    pt=predsave['tree_clean'].argmax(1); pb=predsave['bagging_clean'].argmax(1)
    corrected=np.flatnonzero((pt!=ye)&(pb==ye))[:2]
    wrong=np.flatnonzero((pt!=ye)&(pb!=ye))[:2]
    changed_b=pb!=predsave['bagging_masked'].argmax(1)
    changed_s=predsave['boosting_clean'].argmax(1)!=predsave['boosting_masked'].argmax(1)
    diff=np.flatnonzero(changed_b!=changed_s)
    if not len(diff): diff=np.flatnonzero(predsave['bagging_masked'].argmax(1)!=predsave['boosting_masked'].argmax(1))
    cases=[]
    for kind,ixs in [('bag_corrects',corrected),('both_wrong',wrong),('different_mask_response',diff[:1])]:
        for i in ixs:
            cases.append(dict(kind=kind,test_position=int(i),dataset_index=s['test'][i],label=int(ye[i]),tree_prediction=int(pt[i]),bag_votes_clean=np.rint(nb*predsave['bagging_clean'][i]).astype(int).tolist(),bag_votes_masked=np.rint(nb*predsave['bagging_masked'][i]).astype(int).tolist(),boost_votes_clean=predsave['boosting_clean'][i].tolist(),boost_votes_masked=predsave['boosting_masked'][i].tolist(),bag_disagreement_clean=disagreement(np.array([[t.predict(xe[i:i+1])[0]] for t in bag.estimators_[:nb]])),bag_disagreement_masked=disagreement(np.array([[t.predict(xm[i:i+1])[0]] for t in bag.estimators_[:nb]]))))
    save('cases.json',cases)
    save('toy.json',toy())
    save('evaluation_log.json',dict(command='python -m src.experiment evaluate',frozen_sha256=fingerprint(np.frombuffer((A/'frozen.json').read_bytes(),dtype=np.uint8)),tree_seconds=tree_time,boost_stop=boost.stop_reason_))
    print(json.dumps(results['selected'],indent=2),flush=True)

def toy():
    x=np.zeros((6,1)); y=np.array([0,0,0,0,1,2])
    # A constant input makes the actual depth-1 tree predict class 0, error 1/3.
    tree=DecisionTreeClassifier(max_depth=1,random_state=SEED).fit(x,y,sample_weight=np.ones(6)/6)
    pred=tree.predict(x); w=np.ones(6)/6
    e,a,u=samme_round(w,pred!=y,3)
    return dict(x=x.tolist(),y=y.tolist(),initial=w.tolist(),prediction=pred.tolist(),error=e,alpha=a,updated=u.tolist())

def bias_variance(p,y,classes):
    mean=p.mean(axis=0); target=(y[:,None]==classes[None,:]).astype(float)
    return float(np.mean(np.sum((mean-target)**2,axis=1))),float(np.mean(np.sum((p-mean)**2,axis=2)))

def resample_worker(task):
    r,ix,xt,yt,xe,specs=task
    fitted=[]; output={}
    # Fit longest bag first. A shorter prefix is the exact same estimator
    # sequence, since number of estimators never changes the per-tree RNG.
    ordered=sorted(specs.items(),key=lambda kv: -kv[1][1].get('n_estimators',0))
    for name,(kind,c) in ordered:
        m=None; reused=False
        for previous_kind,previous_c,previous_model,previous_seconds in fitted:
            if kind!=previous_kind or c.get('min_samples_leaf',1)!=previous_c.get('min_samples_leaf',1): continue
            if kind!='tree' and c['n_estimators']>len(previous_model.estimators_): continue
            same_depth=c['max_depth']==previous_c['max_depth']
            # A finite cap exceeding every achieved tree depth cannot affect
            # any split: reuse only after verifying that structural condition.
            trees=[previous_model] if kind=='tree' else previous_model.estimators_[:c['n_estimators']]
            equivalent_cap=previous_c['max_depth'] is None and c['max_depth'] is not None and all(t.get_depth()<c['max_depth'] for t in trees)
            if same_depth or equivalent_cap:
                m=previous_model; sec=previous_seconds; reused=True; break
        if m is None:
            t0=perf_counter()
            if kind=='tree': m=fit_tree(xt[ix],yt[ix],c)
            else: m=({'bagging':Bagging,'boosting':SAMME}[kind])(**c,random_state=SEED).fit(xt[ix],yt[ix])
            sec=perf_counter()-t0
            fitted.append((kind,c,m,sec))
        n=None if kind=='tree' else min(c['n_estimators'],len(m.estimators_))
        p=m.predict_proba(xe) if n is None else m.predict_proba(xe,n)
        if kind=='tree': diag=dict(**occupancy(m),achieved_depth=int(m.get_depth()))
        elif kind=='bagging': diag=dict(disagreement=disagreement(np.array([t.predict(xe) for t in m.estimators_[:n]])),max_achieved_depth=max(int(t.get_depth()) for t in m.estimators_[:n]))
        else: diag=dict(stop_reason=m.stop_reason_)
        diag['reused_equivalent_fit']=reused
        output[name]=(p,diag,sec)
    return output

def stratified_bootstrap(labels,rng):
    """Six replacement draws from each identity's original training positions."""
    return np.concatenate([rng.choice(np.flatnonzero(labels==c),size=6,replace=True) for c in np.unique(labels)])

def resample():
    f=json.loads((A/'frozen.json').read_text()); x,y,s=prepare()
    xt,yt=x[s['train']],y[s['train']]; xe,ye=x[s['test']],y[s['test']]
    d=f['tree']['max_depth']; depths=[2,d,None] if d not in (2,None) else ([1,2,None] if d==2 else [2,None,64])
    specs={'tree':('tree',f['tree']),'bagging':('bagging',f['bagging']),'boosting':('boosting',f['boosting'])}
    for j,depth in enumerate(depths):
        c=dict(max_depth=depth,min_samples_leaf=f['tree']['min_samples_leaf'])
        specs[f'depth{j}_tree']=('tree',c)
        specs[f'depth{j}_bagging']=('bagging',dict(n_estimators=100,**c))
    rng=np.random.default_rng(CFG['resample_seed'])
    samples=[stratified_bootstrap(yt,rng) for _ in range(CFG['repetitions'])]
    values={k:[] for k in specs}; diagnostics={k:[] for k in specs}; times={k:[] for k in specs}
    start=perf_counter()
    # Independent repetitions run in six processes; their seeds, order and
    # arrays are fixed before dispatch. No cross-process model state is shared.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6) as pool:
        tasks=[(r,ix,xt,yt,xe,specs) for r,ix in enumerate(samples)]
        for r,output in enumerate(pool.map(resample_worker,tasks)):
            for name,(p,diag,sec) in output.items():
                values[name].append(p); diagnostics[name].append(diag); times[name].append(sec)
            np.savez_compressed(A/'resamples.npz',indices=np.array(samples[:r+1]),labels=ye,**{k:np.array(v) for k,v in values.items()})
            print(f'Resample {r+1}/30 elapsed {perf_counter()-start:.1f}s',flush=True)
    result={}
    for name,v in values.items():
        p=np.array(v); b,var=bias_variance(p,ye,np.arange(40)); acc=np.mean(p.argmax(2)==ye,axis=1)
        # Squared-loss decomposition is exact for these finite empirical vectors.
        loss=float(np.mean(np.sum((p-np.eye(40)[ye])**2,axis=2)))
        assert np.isclose(loss,b+var)
        result[name]=dict(config=specs[name][1],bias2=b,variance=var,mean_accuracy=float(acc.mean()),std_accuracy=float(acc.std(ddof=1)),squared_loss=loss,accuracy_per_repeat=acc.tolist(),diagnostics=diagnostics[name],fit_seconds=times[name])
    save('bias_variance.json',result)
    save('resample_log.json',dict(command='python -m src.experiment resample',seconds=perf_counter()-start,repetitions=30,depths=depths,model_rng='fixed across refits',resample_seed=CFG['resample_seed']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['select','evaluate','resample'])
    args=parser.parse_args(); globals()[args.stage]()
