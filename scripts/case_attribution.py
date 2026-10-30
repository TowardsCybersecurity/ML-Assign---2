"""Inspect the frozen booster's changed weak decisions for one saved test case."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import ROOT,prepare,occlude
from src.models import SAMME
A=ROOT/'artifacts'; x,y,s=prepare(); f=json.loads((A/'frozen.json').read_text()); cases=json.loads((A/'cases.json').read_text()); q=next(q for q in cases if q['kind']=='different_mask_response')
m=SAMME(**f['boosting'],random_state=f['model_seed']).fit(x[s['train']],y[s['train']]); clean=x[q['dataset_index']:q['dataset_index']+1]; masked=occlude(clean)
rows=[]; total=sum(m.weights_)
for t,(tree,alpha) in enumerate(zip(m.estimators_,m.weights_)):
    a=int(tree.predict(clean)[0]);b=int(tree.predict(masked)[0])
    if a==b:continue
    before=np.ones(240)/240 if t==0 else m.sample_weights_[t-1]
    classmass=np.bincount(y[s['train']],weights=before,minlength=40)
    peak=int(np.argmax(before));features=tree.tree_.feature
    path=tree.decision_path(clean).indices
    intersect=[int(i) for i in path if features[i]>=0 and 24<=features[i]//64<40 and 24<=features[i]%64<40]
    rows.append(dict(round=t+1,clean_prediction=a,masked_prediction=b,alpha=float(alpha),normalized_alpha=float(alpha/total),weighted_error=m.trace_[t]['error'],largest_training_weight=float(before[peak]),heaviest_training_image=s['train'][peak],largest_weight_identity=int(y[s['train'][peak]]),top_weight_class=int(classmass.argmax()),top_class_mass=float(classmass.max()),clean_path_nodes_in_mask=intersect))
assert np.allclose(m.predict_proba(clean)[0],q['boost_votes_clean'])
assert np.allclose(m.predict_proba(masked)[0],q['boost_votes_masked'])
result=dict(dataset_index=q['dataset_index'],changed_rounds=rows,accepted_rounds=len(m.estimators_),total_alpha=float(total),final_clean=m.predict_proba(clean)[0].tolist(),final_masked=m.predict_proba(masked)[0].tolist())
(A/'case_attribution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
