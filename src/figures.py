"""Generate all plots and tables from saved artifacts; never fit models."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.data import ROOT,dataset,occlude
A=ROOT/'artifacts'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42})

def read(n): return json.loads((A/n).read_text())
def finish(fig,name):
    fig.tight_layout(); fig.savefig(F/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':datetime(2026,10,26,tzinfo=timezone.utc),'ModDate':datetime(2026,10,26,tzinfo=timezone.utc)}); plt.close(fig)
def table(name,header,rows):
    with (F/(name+'.csv')).open('w',newline='') as output:
        writer=csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
def pct(x): return f'{100*x:.2f}'
def depth(d): return 'None' if d is None else str(d)

def main():
    r=read('results.json'); c=read('candidates.json'); bv=read('bias_variance.json'); frozen=read('frozen.json'); pred=np.load(A/'predictions.npz'); traces=read('trace.json'); heavy=read('heavy.json'); cases=read('cases.json')
    tests={str(x['config']):x['test_accuracy'] for x in r['tree_candidates_test']}
    rows=[]
    for q in c['tree']:
        rows.append([depth(q['config']['max_depth']),q['config']['min_samples_leaf'],pct(q['train_accuracy']),pct(q['validation_correct']/80),pct(tests[str(q['config'])]),q['leaves'],f"{q['mean_samples']:.2f}",pct(q['singleton_fraction'])])
    table('baseline',['Depth','Leaf','Train','Val.','Test','Leaves','Mean samples per leaf','Singleton leaves (%)'],rows)
    rows=[]
    for q in r['bagging']:
        rows.append([q['n'],pct(q['train']['accuracy']),pct(q['validation']['accuracy']),pct(q['clean']['accuracy']),f"{q['distinct']:.2f}",pct(q['oob_accuracy']),q['oob_coverage']])
    table('bagging',['B','Train','Val.','Test','Distinct','OOB','Cov.'],rows)
    table('boost_validation',['Depth']+[str(n) for n in [10,25,50,100]],[[d]+[pct(next(q['validation_correct']/80 for q in c['boosting'] if q['depth']==d and q['n']==n)) for n in [10,25,50,100]] for d in [1,2,3,4]])
    rows=[]
    for n in [25,50,100]:
        for model in ['bagging','boosting']:
            q=next(q for q in r[model] if q['n']==n)
            rows.append([model.title(),n,pct(q['train']['accuracy']),pct(q['validation']['accuracy']),pct(q['clean']['accuracy']),f"{q['fit_seconds']:.2f}"])
    table('comparison',['Model','M','Train','Val.','Test','Fit (s)'],rows)
    rows=[]
    for model in ['tree','bagging','boosting']:
        q=r['selected'][model]; rows.append([model.title(),f"{q['clean']['correct']}/80",pct(q['clean']['accuracy']),pct(q['masked']['accuracy']),f"{q['drop_pp']:.2f}",f"{q['clean']['confidence']:.3f}",f"{q['masked']['confidence']:.3f}"])
    table('robustness',['Model','Correct','Clean','Mask','Drop (pp)','Conf. C','Conf. M'],rows)
    rows=[]
    for model in ['tree','bagging','boosting']:
        q=bv[model]; rows.append([model.title(),f"{q['bias2']:.4f}",f"{q['variance']:.4f}",pct(q['mean_accuracy']),f"{100*q['std_accuracy']:.2f}"])
    table('bias_variance',['Model','Bias squared','Variance','Mean acc.','SD (pp)'],rows)
    rows=[]
    for j in range(3):
        for model in ['tree','bagging']:
            q=bv[f'depth{j}_{model}']; diag=q['diagnostics']
            occ=f"{np.mean([d['mean_samples'] for d in diag]):.2f}" if model=='tree' else '--'
            dis=f"{np.mean([d['disagreement'] for d in diag]):.3f}" if model=='bagging' else '--'
            rows.append([depth(q['config']['max_depth']),model.title(),f"{q['bias2']:.3f}",f"{q['variance']:.3f}",pct(q['mean_accuracy']),occ,dis])
    table('depth',['Depth','Model','Bias squared','Var.','Acc.','Mean samples per leaf','Disagr.'],rows)
    rows=[]
    for model in ['tree','bagging','boosting']:
        q=r['selected'][model]; b=bv[model]; rows.append([model.title(),pct(q['clean']['accuracy']),pct(q['masked']['accuracy']),f"{b['bias2']:.4f}",f"{b['variance']:.4f}"])
    table('final',['Model','Clean','Masked','Bias squared','Variance'],rows)
    b=r['bagging']; ns=[q['n'] for q in b]
    fig,axs=plt.subplots(3,1,figsize=(3.3,4.5),sharex=True)
    for ax,key,label in zip(axs,['oob_accuracy','disagreement','margin'],['OOB accuracy','Pairwise disagreement','Mean vote margin']):
        vals=[np.nan if q[key] is None else q[key] for q in b]
        ax.plot(ns,vals,'o-',markersize=3); ax.set_ylabel(label); ax.grid(alpha=.25); ax.set_xscale('log'); ax.axvline(frozen['bagging']['n_estimators'],color='gray',linestyle='--')
    axs[-1].set_xticks(ns,labels=ns); axs[-1].set_xlabel('Number of trees B')
    finish(fig,'diagnostics')
    fig,axs=plt.subplots(2,2,figsize=(3.3,3.1))
    for ax,key,label in zip(axs.ravel(),['error','alpha','max_weight','top10_weight'],['Weighted error',r'Learner weight $\alpha$','Max sample weight','Top-10 weight mass']):
        ax.plot([t['round'] for t in traces],[t[key] for t in traces]); ax.set_title(label,fontsize=8); ax.set_xlabel('Round'); ax.grid(alpha=.25)
    finish(fig,'traces')
    x,y=dataset()
    fig,axs=plt.subplots(1,2,figsize=(3.3,1.6),gridspec_kw={'width_ratios':[1,2]})
    axs[0].imshow(x[heavy['dataset_index']].reshape(64,64),cmap='gray',vmin=0,vmax=1); axs[0].axis('off'); axs[0].set_title(f"Train image {heavy['dataset_index']}\nclass {heavy['label']}")
    rounds=np.arange(1,len(heavy['weights'])+1); axs[1].plot(rounds,heavy['weights']); bad=np.array(heavy['incorrect']); axs[1].scatter(rounds[bad],np.array(heavy['weights'])[bad],s=6,color='red',label='Weak learner wrong'); axs[1].set_xlabel('Round'); axs[1].set_ylabel('Updated weight'); axs[1].legend(fontsize=6)
    finish(fig,'heavy')
    fig,axs=plt.subplots(2,2,figsize=(3.3,3.4))
    for ax,q in zip(axs.ravel(),[q for q in cases if q['kind']!='different_mask_response']):
        ix=q['dataset_index']; votes=np.array(q['bag_votes_clean']); nonzero=np.flatnonzero(votes)
        counts=', '.join(f'{j}:{votes[j]}' for j in sorted(nonzero,key=lambda j:(-votes[j],j))[:3])
        ax.imshow(x[ix].reshape(64,64),cmap='gray',vmin=0,vmax=1); ax.axis('off')
        ax.set_title(f"Image {ix}; true {q['label']}; tree {q['tree_prediction']}\nBag {votes.argmax()}; "+q['kind'].replace('_',' '),fontsize=7)
        ax.text(.5,-.05,counts,transform=ax.transAxes,ha='center',va='top',fontsize=8,wrap=True)
    finish(fig,'cases')
    q=next((q for q in cases if q['kind']=='different_mask_response'),None)
    if q:
        z=x[q['dataset_index']:q['dataset_index']+1]; zm=occlude(z)
        fig,axs=plt.subplots(1,2,figsize=(3.3,1.8))
        for ax,zv,split in zip(axs,[z,zm],['clean','masked']):
            ax.imshow(zv.reshape(64,64),cmap='gray',vmin=0,vmax=1); ax.axis('off'); ax.set_title(f"{split.title()}: true {q['label']}\nBag {np.argmax(q['bag_votes_'+split])}; SAMME {np.argmax(q['boost_votes_'+split])}")
        finish(fig,'different_response')
    fig,axs=plt.subplots(1,3,figsize=(7,2.5),sharex=True,sharey=True)
    for ax,model in zip(axs,['tree','bagging','boosting']):
        cm=confusion_matrix(pred['labels'],pred[model+'_clean'].argmax(1),labels=np.arange(40))
        ax.imshow(cm,vmin=0,vmax=2,cmap='Blues'); ax.set_title(model.title()); ax.set_xlabel('Predicted identity'); ax.set_ylabel('True identity'); ax.set_xticks([0,10,20,30,39]); ax.set_yticks([0,10,20,30,39])
    finish(fig,'confusions')
    print('Figures and tables regenerated from saved artifacts')

if __name__=='__main__': main()
