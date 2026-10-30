"""Small scratch ensembles. No sklearn ensemble wrappers."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array

def samme_round(weights, incorrect, k):
    """Return error, alpha and normalized exp(alpha * incorrect) update."""
    w = np.asarray(weights,dtype=float)
    bad = np.asarray(incorrect,dtype=bool)
    if k < 2 or w.shape != bad.shape or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('Invalid weights or class count')
    w = w/w.sum()
    error = float(w @ bad)
    if error >= (k-1)/k - 8*np.finfo(float).eps:
        return error, None, w.copy()
    # Machine epsilon cap gives a finite perfect-learner coefficient.
    # Fitting stops after acceptance; no infinite scores enter normalization.
    e = max(error, np.finfo(float).eps)
    alpha = float(np.log1p(-e)-np.log(e)+np.log(k-1))
    logw = np.full(w.shape,-np.inf)
    np.log(w,out=logw,where=w>0)
    logw += alpha*bad
    logw -= np.max(logw)
    updated = np.exp(logw)
    updated /= updated.sum()
    return error,alpha,updated

class Votes:
    def _check_predict(self,x):
        if not getattr(self,'estimators_',[]):
            raise ValueError('No fitted learners')
        x=check_array(x)
        if x.shape[1] != self.n_features_in_: raise ValueError('Feature count mismatch')
        return x

    def predict_proba(self,x,n=None):
        x=self._check_predict(x)
        n=len(self.estimators_) if n is None else n
        if not 1 <= n <= len(self.estimators_): raise ValueError('Invalid prefix')
        scores=np.zeros((len(x),len(self.classes_)))
        for tree,a in zip(self.estimators_[:n],self.weights_[:n]):
            # Map actual labels, since bootstrap trees can miss entire classes.
            j=np.searchsorted(self.classes_,tree.predict(x))
            scores[np.arange(len(x)),j]+=a
        return scores/np.sum(self.weights_[:n])

    def predict(self,x,n=None):
        # np.argmax breaks ties at the first (smallest sorted) class label.
        return self.classes_[self.predict_proba(x,n).argmax(axis=1)]

    def _start(self,x,y):
        x,y=check_X_y(x,y)
        if not isinstance(self.n_estimators,int) or self.n_estimators<1: raise ValueError('n_estimators must be positive integer')
        if self.max_depth is not None and (not isinstance(self.max_depth,int) or self.max_depth<1): raise ValueError('Invalid depth')
        if not isinstance(self.min_samples_leaf,int) or self.min_samples_leaf<1: raise ValueError('Invalid leaf size')
        self.classes_=np.unique(y)
        if len(self.classes_)<2: raise ValueError('Need >=2 classes')
        self.n_features_in_=x.shape[1]
        self.estimators_=[]; self.weights_=[]
        return x,y,np.random.default_rng(self.random_state)

class Bagging(Votes):
    def __init__(self,n_estimators=100,max_depth=None,min_samples_leaf=1,random_state=0):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf; self.random_state=random_state

    def fit(self,x,y):
        x,y,rng=self._start(x,y)
        self.bootstrap_indices_=[]; self.fit_seconds_=[]
        from time import perf_counter
        for _ in range(self.n_estimators):
            start=perf_counter()
            ix=rng.integers(0,len(x),size=len(x))
            tree=DecisionTreeClassifier(max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,random_state=int(rng.integers(2**31-1)))
            tree.fit(x[ix],y[ix])
            self.estimators_.append(tree); self.weights_.append(1.)
            self.bootstrap_indices_.append(ix)
            self.fit_seconds_.append(perf_counter()-start)
        return self

class SAMME(Votes):
    def __init__(self,n_estimators=100,max_depth=1,min_samples_leaf=1,random_state=0):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf; self.random_state=random_state

    def fit(self,x,y):
        x,y,rng=self._start(x,y)
        w=np.full(len(x),1/len(x)); self.trace_=[]; self.sample_weights_=[]; self.training_predictions_=[]; self.fit_seconds_=[]
        self.stop_reason_='round_limit'
        from time import perf_counter
        for t in range(self.n_estimators):
            start=perf_counter()
            tree=DecisionTreeClassifier(max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,random_state=int(rng.integers(2**31-1)))
            tree.fit(x,y,sample_weight=w)
            pred=tree.predict(x)
            error,a,updated=samme_round(w,pred!=y,len(self.classes_))
            if a is None:
                self.stop_reason_='not_better_than_random'
                if not self.estimators_: raise ValueError('First learner is not better than random')
                break
            self.estimators_.append(tree); self.weights_.append(a)
            self.training_predictions_.append(pred)
            self.trace_.append(dict(round=t+1,error=error,alpha=a,max_weight=float(updated.max()),top10_weight=float(np.sort(updated)[-10:].sum())))
            self.sample_weights_.append(updated.copy())
            self.fit_seconds_.append(perf_counter()-start)
            w=updated
            if error<=np.finfo(float).eps:
                self.stop_reason_='perfect_learner'; break
        return self
