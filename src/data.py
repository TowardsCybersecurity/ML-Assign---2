"""Raw Olivetti pixels and one persisted identity-stratified split."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROOT = Path(__file__).resolve().parents[1]

def seed_from_roll(roll):
    digits = ''.join(c for c in roll if c.isdigit())
    if not digits:
        raise ValueError('Roll must contain digits')
    return int(digits)

def dataset():
    d = fetch_olivetti_faces(shuffle=False)
    assert d.data.shape == (400, 4096)
    assert np.array_equal(np.unique(d.target, return_counts=True)[1], np.full(40,10))
    assert np.array_equal(d.data, d.images.reshape(400,4096))
    return d.data, d.target

def fingerprint(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def make_split(y, roll):
    rng = np.random.default_rng(seed_from_roll(roll))
    out = dict(train=[], validation=[], test=[])
    for identity in np.unique(y):
        ix = rng.permutation(np.flatnonzero(y == identity))
        for name, values in zip(out, (ix[:6], ix[6:8], ix[8:])):
            out[name].extend(values.tolist())
    return out

def verify_split(s, y):
    all_ix = []
    for name, n in [('train',6),('validation',2),('test',2)]:
        ix = np.array(s[name])
        assert len(ix) == 40*n
        assert np.array_equal(np.bincount(y[ix],minlength=40), np.full(40,n))
        all_ix.extend(ix.tolist())
    assert sorted(all_ix) == list(range(400))

def prepare():
    x,y = dataset()
    path = ROOT/'artifacts/split.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        s = json.loads(path.read_text())
    else:
        s = make_split(y,'P26CS0012')
        s.update(seed=260012,roll='P26CS0012',dataset_sha256=fingerprint(x),label_sha256=fingerprint(y),order='fetch_olivetti_faces(shuffle=False); ascending identities')
        path.write_text(json.dumps(s,indent=2)+'\n')
    verify_split(s,y)
    assert s['dataset_sha256'] == fingerprint(x)
    assert s['seed'] == seed_from_roll(s['roll']) == 260012
    return x,y,s

def occlude(x):
    z = x.reshape(-1,64,64).copy()
    z[:,24:40,24:40] = 0
    return z.reshape(-1,4096)

if __name__ == '__main__':
    _,_,s = prepare()
    print('Verified saved split:', {k:len(s[k]) for k in ('train','validation','test')},s['dataset_sha256'])
