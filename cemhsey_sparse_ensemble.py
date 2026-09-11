from __future__ import annotations

import json
import numpy as np
from remotezip import RemoteZip

import cemhsey_longitudinal as m


def robust_consensus(P):
    med = np.median(P, axis=0)
    mad = np.median(np.abs(P - med[None, :]), axis=0) + 1e-9
    keep = np.abs(P - med[None, :]) <= 3.0 * mad[None, :]
    num = np.sum(np.where(keep, P, 0.0), axis=0)
    den = np.maximum(np.sum(keep, axis=0), 1)
    return num / den, np.mean(keep, axis=1)


def train_single_channel_models(X, y, channels):
    return [m.fit_model(X, y, [int(ch)]) for ch in channels]


def prediction_matrix(models, X):
    return np.vstack([m.predict(f, X) for f in models])


def main():
    trials={}
    with RemoteZip(m.URL) as rz:
        for day in range(1,12):
            for tr in (1,2):
                emg,force,mvc=m.load_trial(rz,day,tr)
                X,y,t=m.featurize(emg,force)
                trials[(day,tr)]=(X,y,t,mvc)
                print('LOADED',day,tr,flush=True)
    Xtr,ytr,_,_=trials[(1,1)]
    Xval,yval,_,_=trials[(1,2)]
    rank,_=m.rank_channels(Xtr,ytr)
    out={'subject':1,'protocol':'single-channel models fit on Day1 Trial1; all ensemble choices use Day1 only; Days2-11 untouched','sets':{}}
    for k in (8,16):
        ch=rank[:k]
        models=train_single_channel_models(Xtr,ytr,ch)
        Pv=prediction_matrix(models,Xval)
        val_rmse=np.sqrt(np.mean((Pv-yval[None,:])**2,axis=1))
        weights=1.0/(val_rmse**2+1e-8); weights/=weights.sum()
        per_day={}
        for day in range(2,12):
            ys=[]; Ps=[]
            for tr in (1,2):
                X,y,_,_=trials[(day,tr)]
                ys.append(y); Ps.append(prediction_matrix(models,X))
            y=np.concatenate(ys); P=np.concatenate(Ps,axis=1)
            median=np.median(P,axis=0)
            weighted=np.sum(weights[:,None]*P,axis=0)
            consensus,ret=robust_consensus(P)
            # trimmed mean: remove one high and one low prediction at each time point.
            sortP=np.sort(P,axis=0)
            trimmed=np.mean(sortP[1:-1],axis=0) if k>2 else median
            individual=[m.metrics(y,P[i]) for i in range(k)]
            per_day[str(day)]={
                'median':m.metrics(y,median),
                'weighted_day1_validation':m.metrics(y,weighted),
                'consensus_3mad':m.metrics(y,consensus),
                'trimmed_mean':m.metrics(y,trimmed),
                'individual_r2':[x['r2'] for x in individual],
                'consensus_retention_fraction_by_channel':[float(x) for x in ret],
            }
        summary={}
        for method in ('median','weighted_day1_validation','consensus_3mad','trimmed_mean'):
            vals=[per_day[str(d)][method]['r2'] for d in range(2,12)]
            summary[method]={'median_r2':float(np.median(vals)),'min_r2':float(np.min(vals)),'max_r2':float(np.max(vals))}
        out['sets'][str(k)]={
            'channels_1based':[int(x+1) for x in ch],
            'day1_trial2_single_channel_rmse':[float(x) for x in val_rmse],
            'day1_validation_weights':[float(x) for x in weights],
            'per_day':per_day,'summary':summary,
        }
        print('ENSEMBLE',k,json.dumps(summary),flush=True)
    p=m.OUT/'cemhsey_s1_sparse_ensemble.json'
    p.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('ENSEMBLE_JSON',json.dumps(out),flush=True)

if __name__=='__main__':
    main()
