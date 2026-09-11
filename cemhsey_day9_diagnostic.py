from __future__ import annotations

import io
import json
import pathlib

import numpy as np
from remotezip import RemoteZip
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

URL = "https://zenodo.org/api/records/15077957/files/GRASP_S1.zip/content"
FS = 2048.0
WIN = int(round(0.200 * FS))
HOP = int(round(0.100 * FS))
ALPHAS = np.logspace(-3, 5, 17)
OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)


def fname(day, trial):
    return f"S1/D{day}/S1_Day{day}_Session1_Task2_Trial{trial}.mat"


def read_trial(rz, day, trial):
    d = loadmat(io.BytesIO(rz.read(fname(day, trial))))
    emg = np.asarray(d["data_sEMG"], float)
    f = np.asarray(d["data_force"], float).ravel()
    starts = np.arange(0, emg.shape[1] - WIN + 1, HOP, dtype=int)
    centers = (starts + (WIN-1)/2) / FS
    dur = emg.shape[1]/FS
    ft = np.linspace(0, dur, f.size, endpoint=False)
    y = np.interp(centers, ft, f)
    n, ch = len(starts), emg.shape[0]
    mav=np.empty((n,ch)); rms=np.empty((n,ch)); wl=np.empty((n,ch))
    for i,s in enumerate(starts):
        w=emg[:,s:s+WIN]
        mav[i]=np.mean(np.abs(w),axis=1)
        rms[i]=np.sqrt(np.mean(w*w,axis=1))
        wl[i]=np.sum(np.abs(np.diff(w,axis=1)),axis=1)
    X=np.concatenate([mav,rms,wl],axis=1)
    return X,y


def rank_channels(X,y):
    yc=y-y.mean(); yn=np.linalg.norm(yc)+1e-12
    scores=[]
    for j in range(320):
        x=X[:,j]-X[:,j].mean()
        scores.append(abs(float(np.dot(x,yc)/((np.linalg.norm(x)+1e-12)*yn))))
    return np.argsort(scores)[::-1], np.asarray(scores)


def cols(ch):
    c=np.asarray(ch,int)
    return np.concatenate([c,c+320,c+640])


def fit_predict(Xtr,ytr,Xte,ch):
    c=cols(ch); A=Xtr[:,c]; B=Xte[:,c]
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-12]=1
    model=RidgeCV(alphas=ALPHAS).fit((A-mu)/sd,ytr)
    return model.predict((B-mu)/sd), float(model.alpha_)


def met(y,p):
    return {"r2":float(r2_score(y,p)),"rmse":float(np.sqrt(np.mean((y-p)**2)))}


def main():
    with RemoteZip(URL) as rz:
        X11,y11=read_trial(rz,1,1)
        X12,y12=read_trial(rz,1,2)
        X91,y91=read_trial(rz,9,1)
        X92,y92=read_trial(rz,9,2)
    X9=np.vstack([X91,X92]); y9=np.concatenate([y91,y92])
    rank,scores=rank_channels(X11,y11)
    top8=rank[:8]
    out={"top8_1based":[int(x+1) for x in top8],"incremental":{},"leave_one_out":{},"channel_shift":{}}
    for k in range(1,9):
        p,a=fit_predict(X11,y11,X9,top8[:k])
        pw,aw=fit_predict(X11,y11,X12,top8[:k])
        out["incremental"][str(k)]={"day9":met(y9,p),"within_day":met(y12,pw),"alpha":a}
    for j in top8:
        subset=[x for x in top8 if x!=j]
        p,a=fit_predict(X11,y11,X9,subset)
        out["leave_one_out"][str(int(j+1))]={"day9":met(y9,p),"alpha":a}
    for j in top8:
        rec={}
        for feat,off in (("mav",0),("rms",320),("wl",640)):
            a=X11[:,j+off]; b=X9[:,j+off]
            rec[feat]={
                "train_mean":float(a.mean()),"day9_mean":float(b.mean()),
                "mean_ratio":float(b.mean()/(a.mean()+1e-12)),
                "standardized_mean_shift":float((b.mean()-a.mean())/(a.std()+1e-12)),
                "train_sd":float(a.std()),"day9_sd":float(b.std())
            }
        p,a=fit_predict(X11,y11,X9,[j])
        rec["single_channel_day9"]=met(y9,p)
        rec["day1_rank_score"]=float(scores[j])
        out["channel_shift"][str(int(j+1))]=rec
    path=OUT/"cemhsey_s1_day9_diagnostic.json"
    path.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("DAY9_DIAGNOSTIC",json.dumps(out),flush=True)

if __name__=="__main__":
    main()
