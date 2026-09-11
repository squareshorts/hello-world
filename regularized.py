import json, os, pathlib
import numpy as np, wfdb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT=pathlib.Path('_data'); OUT=pathlib.Path('results'); OUT.mkdir(exist_ok=True)
SUB=os.environ['SUBJECT']; COMBOS=[1,8,15]; KLIST=[256,64,16,8,4]
ALPHAS=np.logspace(-4,6,21)

def load(s,c,k,kind):
    p=ROOT/f'subject{SUB}_session{s}'/f'ndof_{kind}_combination{c}_sample{k}'
    r=wfdb.rdrecord(str(p)); return np.asarray(r.p_signal,float),float(r.fs)

def feat(s,c,k):
    e,fe=load(s,c,k,'preprocess'); y,fy=load(s,c,k,'force')
    w=int(round(.2*fe)); n=len(e)//w; e=e[:n*w].reshape(n,w,256)
    M=np.mean(np.abs(e),1); R=np.sqrt(np.mean(e*e,1)); W=np.sum(np.abs(np.diff(e,axis=1)),1)
    X=np.c_[M,R,W]
    tc=(np.arange(n)*w+(w-1)/2)/fe; yf=np.linalg.norm(y,axis=1)
    z=np.interp(tc,np.arange(len(yf))/fy,yf)
    return X,z,c

def cat(v): return np.vstack([x for x,y,g in v]),np.concatenate([y for x,y,g in v])
def met(y,p):
    rm=float(np.sqrt(np.mean((y-p)**2)))
    return {'r2':float(r2_score(y,p)),'rmse':rm,'n':int(len(y))}
def columns(ch):
    ch=np.asarray(ch); return np.r_[ch,ch+256,ch+512]

def rank_channels(train):
    X,y=cat(train); M=X[:,:256]; yc=y-y.mean(); den=np.linalg.norm(yc)+1e-12
    s=np.array([abs(np.dot(M[:,j]-M[:,j].mean(),yc)/((np.linalg.norm(M[:,j]-M[:,j].mean())+1e-12)*den)) for j in range(256)])
    return np.argsort(s)[::-1]

def fit_scaler(X):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1; return mu,sd

def choose_alpha(train, ch):
    c=columns(ch); fold_rmse=[]
    for a in ALPHAS:
        vals=[]
        for held in COMBOS:
            tr=[v for v in train if v[2]!=held]; va=[v for v in train if v[2]==held]
            Xt,yt=cat(tr); Xv,yv=cat(va)
            mu,sd=fit_scaler(Xt[:,c]); m=Ridge(alpha=float(a)).fit((Xt[:,c]-mu)/sd,yt)
            pv=m.predict((Xv[:,c]-mu)/sd); vals.append(np.sqrt(np.mean((yv-pv)**2)))
        fold_rmse.append(float(np.mean(vals)))
    i=int(np.argmin(fold_rmse)); return float(ALPHAS[i]),fold_rmse

def fit_final(train,ch,alpha):
    X,y=cat(train); c=columns(ch); mu,sd=fit_scaler(X[:,c]); m=Ridge(alpha=alpha).fit((X[:,c]-mu)/sd,y); return m,mu,sd,c

def pred(F,v):
    m,mu,sd,c=F; X,y=cat(v); return y,m.predict((X[:,c]-mu)/sd)

train=[]; within=[]; cross=[]
for co in COMBOS:
    train.append(feat(1,co,1)); within.append(feat(1,co,2)); cross += [feat(2,co,1),feat(2,co,2)]
rank=rank_channels(train)
out={'subject':SUB,'alpha_selection':'leave-one-combination-out on session1/sample1','alphas':ALPHAS.tolist(),'channels':{}}
for K in KLIST:
    ch=rank[:K]; alpha,cv=choose_alpha(train,ch); F=fit_final(train,ch,alpha)
    yw,pw=pred(F,within); yc,pc=pred(F,cross)
    out['channels'][str(K)]={'alpha':alpha,'cv_rmse_by_alpha':cv,'within':met(yw,pw),'cross':met(yc,pc),'delta_r2':float(r2_score(yc,pc)-r2_score(yw,pw)),'top_channels_1based':[int(x+1) for x in ch[:min(8,K)]]}
print('RESULTS_JSON',json.dumps(out),flush=True)
(OUT/f'subject_{SUB}.json').write_text(json.dumps(out,indent=2))
