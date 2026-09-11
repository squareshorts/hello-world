import json, pathlib
import numpy as np, wfdb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT=pathlib.Path('_data'); OUT=pathlib.Path('results'); OUT.mkdir(exist_ok=True)
COMBOS=[1,8,15]; SUB='01'; KLIST=[256,64,16,8,4]

def load(sess,combo,sample,kind):
    p=ROOT/f'subject{SUB}_session{sess}'/f'ndof_{kind}_combination{combo}_sample{sample}'
    r=wfdb.rdrecord(str(p)); return np.asarray(r.p_signal,float),float(r.fs)

def feat(sess,combo,sample):
    e,fe=load(sess,combo,sample,'preprocess'); y,fy=load(sess,combo,sample,'force')
    w=int(round(.2*fe)); n=e.shape[0]//w
    e=e[:n*w].reshape(n,w,256)
    mav=np.mean(np.abs(e),1); rms=np.sqrt(np.mean(e*e,1)); wl=np.sum(np.abs(np.diff(e,axis=1)),1)
    X=np.c_[mav,rms,wl]
    tc=(np.arange(n)*w+(w-1)/2)/fe; yf=np.linalg.norm(y,axis=1); tf=np.arange(len(yf))/fy
    z=np.interp(tc,tf,yf)
    return X,z,tc

def cat(items): return np.vstack([x[0] for x in items]),np.concatenate([x[1] for x in items])
def metrics(y,p):
    rm=float(np.sqrt(np.mean((y-p)**2))); sd=float(np.std(y)); rg=float(np.ptp(y))
    return {'r2':float(r2_score(y,p)),'rmse':rm,'nrmse_sd':rm/sd,'nrmse_range':rm/rg,'n':int(len(y))}
def cols(ch):
    ch=np.asarray(ch); return np.r_[ch,ch+256,ch+512]
def rankch(X,y):
    M=X[:,:256]; yc=y-y.mean(); den=np.sqrt(np.sum(yc*yc))+1e-12
    s=[]
    for j in range(256):
        x=M[:,j]-M[:,j].mean(); s.append(abs(np.dot(x,yc)/((np.sqrt(np.sum(x*x))+1e-12)*den)))
    return np.argsort(s)[::-1]
def fit(train,ch):
    X,y=cat(train); c=cols(ch); X=X[:,c]; mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1
    m=Ridge(alpha=10.0).fit((X-mu)/sd,y); return m,mu,sd,c
def pred(F,items):
    m,mu,sd,c=F; ys=[]; ps=[]
    for X,y,t in items: ys.append(y); ps.append(m.predict((X[:,c]-mu)/sd))
    return np.concatenate(ys),np.concatenate(ps)

tr=[]; wi=[]; c1=[]; c2=[]
for co in COMBOS:
    tr.append(feat(1,co,1)); wi.append(feat(1,co,2)); c1.append(feat(2,co,1)); c2.append(feat(2,co,2))
Xtr,ytr=cat(tr); ranking=rankch(Xtr,ytr)
res={'dataset':'Hyser v2 N-DoF','subject':'01','combinations':COMBOS,'features':'MAV RMS WL, 200 ms non-overlap','train':'session1 sample1','within':'session1 sample2','cross':'session2 samples1+2','channel_selection':'training-only MAV-force correlation','channels':{}}
for k in KLIST:
    F=fit(tr,ranking[:k]); yw,pw=pred(F,wi); yc,pc=pred(F,c1+c2)
    res['channels'][str(k)]={'selected_1based':[int(v+1) for v in ranking[:k]],'within':metrics(yw,pw),'cross':metrics(yc,pc),'delta_r2':float(r2_score(yc,pc)-r2_score(yw,pw))}

# calibration on session2 sample1, evaluation on sample2 only; fit affine correction from first N seconds pooled across combos.
F8=fit(tr,ranking[:8]); cal={}
for sec in [0,2,5,10]:
    if sec==0:
        y,p=pred(F8,c2); cal[str(sec)]={'eval':metrics(y,p),'slope':1.0,'intercept':0.0,'n_cal':0}
        continue
    cy=[]; cp=[]
    m,mu,sd,c=F8
    for X,y,t in c1:
        pp=m.predict((X[:,c]-mu)/sd); mask=t<=t[0]+sec; cy.append(y[mask]); cp.append(pp[mask])
    cy=np.concatenate(cy); cp=np.concatenate(cp); A=np.c_[cp,np.ones(len(cp))]; a,b=np.linalg.lstsq(A,cy,rcond=None)[0]
    y,p=pred(F8,c2); p=a*p+b
    cal[str(sec)]={'eval':metrics(y,p),'slope':float(a),'intercept':float(b),'n_cal':int(len(cy))}
res['calibration_8ch']=cal
print('RESULTS_JSON',json.dumps(res,indent=2),flush=True)
(OUT/'expanded.json').write_text(json.dumps(res,indent=2))
