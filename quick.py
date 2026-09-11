import json, pathlib, urllib.request
import numpy as np, wfdb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE='https://physionet.org/files/hd-semg/2.0.0/ndof_dataset'
ROOT=pathlib.Path('_quick_data'); ROOT.mkdir(exist_ok=True)
OUT=pathlib.Path('results'); OUT.mkdir(exist_ok=True)
SUB='01'; COMBO=15

def dl(sess, kind, sample):
    d=ROOT/f'subject{SUB}_session{sess}'; d.mkdir(exist_ok=True)
    stem=f'ndof_{kind}_combination{COMBO}_sample{sample}'
    for ext in ('hea','dat'):
        p=d/f'{stem}.{ext}'
        if not p.exists():
            u=f'{BASE}/subject{SUB}_session{sess}/{stem}.{ext}'
            print('DOWNLOAD',u,flush=True)
            req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=240) as r, open(p,'wb') as f:
                while True:
                    b=r.read(4*1024*1024)
                    if not b: break
                    f.write(b)
    rec=wfdb.rdrecord(str(d/stem))
    return np.asarray(rec.p_signal,dtype=float), float(rec.fs)

def feat(sess,sample):
    e,fe=dl(sess,'preprocess',sample); y,fy=dl(sess,'force',sample)
    # 200-ms nonoverlapping windows for a fast diagnostic.
    w=int(round(.2*fe)); n=e.shape[0]//w
    e=e[:n*w].reshape(n,w,e.shape[1])
    X=np.concatenate([np.mean(np.abs(e),axis=1),np.sqrt(np.mean(e*e,axis=1)),np.sum(np.abs(np.diff(e,axis=1)),axis=1)],axis=1)
    tc=(np.arange(n)*w+(w-1)/2)/fe
    yf=np.linalg.norm(y,axis=1); tf=np.arange(len(yf))/fy
    z=np.interp(tc,tf,yf)
    print('TRIAL',sess,sample,X.shape,z.shape,float(z.min()),float(z.max()),flush=True)
    return X,z

def met(y,p):
    rm=float(np.sqrt(np.mean((y-p)**2))); sd=float(np.std(y)); rg=float(np.ptp(y))
    return {'r2':float(r2_score(y,p)),'rmse':rm,'nrmse_sd':rm/sd,'nrmse_range':rm/rg,'n':int(len(y))}

Xtr,ytr=feat(1,1); Xw,yw=feat(1,2); X21,y21=feat(2,1); X22,y22=feat(2,2)
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-12]=1
m=Ridge(alpha=10.0).fit((Xtr-mu)/sd,ytr)
pw=m.predict((Xw-mu)/sd); p21=m.predict((X21-mu)/sd); p22=m.predict((X22-mu)/sd)
yc=np.r_[y21,y22]; pc=np.r_[p21,p22]
res={'dataset':'Hyser v2 N-DoF','subject':'01','combination':15,'train':'session1 sample1','within':'session1 sample2','cross':'session2 samples1+2','within':met(yw,pw),'cross':met(yc,pc),'delta_r2':met(yc,pc)['r2']-met(yw,pw)['r2']}
print('RESULTS_JSON',json.dumps(res,indent=2),flush=True)
(OUT/'quick.json').write_text(json.dumps(res,indent=2))
