import json, os, pathlib
import numpy as np, wfdb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
ROOT=pathlib.Path('_data'); OUT=pathlib.Path('results'); OUT.mkdir(exist_ok=True)
SUB=os.environ['SUBJECT']; COMBOS=[1,8,15]; KLIST=[256,64,16,8,4]
def load(s,c,k,kind):
 p=ROOT/f'subject{SUB}_session{s}'/f'ndof_{kind}_combination{c}_sample{k}'; r=wfdb.rdrecord(str(p)); return np.asarray(r.p_signal,float),float(r.fs)
def feat(s,c,k):
 e,fe=load(s,c,k,'preprocess'); y,fy=load(s,c,k,'force'); w=int(round(.2*fe)); n=len(e)//w; e=e[:n*w].reshape(n,w,256)
 M=np.mean(np.abs(e),1); R=np.sqrt(np.mean(e*e,1)); W=np.sum(np.abs(np.diff(e,axis=1)),1); X=np.c_[M,R,W]
 tc=(np.arange(n)*w+(w-1)/2)/fe; yf=np.linalg.norm(y,axis=1); z=np.interp(tc,np.arange(len(yf))/fy,yf); return X,z
def cat(v): return np.vstack([x for x,y in v]),np.concatenate([y for x,y in v])
def met(y,p):
 rm=float(np.sqrt(np.mean((y-p)**2))); return {'r2':float(r2_score(y,p)),'rmse':rm,'n':int(len(y))}
def columns(ch): ch=np.asarray(ch); return np.r_[ch,ch+256,ch+512]
tr=[]; wi=[]; cr=[]
for c in COMBOS:
 tr.append(feat(1,c,1)); wi.append(feat(1,c,2)); cr += [feat(2,c,1),feat(2,c,2)]
Xt,yt=cat(tr); M=Xt[:,:256]; yc=yt-yt.mean(); den=np.linalg.norm(yc)+1e-12
scores=np.array([abs(np.dot(M[:,j]-M[:,j].mean(),yc)/((np.linalg.norm(M[:,j]-M[:,j].mean())+1e-12)*den)) for j in range(256)])
rank=np.argsort(scores)[::-1]; out={'subject':SUB,'channels':{}}
for K in KLIST:
 co=columns(rank[:K]); mu=Xt[:,co].mean(0); sd=Xt[:,co].std(0); sd[sd<1e-12]=1; m=Ridge(alpha=10).fit((Xt[:,co]-mu)/sd,yt)
 def pp(v):
  X,y=cat(v); return y,m.predict((X[:,co]-mu)/sd)
 yw,pw=pp(wi); yz,pz=pp(cr); out['channels'][str(K)]={'within':met(yw,pw),'cross':met(yz,pz),'delta_r2':float(r2_score(yz,pz)-r2_score(yw,pw)),'top_channels_1based':[int(x+1) for x in rank[:min(K,8)]]}
print('RESULTS_JSON',json.dumps(out),flush=True); (OUT/f'subject_{SUB}.json').write_text(json.dumps(out,indent=2))