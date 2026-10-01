"""Unified baselines. No prompt/configuration is selected on test data."""
import time
import numpy as np, pandas as pd, torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM

from final_config import *
from experiment_utils import *


def main():
    device=resolve_device('cuda')
    df,X,y=load_inputs(MANIFEST,EMB_DIR/'embeddings.npy')
    trc=((df.split=='train')&(df.label==0)).to_numpy(); tra=(df.split=='train').to_numpy()
    va=(df.split=='val').to_numpy(); te=(df.split=='test').to_numpy()
    Xc=np.asarray(X[trc]); mu=Xc.mean(0); mu_n=mu/(np.linalg.norm(mu)+1e-12)
    def euc(q): return np.linalg.norm(q-mu,axis=1)
    def cos(q):
        qn=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12); return 1-qn@mu_n
    precision=np.linalg.inv(np.cov(Xc,rowvar=False)+1e-3*np.eye(Xc.shape[1]))
    def mah(q):
        d=q-mu; return np.sqrt(np.einsum('ij,jk,ik->i',d,precision,d))
    Xct=torch.as_tensor(Xc,dtype=torch.float32,device=device)
    def knn(q,k=10):
        qt=torch.as_tensor(np.asarray(q),dtype=torch.float32,device=device); out=[]
        with torch.inference_mode():
            for s in range(0,len(qt),256):
                out.append(torch.cdist(qt[s:s+256],Xct).topk(k,largest=False).values.mean(1).cpu().numpy())
        return np.concatenate(out)
    iso=IsolationForest(n_estimators=500,random_state=42,n_jobs=-1).fit(Xc)
    rng=np.random.RandomState(42); sub=Xc[rng.choice(len(Xc),min(3000,len(Xc)),replace=False)]
    svm=OneClassSVM(kernel='rbf',gamma='scale',nu=0.1).fit(sub)
    lp=LogisticRegression(max_iter=5000,random_state=42).fit(np.asarray(X[tra]),y[tra])
    methods=[('Euclidean distance',euc),('Cosine distance',cos),('Mahalanobis distance',mah),
             ('kNN distance (k=10)',knn),('Isolation Forest',lambda q:-iso.score_samples(q)),
             ('One-Class SVM',lambda q:-svm.decision_function(q)),
             ('Linear probe (CLIP)',lambda q:lp.predict_proba(q)[:,1])]
    try:
        import open_clip
        m,_,_=open_clip.create_model_and_transforms(CLIP_MODEL,pretrained=CLIP_WEIGHTS); m.eval().to(device)
        tok=open_clip.get_tokenizer(CLIP_MODEL)
        pos=['a driving scene containing an unusual object placed on the road','a road scene with a suspicious added object']
        neg=['a normal driving scene from a front car camera','an ordinary street photo taken from a vehicle']
        with torch.inference_mode():
            ep=m.encode_text(tok(pos).to(device)).float(); en=m.encode_text(tok(neg).to(device)).float()
        ep=(ep/ep.norm(dim=-1,keepdim=True)).mean(0); en=(en/en.norm(dim=-1,keepdim=True)).mean(0)
        d=(ep-en).cpu().numpy(); d/=np.linalg.norm(d)+1e-12
        methods.append(('CLIP zero-shot text',lambda q:(q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12))@d))
    except Exception as exc: print('[warning] zero-shot skipped:',exc)
    ours,_=load_checkpoint(CKPT_DIR/'clip_mlp_seed0.pt',X.shape[1],device)
    methods.append(('TAM-VLM detector (seed 0)',lambda q:score_array(ours,q,device)))

    rows=[]; pred_store={}
    for name,fn in methods:
        sv=fn(np.asarray(X[va])); t0=time.perf_counter(); st=fn(np.asarray(X[te])); elapsed=time.perf_counter()-t0
        met=evaluate_scores(y[va],sv,y[te],st)
        rows.append({'Method':name,**met,'score_ms_per_sample':1000*elapsed/int(te.sum())})
        safe=name.lower().replace(' ','_').replace('/','_').replace('(','').replace(')','').replace('%','pct')
        pred_store[f'{safe}_val']=sv; pred_store[f'{safe}_test']=st
        print(f"{name:28s} AUROC={met['AUROC']:.4f} val1TPR={met['val1_TPR']:.4f} FPR={met['val1_FPR']:.4f}")
    pd.DataFrame(rows).to_csv(OUT_DIR/'strong_baselines.csv',index=False)
    np.savez_compressed(OUT_DIR/'baseline_predictions.npz',val_labels=y[va],test_labels=y[te],
                        val_indices=np.flatnonzero(va),test_indices=np.flatnonzero(te),**pred_store)

if __name__=='__main__': main()
