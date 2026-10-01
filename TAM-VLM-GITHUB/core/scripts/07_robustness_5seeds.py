"""Camera degradation robustness across all five final detector seeds.

Embeddings are cached per degradation level. Each seed's thresholds are calibrated once on clean validation data and frozen for all degraded test levels.
"""
import io
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image,ImageEnhance,ImageFilter
from sklearn.metrics import average_precision_score,roc_auc_score
from final_config import *
from experiment_utils import *

LEVELS={'Clean':dict(blur=0.0,jpeg=100,bright=1.0),'Light':dict(blur=0.8,jpeg=70,bright=0.9),'Medium':dict(blur=1.6,jpeg=45,bright=.75),'Heavy':dict(blur=2.6,jpeg=25,bright=.6)}

def degrade(im,blur,jpeg,bright):
    if blur>0: im=im.filter(ImageFilter.GaussianBlur(blur))
    if bright!=1: im=ImageEnhance.Brightness(im).enhance(bright)
    if jpeg<100:
        b=io.BytesIO(); im.save(b,'JPEG',quality=jpeg); b.seek(0); im=Image.open(b).convert('RGB')
    return im

def encode(frame,params,encoder,pre,device,cache):
    if cache.exists(): return np.load(cache,mmap_mode='r')
    arr=[]
    for s in range(0,len(frame),64):
        ims=[pre(degrade(Image.open(p).convert('RGB'),**params)) for p in frame.image.iloc[s:s+64]]
        xb=torch.stack(ims).to(device)
        with torch.inference_mode(): arr.append(encoder.encode_image(xb).float().cpu().numpy())
        print(cache.name,min(s+64,len(frame)),'/',len(frame))
    x=np.concatenate(arr); np.save(cache,x); return x

def main():
    import open_clip
    device=resolve_device('cuda'); df=pd.read_csv(MANIFEST); va=df[df.split.eq('val')].reset_index(drop=True); te=df[df.split.eq('test')].reset_index(drop=True)
    enc,_,pre=open_clip.create_model_and_transforms(CLIP_MODEL,pretrained=CLIP_WEIGHTS); enc.eval().to(device)
    Xv=encode(va,LEVELS['Clean'],enc,pre,device,CACHE_DIR/'robust_val_clean.npy'); yv=va.label.to_numpy(); yt=te.label.to_numpy()
    level_emb={name:encode(te,p,enc,pre,device,CACHE_DIR/f'robust_test_{name.lower()}.npy') for name,p in LEVELS.items()}
    rows=[]
    for seed in SEEDS_MAIN:
        model,_=load_checkpoint(CKPT_DIR/f'clip_mlp_seed{seed}.pt',Xv.shape[1],device); sv=score_array(model,Xv,device)
        tb=best_balanced_threshold(yv,sv); t1=threshold_from_validation_fpr(yv,sv,.01)
        for name,Xt in level_emb.items():
            st=score_array(model,Xt,device); row={'seed':seed,'Level':name,'AUROC':roc_auc_score(yt,st),'AUPRC':average_precision_score(yt,st),
                'threshold_balacc_clean_val':tb,'threshold_1pct_clean_val':t1}
            row.update(fixed_threshold_metrics(yt,st,tb,'bal_')); row.update(fixed_threshold_metrics(yt,st,t1,'val1_')); rows.append(row)
            print(seed,name,row['AUROC'],row['bal_F1'],row['val1_TPR'],row['val1_FPR'])
    raw=pd.DataFrame(rows); raw.to_csv(OUT_DIR/'robustness_5seeds_raw.csv',index=False)
    raw.groupby('Level').agg({c:['mean','std'] for c in ['AUROC','AUPRC','bal_F1','bal_BalAcc','bal_FPR','bal_TPR','val1_FPR','val1_TPR']}).to_csv(OUT_DIR/'robustness_5seeds_summary.csv')

if __name__=='__main__': main()
