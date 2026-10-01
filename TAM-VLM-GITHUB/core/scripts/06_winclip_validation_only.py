"""Simplified WinCLIP-style baseline. Prompt family selected on validation only."""
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from final_config import *
from experiment_utils import evaluate_scores, resolve_device

TOPK=8
TEMPLATES=['a photo of {}','a close-up photo of {}','a cropped photo of {}','a road scene with {}']
PROMPTS={
 'generic':(['an unusual object placed on the road','a suspicious object added to the scene','an out-of-place object in the street'],['a normal driving scene','an ordinary street with nothing unusual','a clean road ahead']),
 'state_words':(['an abnormal road scene','a damaged scene','a scene with a defect','an anomalous street view'],['a normal road scene','a flawless scene','a scene without defect','a typical street view']),
 'object_presence':(['a street with an extra object that does not belong','a road with a foreign object on it'],['a street with only vehicles and road markings','a road with nothing on it'])}


def main():
    import open_clip
    device=resolve_device('cuda'); df=pd.read_csv(MANIFEST); y=df.label.to_numpy()
    tok_path=EMB_DIR/'patch_tokens.npy'
    if not tok_path.exists(): raise FileNotFoundError(tok_path)
    T=np.load(tok_path,mmap_mode='r'); va=df.split.eq('val').to_numpy(); te=df.split.eq('test').to_numpy()
    m,_,_=open_clip.create_model_and_transforms(CLIP_MODEL,pretrained=CLIP_WEIGHTS); m.eval().to(device)
    tokenizer=open_clip.get_tokenizer(CLIP_MODEL); projection=m.visual.proj.detach().float().to(device)
    def direction(pos,neg):
        def enc(ps):
            prompts=[temp.format(p) for p in ps for temp in TEMPLATES]
            with torch.inference_mode(): e=m.encode_text(tokenizer(prompts).to(device)).float()
            e=e/e.norm(dim=-1,keepdim=True); e=e.mean(0); return e/e.norm()
        d=enc(pos)-enc(neg); return d/d.norm()
    def score(mask,d):
        idx=np.flatnonzero(mask); out=np.empty(len(idx),np.float32)
        with torch.inference_mode():
            for s in range(0,len(idx),256):
                part=idx[s:s+256]; x=torch.as_tensor(np.asarray(T[part]),dtype=torch.float32,device=device)
                x=x@projection; x=x/(x.norm(dim=-1,keepdim=True)+1e-8); ps=x@d
                out[s:s+len(part)]=ps.topk(min(TOPK,ps.shape[1]),dim=1).values.mean(1).cpu().numpy()
        return out
    dirs={}; cand=[]
    for name,(pos,neg) in PROMPTS.items():
        d=direction(pos,neg); dirs[name]=d; sv=score(va,d)
        cand.append({'prompt_set':name,'val_AUROC':roc_auc_score(y[va],sv),'val_AUPRC':average_precision_score(y[va],sv)})
    c=pd.DataFrame(cand).sort_values(['val_AUROC','prompt_set'],ascending=[False,True]); selected=str(c.iloc[0].prompt_set); c['selected']=c.prompt_set.eq(selected)
    c.to_csv(OUT_DIR/'winclip_prompt_selection.csv',index=False); print(c); print('selected:',selected)
    sv=score(va,dirs[selected]); st=score(te,dirs[selected]); met=evaluate_scores(y[va],sv,y[te],st)
    pd.DataFrame([{'selected_prompt_set':selected,**met}]).to_csv(OUT_DIR/'winclip_test.csv',index=False)
    clean=te&df.trigger.eq('none').to_numpy(); sc=score(clean,dirs[selected]); rows=[]
    for t in ALL_TRIGGERS:
        mask=te&df.trigger.eq(t).to_numpy(); ss=score(mask,dirs[selected]); lab=np.r_[np.zeros(len(sc)),np.ones(len(ss))]
        rows.append({'trigger':t,'AUROC':roc_auc_score(lab,np.r_[sc,ss])})
    pd.DataFrame(rows).to_csv(OUT_DIR/'winclip_per_trigger.csv',index=False)

if __name__=='__main__': main()
