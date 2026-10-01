"""Publication audit for v3 inputs and outputs."""
import json
import numpy as np,pandas as pd
from final_config import *
from experiment_utils import *

def main():
    df,X,y=load_inputs(MANIFEST,EMB_DIR/'embeddings.npy'); checks=[]
    def chk(name,cond,detail=''):
        checks.append({'check':name,'pass':bool(cond),'detail':str(detail)}); print('PASS' if cond else 'FAIL',name,detail)
    chk('rows',len(df)==48633,len(df)); chk('source images',int(df.trigger.eq('none').sum())==3741)
    chk('trigger set',set(df.loc[~df.trigger.eq('none'),'trigger'])==set(ALL_TRIGGERS))
    chk('embedding alignment',len(df)==len(X),X.shape); chk('finite embeddings',np.isfinite(X).all()); chk('nonzero embeddings',np.linalg.norm(X,axis=1).min()>1e-8)
    scenes={s:set(df.loc[df.split.eq(s),'scene']) for s in ('train','val','test')}
    chk('train-val scenes disjoint',not scenes['train']&scenes['val']); chk('train-test scenes disjoint',not scenes['train']&scenes['test']); chk('val-test scenes disjoint',not scenes['val']&scenes['test'])
    temp=df.assign(_base=df.image.map(base_image_id)); chk('all variants share split',int((temp.groupby('_base').split.nunique()>1).sum())==0)
    chk('13 variants/source',bool((temp.groupby('_base').trigger.nunique()==13).all()))
    req=['main_5seeds.csv','strong_baselines.csv','main_hierarchical_scene_bootstrap_ci.csv','paired_mlp_vs_linear_probe_scene_bootstrap.csv','loto_raw.csv','diversity_raw.csv','diversity_stats.csv','winclip_test.csv','robustness_5seeds_raw.csv']
    for n in req: chk('exists '+n,(OUT_DIR/n).exists())
    if (OUT_DIR/'diversity_raw.csv').exists():
        d=pd.read_csv(OUT_DIR/'diversity_raw.csv'); chk('div train budget exact',bool((d.n_train_pos==DIVERSITY_TRAIN_POS_BUDGET).all()),d.n_train_pos.unique()); chk('div val budget exact',bool((d.n_val_pos==DIVERSITY_VAL_POS_BUDGET).all()),d.n_val_pos.unique()); chk('div runs',len(d)==12*5*DIVERSITY_REPS,len(d))
        paired=d.groupby(['held_out','rep']).model_seed_paired_across_k.nunique(); chk('same seed paired across k',bool((paired==1).all()))
    if (OUT_DIR/'winclip_prompt_selection.csv').exists():
        w=pd.read_csv(OUT_DIR/'winclip_prompt_selection.csv'); chk('one WinCLIP selected',int(w.selected.sum())==1); chk('no test selection columns',not any(c.startswith('test') for c in w.columns))
    report={'manifest_sha256':sha256_file(MANIFEST),'authoritative_scene_count':int(df.scene.nunique()),'split_scenes':{k:len(v) for k,v in scenes.items()},'checks':checks,'passed':sum(c['pass'] for c in checks),'total':len(checks)}
    with open(OUT_DIR/'AUDIT_V3.json','w') as f: json.dump(report,f,indent=2)
    if report['passed']!=report['total']: raise SystemExit(f"AUDIT FAILED {report['passed']}/{report['total']}")
    print(f"AUDIT PASS {report['passed']}/{report['total']}")
if __name__=='__main__': main()
