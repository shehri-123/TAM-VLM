"""Hierarchical seed + scene bootstrap and paired MLP-vs-linear-probe differences."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from final_config import *
from experiment_utils import fixed_threshold_metrics


def sample_scene_rows(test_df, rng):
    scenes=np.array(sorted(test_df.scene.unique()))
    drawn=rng.choice(scenes,size=len(scenes),replace=True)
    groups={s:np.flatnonzero(test_df.scene.to_numpy()==s) for s in scenes}
    return np.concatenate([groups[s] for s in drawn])


def summarize(a):
    a=np.asarray(a,float); return {'estimate':float(np.mean(a)),'ci_low':float(np.quantile(a,.025)),'ci_high':float(np.quantile(a,.975))}


def main():
    df=pd.read_csv(MANIFEST); test_df=df[df.split.eq('test')].reset_index(drop=True)
    main=pd.read_csv(OUT_DIR/'main_5seeds.csv')
    seed_data={}
    for seed in SEEDS_MAIN:
        z=np.load(OUT_DIR/f'main_predictions_seed{seed}.npz')
        seed_data[seed]=(z['test_labels'],z['test_scores'],float(main.loc[main.seed.eq(seed),'threshold_balacc_val'].iloc[0]),
                         float(main.loc[main.seed.eq(seed),'threshold_val_1pct_fpr'].iloc[0]))
    rng=np.random.RandomState(20260720); vals={k:[] for k in ['AUROC','AUPRC','bal_BalAcc','bal_FPR','bal_TPR','val1_FPR','val1_TPR']}
    for _ in range(BOOTSTRAP_REPS):
        seed=int(rng.choice(SEEDS_MAIN)); y,s,tb,t1=seed_data[seed]; idx=sample_scene_rows(test_df,rng); yy=y[idx]; ss=s[idx]
        if len(np.unique(yy))<2: continue
        vals['AUROC'].append(roc_auc_score(yy,ss)); vals['AUPRC'].append(average_precision_score(yy,ss))
        vals['bal_BalAcc'].append(balanced_accuracy_score(yy,ss>tb))
        vals['bal_FPR'].append((ss[yy==0]>tb).mean()); vals['bal_TPR'].append((ss[yy==1]>tb).mean())
        vals['val1_FPR'].append((ss[yy==0]>t1).mean()); vals['val1_TPR'].append((ss[yy==1]>t1).mean())
    pd.DataFrame([{'metric':k,**summarize(v)} for k,v in vals.items()]).to_csv(OUT_DIR/'main_hierarchical_scene_bootstrap_ci.csv',index=False)

    b=pd.read_csv(OUT_DIR/'strong_baselines.csv'); z=np.load(OUT_DIR/'baseline_predictions.npz')
    lp='linear_probe_clip'; ours='tam-vlm_detector_seed_0'
    lp_s=z[f'{lp}_test']; our_s=z[f'{ours}_test']; y=z['test_labels']
    lp_tb=float(b.loc[b.Method.eq('Linear probe (CLIP)'),'threshold_balacc_val'].iloc[0]); lp_t1=float(b.loc[b.Method.eq('Linear probe (CLIP)'),'threshold_val_1pct_fpr'].iloc[0])
    our_tb=float(b.loc[b.Method.eq('TAM-VLM detector (seed 0)'),'threshold_balacc_val'].iloc[0]); our_t1=float(b.loc[b.Method.eq('TAM-VLM detector (seed 0)'),'threshold_val_1pct_fpr'].iloc[0])
    diffs={'AUROC_delta':[],'val1_TPR_delta':[],'bal_BalAcc_delta':[]}
    rng=np.random.RandomState(20260721)
    for _ in range(BOOTSTRAP_REPS):
        idx=sample_scene_rows(test_df,rng); yy=y[idx]
        if len(np.unique(yy))<2: continue
        diffs['AUROC_delta'].append(roc_auc_score(yy,our_s[idx])-roc_auc_score(yy,lp_s[idx]))
        diffs['val1_TPR_delta'].append((our_s[idx][yy==1]>our_t1).mean()-(lp_s[idx][yy==1]>lp_t1).mean())
        diffs['bal_BalAcc_delta'].append(balanced_accuracy_score(yy,our_s[idx]>our_tb)-balanced_accuracy_score(yy,lp_s[idx]>lp_tb))
    pd.DataFrame([{'metric':k,**summarize(v),'probability_gt_0':float((np.asarray(v)>0).mean())} for k,v in diffs.items()]).to_csv(OUT_DIR/'paired_mlp_vs_linear_probe_scene_bootstrap.csv',index=False)

if __name__=='__main__': main()
