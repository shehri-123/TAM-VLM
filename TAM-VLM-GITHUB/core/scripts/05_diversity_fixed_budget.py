"""Trigger-diversity scaling with exact fixed TRAIN and VALIDATION positive budgets.

Controls:
- train positives: exactly 5,000 at every k
- validation positives: exactly 1,000 at every k
- clean train/validation pools fixed
- chosen trigger sets nested within held-trigger/replicate
- per-trigger shuffled pools fixed before k is varied
- same model seed across k within a paired held-trigger/replicate
"""
from dataclasses import asdict
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
from final_config import *
from experiment_utils import *


def needs(total,n):
    q,r=divmod(total,n); return [q+(i<r) for i in range(n)]


def build_index(chosen,pools,permutations,total):
    parts=[]
    for t,n in zip(chosen,needs(total,len(chosen))):
        if len(pools[t])<n: raise ValueError(f'{t}: need {n}, available {len(pools[t])}')
        parts.append(permutations[t][:n])
    out=np.concatenate(parts)
    if len(out)!=total: raise AssertionError((len(out),total))
    return out


def main():
    device=resolve_device('cuda'); df,X,y=load_inputs(MANIFEST,EMB_DIR/'embeddings.npy')
    sp=df.split.to_numpy(); trigs=df.trigger.to_numpy()
    tr_clean=np.flatnonzero((sp=='train')&(trigs=='none')); va_clean=np.flatnonzero((sp=='val')&(trigs=='none'))
    te_clean=np.flatnonzero((sp=='test')&(trigs=='none'))
    tr_pool={t:np.flatnonzero((sp=='train')&(trigs==t)) for t in ALL_TRIGGERS}
    va_pool={t:np.flatnonzero((sp=='val')&(trigs==t)) for t in ALL_TRIGGERS}
    if DIVERSITY_TRAIN_POS_BUDGET>2*min(map(len,tr_pool.values())): raise ValueError('train budget infeasible at k=2')
    if DIVERSITY_VAL_POS_BUDGET>2*min(map(len,va_pool.values())): raise ValueError('validation budget infeasible at k=2')
    rows=[]
    for hi,held in enumerate(ALL_TRIGGERS):
        others=[t for t in ALL_TRIGGERS if t!=held]
        held_te=np.flatnonzero((sp=='test')&(trigs==held)); te=np.r_[te_clean,held_te]
        for rep in range(DIVERSITY_REPS):
            rng=np.random.RandomState(10000+hi*101+rep); order=list(rng.permutation(others))
            tr_perm={t:np.random.RandomState(20000+hi*1000+rep*100+ALL_TRIGGERS.index(t)).permutation(tr_pool[t]) for t in others}
            va_perm={t:np.random.RandomState(30000+hi*1000+rep*100+ALL_TRIGGERS.index(t)).permutation(va_pool[t]) for t in others}
            model_seed=40000+hi*100+rep
            for k in DIVERSITY_K:
                chosen=order[:k]
                pos_tr=build_index(chosen,tr_pool,tr_perm,DIVERSITY_TRAIN_POS_BUDGET)
                pos_va=build_index(chosen,va_pool,va_perm,DIVERSITY_VAL_POS_BUDGET)
                tr=np.r_[tr_clean,pos_tr]; va=np.r_[va_clean,pos_va]
                model,info=train_detector(X,y,tr,va,seed=model_seed,device=device,batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS,patience=PATIENCE,lr=LR,weight_decay=WEIGHT_DECAY)
                sv=score_model(model,X,va,device); st=score_model(model,X,te,device); met=evaluate_scores(y[va],sv,y[te],st)
                rows.append({'held_out':held,'rep':rep,'k':k,'chosen_triggers':'|'.join(chosen),
                    'model_seed_paired_across_k':model_seed,'n_train_pos':len(pos_tr),'n_val_pos':len(pos_va),
                    **met,**asdict(info)})
                print(f"{held:18s} rep={rep} k={k:2d} AUROC={met['AUROC']:.4f} ntr={len(pos_tr)} nva={len(pos_va)}")
    raw=pd.DataFrame(rows); raw.to_csv(OUT_DIR/'diversity_raw.csv',index=False)
    raw.groupby('k').AUROC.agg(['mean','std','count']).to_csv(OUT_DIR/'diversity_summary.csv')
    per=raw.pivot_table(index='held_out',columns='k',values='AUROC',aggfunc='mean'); per.to_csv(OUT_DIR/'diversity_per_trigger.csv')
    delta=per[11]-per[2]; w=wilcoxon(delta,zero_method='wilcox',alternative='two-sided')
    nonzero=delta[delta!=0]; sign=binomtest(int((nonzero>0).sum()),len(nonzero),.5,alternative='two-sided') if len(nonzero) else None
    pd.DataFrame([{'train_positive_budget':DIVERSITY_TRAIN_POS_BUDGET,'validation_positive_budget':DIVERSITY_VAL_POS_BUDGET,
        'mean_delta':delta.mean(),'median_delta':delta.median(),'improved':int((delta>0).sum()),'n_triggers':len(delta),
        'wilcoxon_W':w.statistic,'wilcoxon_p':w.pvalue,'sign_test_p':sign.pvalue if sign else np.nan}]).to_csv(OUT_DIR/'diversity_stats.csv',index=False)

if __name__=='__main__': main()
