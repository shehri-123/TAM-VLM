"""12-trigger LOTO. Held-out trigger is absent from both training and validation."""
from dataclasses import asdict
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from final_config import *
from experiment_utils import *


def main():
    device=resolve_device('cuda'); df,X,y=load_inputs(MANIFEST,EMB_DIR/'embeddings.npy')
    split=df.split.to_numpy(); trig=df.trigger.to_numpy(); rows=[]
    for held in ALL_TRIGGERS:
        tr=np.flatnonzero((split=='train')&(trig!=held)); va=np.flatnonzero((split=='val')&(trig!=held))
        unseen=np.flatnonzero((split=='test')&np.isin(trig,['none',held]))
        seen=np.flatnonzero((split=='test')&(trig!=held))
        for seed in SEEDS_LOTO:
            model,info=train_detector(X,y,tr,va,seed=seed,device=device,batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,patience=PATIENCE,lr=LR,weight_decay=WEIGHT_DECAY)
            sv=score_model(model,X,va,device); su=score_model(model,X,unseen,device); ss=score_model(model,X,seen,device)
            met=evaluate_scores(y[va],sv,y[unseen],su)
            rows.append({'held_out_trigger':held,'seed':seed,'seen_test_AUROC':float(roc_auc_score(y[seen],ss)),
                         **{f'unseen_{k}':v for k,v in met.items()},**asdict(info)})
            print(f"{held:18s} seed={seed} unseenAUROC={met['AUROC']:.4f} val1TPR={met['val1_TPR']:.4f}")
    raw=pd.DataFrame(rows); raw.to_csv(OUT_DIR/'loto_raw.csv',index=False)
    agg=raw.groupby('held_out_trigger').agg(unseen_AUROC_mean=('unseen_AUROC','mean'),
        unseen_AUROC_std=('unseen_AUROC',lambda x:x.std(ddof=0)),seen_AUROC_mean=('seen_test_AUROC','mean'),
        val1_TPR_mean=('unseen_val1_TPR','mean'),val1_FPR_mean=('unseen_val1_FPR','mean'),best_epoch_mean=('best_epoch','mean'))
    agg.sort_values('unseen_AUROC_mean',ascending=False).to_csv(OUT_DIR/'loto_summary.csv')

if __name__=='__main__': main()
