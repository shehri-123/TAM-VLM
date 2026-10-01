"""Final OpenCLIP MLP: five seeds, validation-only model and threshold selection."""
from dataclasses import asdict
import numpy as np, pandas as pd

from final_config import *
from experiment_utils import *


def main():
    device = resolve_device('cuda')
    df, X, y = load_inputs(MANIFEST, EMB_DIR/'embeddings.npy')
    tr = np.flatnonzero(df.split.eq('train').to_numpy())
    va = np.flatnonzero(df.split.eq('val').to_numpy())
    te = np.flatnonzero(df.split.eq('test').to_numpy())
    rows=[]
    for seed in SEEDS_MAIN:
        model, info = train_detector(X,y,tr,va,seed=seed,device=device,batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,patience=PATIENCE,lr=LR,weight_decay=WEIGHT_DECAY)
        sv=score_model(model,X,va,device); st=score_model(model,X,te,device)
        met=evaluate_scores(y[va],sv,y[te],st)
        rows.append({**asdict(info),**met})
        save_checkpoint(CKPT_DIR/f'clip_mlp_seed{seed}.pt',model,info,
            {'encoder':'OpenCLIP ViT-B/16','manifest_sha256':sha256_file(MANIFEST)})
        np.savez_compressed(OUT_DIR/f'main_predictions_seed{seed}.npz',
            val_indices=va,val_labels=y[va],val_scores=sv,test_indices=te,test_labels=y[te],test_scores=st)
        print(f"seed={seed} AUROC={met['AUROC']:.4f} bal={met['bal_BalAcc']:.4f} "
              f"testROC-TPR1={met['TPR_at_test_1pct_FPR_diagnostic']:.4f} "
              f"val1-TPR={met['val1_TPR']:.4f} actualFPR={met['val1_FPR']:.4f}")
    raw=pd.DataFrame(rows); raw.to_csv(OUT_DIR/'main_5seeds.csv',index=False)
    num=raw.select_dtypes(include=[np.number])
    pd.DataFrame({'mean':num.mean(),'std':num.std(ddof=0)}).to_csv(OUT_DIR/'main_5seeds_summary.csv')

if __name__=='__main__': main()
