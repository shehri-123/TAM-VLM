"""Positive-supervision and total-training-size ablations using the unified trainer."""
import numpy as np,pandas as pd
from final_config import *
from experiment_utils import *
SEEDS=(0,1,2); FRACTIONS=(.01,.02,.05,.10,.25,.50,1.0)

def run(X,y,tr,va,te,seed,device):
    m,info=train_detector(X,y,tr,va,seed=seed,device=device,batch_size=BATCH_SIZE,max_epochs=MAX_EPOCHS,patience=PATIENCE,lr=LR,weight_decay=WEIGHT_DECAY)
    sv=score_model(m,X,va,device); st=score_model(m,X,te,device); return {**evaluate_scores(y[va],sv,y[te],st),'best_epoch':info.best_epoch}

def main():
    device=resolve_device('cuda'); df,X,y=load_inputs(MANIFEST,EMB_DIR/'embeddings.npy')
    train=df.split.eq('train').to_numpy(); va=np.flatnonzero(df.split.eq('val')); te=np.flatnonzero(df.split.eq('test'))
    pos=np.flatnonzero(train&(y==1)); neg=np.flatnonzero(train&(y==0)); out=[]
    for f in FRACTIONS:
        npos=max(50,int(round(len(pos)*f)))
        for seed in SEEDS:
            rng=np.random.RandomState(100000+seed*1000+int(f*10000)); pp=rng.choice(pos,npos,replace=False)
            out.append({'experiment':'positive_supervision','requested_fraction':f,'actual_positive_fraction':npos/len(pos),'n_pos':npos,'n_neg':len(neg),'seed':seed,**run(X,y,np.r_[neg,pp],va,te,seed,device)})
    for f in FRACTIONS:
        npos=max(50,int(round(len(pos)*f))); nneg=max(50,int(round(len(neg)*f)))
        for seed in SEEDS:
            rng=np.random.RandomState(200000+seed*1000+int(f*10000)); pp=rng.choice(pos,npos,False); nn=rng.choice(neg,nneg,False)
            out.append({'experiment':'training_size','requested_fraction':f,'actual_positive_fraction':npos/len(pos),'actual_clean_fraction':nneg/len(neg),'n_pos':npos,'n_neg':nneg,'seed':seed,**run(X,y,np.r_[nn,pp],va,te,seed,device)})
    pd.DataFrame(out).to_csv(OUT_DIR/'ablations_raw.csv',index=False)

if __name__=='__main__': main()
