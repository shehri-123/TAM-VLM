from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


class Detector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainInfo:
    seed: int
    best_epoch: int
    best_val_auroc: float
    epochs_run: int
    n_train: int
    n_train_clean: int
    n_train_triggered: int
    n_val: int


def base_image_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path).split('__')[0])[0]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = 'cuda') -> torch.device:
    if requested.startswith('cuda') and not torch.cuda.is_available():
        print('[warning] CUDA unavailable; using CPU')
        return torch.device('cpu')
    return torch.device(requested)


def load_inputs(manifest: str | Path, embeddings: str | Path):
    df = pd.read_csv(manifest)
    X = np.load(embeddings, mmap_mode='r')
    required = {'image', 'scene', 'trigger', 'label', 'split'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'manifest missing columns: {sorted(missing)}')
    if len(df) != len(X):
        raise ValueError(f'manifest rows ({len(df)}) != embedding rows ({len(X)})')
    y = df['label'].to_numpy(dtype=np.int64)
    return df, X, y


def score_model(model, X, indices, device, batch_size: int = 4096):
    idx = np.asarray(indices)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(idx), batch_size):
            part = idx[start:start + batch_size]
            xb = torch.as_tensor(np.asarray(X[part]), dtype=torch.float32, device=device)
            out[start:start + len(part)] = torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()
    return out


def score_array(model, features: np.ndarray, device, batch_size: int = 4096):
    out = np.empty(len(features), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            xb = torch.as_tensor(np.asarray(features[start:start + batch_size]), dtype=torch.float32, device=device)
            out[start:start + len(xb)] = torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()
    return out


def best_balanced_threshold(labels, scores) -> float:
    fpr, tpr, thresholds = roc_curve(np.asarray(labels), np.asarray(scores))
    bal = 0.5 * (tpr + 1.0 - fpr)
    finite = np.isfinite(thresholds)
    candidates = np.flatnonzero(finite)
    if not len(candidates):
        raise ValueError('No finite ROC threshold')
    best = np.nanmax(bal[candidates])
    tied = candidates[np.isclose(bal[candidates], best, rtol=0, atol=1e-12)]
    return float(np.max(thresholds[tied]))


def threshold_from_validation_fpr(labels, scores, target_fpr: float = 0.01) -> float:
    neg = np.asarray(scores)[np.asarray(labels) == 0]
    if not len(neg):
        raise ValueError('Validation split contains no clean negatives')
    try:
        return float(np.quantile(neg, 1.0 - target_fpr, method='higher'))
    except TypeError:
        return float(np.quantile(neg, 1.0 - target_fpr, interpolation='higher'))


def tpr_at_test_fpr(labels, scores, target_fpr: float = 0.01) -> float:
    labels = np.asarray(labels); scores = np.asarray(scores)
    neg, pos = scores[labels == 0], scores[labels == 1]
    if not len(neg) or not len(pos):
        return float('nan')
    try:
        thr = np.quantile(neg, 1.0 - target_fpr, method='higher')
    except TypeError:
        thr = np.quantile(neg, 1.0 - target_fpr, interpolation='higher')
    return float((pos > thr).mean())


def fixed_threshold_metrics(labels, scores, threshold: float, prefix: str = ''):
    labels = np.asarray(labels); scores = np.asarray(scores)
    pred = scores > threshold
    neg, pos = labels == 0, labels == 1
    return {
        f'{prefix}F1': float(f1_score(labels, pred, zero_division=0)),
        f'{prefix}BalAcc': float(balanced_accuracy_score(labels, pred)),
        f'{prefix}FPR': float(pred[neg].mean()) if neg.any() else float('nan'),
        f'{prefix}TPR': float(pred[pos].mean()) if pos.any() else float('nan'),
        f'{prefix}Precision': float(precision_score(labels, pred, zero_division=0)),
        f'{prefix}Recall': float(recall_score(labels, pred, zero_division=0)),
    }


def evaluate_scores(val_y, val_scores, test_y, test_scores, target_fpr: float = 0.01):
    thr_bal = best_balanced_threshold(val_y, val_scores)
    thr_low = threshold_from_validation_fpr(val_y, val_scores, target_fpr)
    out = {
        'AUROC': float(roc_auc_score(test_y, test_scores)),
        'AUPRC': float(average_precision_score(test_y, test_scores)),
        'threshold_balacc_val': thr_bal,
        'threshold_val_1pct_fpr': thr_low,
        'TPR_at_test_1pct_FPR_diagnostic': tpr_at_test_fpr(test_y, test_scores, target_fpr),
    }
    out.update(fixed_threshold_metrics(test_y, test_scores, thr_bal, 'bal_'))
    out.update(fixed_threshold_metrics(test_y, test_scores, thr_low, 'val1_'))
    return out


def train_detector(X, y, train_idx, val_idx, *, seed: int, device, batch_size: int = 256,
                   max_epochs: int = 60, patience: int = 12, lr: float = 1e-3,
                   weight_decay: float = 1e-4, hidden_dim: int = 128, dropout: float = 0.2):
    seed_everything(seed)
    train_idx = np.asarray(train_idx); val_idx = np.asarray(val_idx)
    if train_idx.dtype == bool: train_idx = np.flatnonzero(train_idx)
    if val_idx.dtype == bool: val_idx = np.flatnonzero(val_idx)
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        raise ValueError('Train and validation must both contain clean and triggered samples')

    Xt = torch.as_tensor(np.asarray(X[train_idx]), dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[train_idx], dtype=torch.float32, device=device)
    n_pos = int((yt == 1).sum().item()); n_neg = int((yt == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    model = Detector(X.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, best_epoch, bad = -np.inf, None, -1, 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed * 100_000 + epoch)
        perm = torch.randperm(len(Xt), generator=generator, device=device)
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward(); optimizer.step()
        scheduler.step(); epochs_run = epoch + 1
        val_scores = score_model(model, X, val_idx, device)
        val_auc = roc_auc_score(y[val_idx], val_scores)
        if val_auc > best_auc + 1e-8:
            best_auc = float(val_auc); best_epoch = epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError('No checkpoint produced')
    model.load_state_dict(best_state); model.to(device).eval()
    info = TrainInfo(seed, best_epoch, best_auc, epochs_run, len(train_idx), n_neg, n_pos, len(val_idx))
    return model, info


def save_checkpoint(path, model, info, metadata: Optional[dict] = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'train_info': asdict(info), 'metadata': metadata or {}}, path)


def load_checkpoint(path, input_dim: int, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = Detector(input_dim)
    if isinstance(payload, dict) and 'state_dict' in payload:
        model.load_state_dict(payload['state_dict']); meta = payload
    else:
        model.load_state_dict(payload); meta = {}
    return model.to(device).eval(), meta


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
