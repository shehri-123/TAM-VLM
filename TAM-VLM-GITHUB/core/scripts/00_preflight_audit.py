"""Run before any GPU experiment. Stops on data, split, or alignment failure."""
from pathlib import Path
import json
import numpy as np

from final_config import ALL_TRIGGERS, EMB_DIR, MANIFEST, OUT_DIR
from experiment_utils import base_image_id, load_inputs, save_json, sha256_file


def main():
    emb = EMB_DIR / 'embeddings.npy'
    df, X, y = load_inputs(MANIFEST, emb)
    fail, warn = [], []
    def req(cond, msg):
        if not cond: fail.append(msg)

    req(set(df.split.unique()) == {'train','val','test'}, 'unexpected split labels')
    req(set(df.label.unique()) == {0,1}, 'labels are not {0,1}')
    req(df.loc[df.trigger.eq('none'), 'label'].eq(0).all(), 'clean label mismatch')
    req(df.loc[~df.trigger.eq('none'), 'label'].eq(1).all(), 'trigger label mismatch')
    req(set(df.loc[~df.trigger.eq('none'), 'trigger'].unique()) == set(ALL_TRIGGERS), 'trigger set mismatch')

    scenes = {s:set(df.loc[df.split.eq(s),'scene']) for s in ('train','val','test')}
    req(not scenes['train'] & scenes['val'], 'train/val scene leakage')
    req(not scenes['train'] & scenes['test'], 'train/test scene leakage')
    req(not scenes['val'] & scenes['test'], 'val/test scene leakage')

    temp = df.assign(_base=df.image.map(base_image_id))
    req(int((temp.groupby('_base').split.nunique() > 1).sum()) == 0, 'source-image variants span splits')
    variant_counts = temp.groupby('_base').trigger.nunique()
    req(bool((variant_counts == 13).all()), f'not every source has 13 variants: {variant_counts.min()}..{variant_counts.max()}')

    counts = df.trigger.value_counts()
    clean_n = int(counts.get('none', 0))
    for t in ALL_TRIGGERS:
        req(int(counts.get(t, 0)) == clean_n, f'{t}: count != clean count')

    req(np.isfinite(X).all(), 'NaN/Inf embeddings')
    norms = np.linalg.norm(X, axis=1)
    req(float(norms.min()) > 1e-8, 'zero-norm embedding')
    missing = int((~df.image.map(Path).map(Path.exists)).sum())
    if missing: warn.append(f'{missing} image paths missing')

    meta = {
        'manifest': str(MANIFEST), 'manifest_sha256': sha256_file(MANIFEST),
        'embedding': str(emb), 'embedding_shape': list(X.shape), 'embedding_dtype': str(X.dtype),
        'rows': int(len(df)), 'source_images': clean_n, 'scenes_authoritative': int(df.scene.nunique()),
        'split_rows': {str(k):int(v) for k,v in df.split.value_counts().items()},
        'split_scenes': {k:len(v) for k,v in scenes.items()},
        'per_trigger_by_split': {
            f'{sp}|{tr}': int(n) for (sp,tr),n in df.groupby(['split','trigger']).size().items()
        },
        'warnings': warn, 'failures': fail,
    }
    save_json(OUT_DIR/'preflight_metadata.json', meta)
    print(json.dumps(meta, indent=2))
    if fail: raise SystemExit('PREFLIGHT FAILED: ' + ' | '.join(fail))
    print('\nPREFLIGHT PASS')

if __name__ == '__main__': main()
