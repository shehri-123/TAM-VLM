#!/bin/bash

# TAM-VLM environment configuration
# Modify these paths according to your local setup.

export TAMVLM_DATA_ROOT=${TAMVLM_DATA_ROOT:-"./data"}

export TAMVLM_EMB_DIR=${TAMVLM_EMB_DIR:-"./data/embeddings"}

export TAMVLM_MANIFEST=${TAMVLM_MANIFEST:-"./data/manifest.csv"}

export TAMVLM_OUT_DIR=${TAMVLM_OUT_DIR:-"./results"}

export TAMVLM_CLIP_WEIGHTS=${TAMVLM_CLIP_WEIGHTS:-"./checkpoints/open_clip_model.safetensors"}

echo "TAM-VLM environment variables configured."
echo "DATA_ROOT: $TAMVLM_DATA_ROOT"
echo "EMB_DIR: $TAMVLM_EMB_DIR"
echo "MANIFEST: $TAMVLM_MANIFEST"
echo "OUT_DIR: $TAMVLM_OUT_DIR"
echo "CLIP_WEIGHTS: $TAMVLM_CLIP_WEIGHTS"
