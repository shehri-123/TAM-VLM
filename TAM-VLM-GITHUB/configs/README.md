# Configuration Files

This directory contains example configuration settings for running TAM-VLM experiments.

Paths to datasets, embeddings, and model weights should be configured locally by each user.

The repository avoids storing absolute paths or large external resources.

Example environment variables:

```bash
export TAMVLM_RESULTS_ROOT=/path/to/results
export TAMVLM_EXTERNAL_ROOT=/path/to/external_audit
