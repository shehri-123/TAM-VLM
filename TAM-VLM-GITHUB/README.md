# TAM-VLM: Detecting Physical Backdoor Triggers in Driving Vision-Language Models

This repository contains the official implementation and reproducibility resources for:

**TAM-VLM: Detecting Physical Backdoor Triggers in Driving Vision-Language Models**

TAM-VLM is a detection framework designed to identify physical backdoor triggers in driving vision-language models (VLMs). The framework evaluates trigger detection under controlled physical settings and includes robustness analysis across different trigger categories, scenes, and evaluation conditions.

---

## Repository Structure
```text
TAM-VLM-GITHUB/
│
├── core/
│   └── scripts/
│       ├── 01_main_5seeds.py
│       ├── 02_strong_baselines.py
│       └── ...
│
├── external_real_image_audit/
│   ├── scripts/
│   └── protocol/
│
├── figure_generation/
│   ├── figures/
│   └── scripts/
│
├── configs/
├── docs/
├── environment.yml
└── README.md
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tam-vlm
