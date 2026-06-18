> Work in progress. The repository is currently being cleaned and documented.

# 2026dcase_task7_experiments

DCASE 2026 Task 7: Domain-Incremental Learning (DIL) experiments using VAE-based domain reliability estimation and ConfidNet-based model selection.

## Overview

This repository contains experimental code for DCASE 2026 Task 7.

The proposed framework combines:

- MCnn14 classifier
- VAE-based domain rejection mechanism
- ConfidNet confidence prediction
- Dynamic model selection for domain-incremental learning

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
2026dcase_task7_experiments/
│
├── config.py
│
├── datasets/
│   └── datasetfactory_task7.py
│
├── models/
│   ├── task7_models.py
│   ├── mcnn14.py
│   ├── vae.py
│   └── confidnet.py
│
├── scripts/
│   ├── organize_dataset.py
│   └── evaluate.py
│
├── data/
│   ├── downloaded_corpus/
│   └── task7_data/
│
└── weights/
```

---

## Dataset Preparation

Download the official DCASE 2026 Task 7 development datasets and place them under:

```text
data/downloaded_corpus/
```

Expected structure:

```text
data/downloaded_corpus/
├── DIL-DCASE26-Dev-D2/
│   ├── d2-dev-train/
│   ├── d2-dev-test/
│   └── metadata/
│
└── DIL-DCASE26-Dev-D3/
    ├── d3-dev-train/
    ├── d3-dev-test/
    └── metadata/
```

Then run:

```bash
python scripts/organize_dataset.py
```

This creates:

```text
data/task7_data/
├── audio/
└── evaluation_setup/
    ├── development_train.txt
    └── development_test.txt
```

---

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

The evaluation script automatically:

- loads Task 1 model
- evaluates D2
- loads Task 2 model
- evaluates D2 and D3
- reports class-wise accuracy

Configuration is controlled through:

```python
config.py
```

---

## Model Weights

Pretrained model weights are expected under:

```text
weights/
```

Example:

```text
weights/
├── Takami_OU_task7_1_D2_dictionary.pth
└── Takami_OU_task7_1_D3_dictionary.pth
```

**Note:** Training scripts and pretrained weights are currently being cleaned and reorganized. They will be added in a future update.

---

## Requirements

Typical dependencies:

```bash
pip install torch torchaudio
pip install pandas numpy librosa
pip install torchlibrosa
```

---

## Status

Current:

- Dataset organization script
- Evaluation pipeline
- VAE implementation
- ConfidNet implementation
- Dynamic model selection

Planned updates:

- Training scripts
- Reproducible training pipeline
- Weight release
- Detailed experiment configurations

---

## Citation

If you use this code, please cite the corresponding DCASE 2026 Task 7 submission (to be added after publication).