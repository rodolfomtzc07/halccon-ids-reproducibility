# halccon-ids-reproducibility
Reproducibility code and experiments for H.A.L.C.CO.N, a hierarchical attention-based CNN for real-world multiclass intrusion detection on LITNET-2020.

## Paper
**Hierarchical Attention-Based Convolutional Neural Network Model for Intrusion Detection**

## Overview
This repository contains the code required to reproduce the main experiments reported in the final accepted version of the paper.

## Main experiments reproduced
- Main LITNET-2020 evaluation
- Stratified 5-fold cross-validation
- Classical machine learning baselines
- Ablation study
- EQLv2 hyperparameter sensitivity analysis
- Cross-dataset evaluation on UNSW-NB15

## Repository structure
- `src/`: core implementation
- `scripts/`: runnable experiment entry points
- `configs/`: experiment configurations
- `data/`: instructions for dataset placement
- `results/`: example outputs and generated artifacts
- `notebooks/archive/`: archived exploratory notebooks

## Environment
- Python 3.10
- PyTorch 2.3.0+cu118

## Reproducibility note
The official reproducibility workflow in this repository is script-based and aligned with the final accepted manuscript.

## Data
The full datasets are not included in this repository. Users must obtain LITNET-2020 and, if needed, UNSW-NB15 separately and place them in the expected local directories before running the pipeline.
