# Data Instructions

This repository does not include the original datasets used in the paper.

## Required datasets
To reproduce the experiments, users must obtain the following datasets separately:

- **LITNET-2020**
- **UNSW-NB15** (only required for the cross-dataset evaluation)

## Data access
Please obtain each dataset from its official source or repository and place the files in local directories before running preprocessing or training scripts.

## Expected usage
The main experiments reported in the paper were conducted on **LITNET-2020**, while **UNSW-NB15** was used only for the cross-dataset evaluation.

## Included split metadata
To facilitate reproducibility, this repository includes predefined dataset split files for both **LITNET-2020** and **UNSW-NB15** in **JSON format**.

These split files define the samples used for:

- **train**
- **validation**
- **test**

## Attack distribution summaries
The repository also includes summary files describing the attack-class distributions for the corresponding:

- **train split**
- **validation split**
- **test split**

for both datasets. These files are intended to document the class balance used in the experiments and to make the evaluation protocol easier to reproduce and verify.

## Recommended local structure
A suggested local organization is:

```text
data/
├── litnet/
│   └── <LITNET-2020 files>
├── unsw_nb15/
│   └── <UNSW-NB15 files>
└── splits/
    ├── litnet/
    │   ├── train_split.json
    │   ├── val_split.json
    │   ├── test_split.json
    │   └── class_distributions/
    └── unsw_nb15/
        ├── train_split.json
        ├── val_split.json
        ├── test_split.json
        └── class_distributions/
