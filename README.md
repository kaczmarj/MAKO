# MAKO

This is the repository for MAKO 🦈 (Mammary Analysis for Knowledge of Outcomes), a framework for inferring transcriptomic recurrence risk scores and intrinsic subtypes in breast cancer.

This repository is a work in progress. Please check back occasionally for updates.

# Install

Install dependencies first.

Installation instructions for PyTorch can be different depending on what computer you have, so please use these [installation instructions](https://pytorch.org/get-started/locally/).

Then install the rest of the dependencies:

```shell
pip install scikit-learn numpy scipy matplotlib seaborn lifelines click torchmetrics
```

Installation should take less than two minutes.

# Train an ROR-P classification model

An ROR-P classification model can be trained using the following command. Please keep reading for an explanation of the arguments.

```shell
python train_classification.py \
    --features-dir path/to/features/directory \
    --csv data.csv \
    --label-col label_column \
    --num-classes 2 \
    --split-json splits.json \
    --fold 0 \
    --output-dir outputs/split-0 \
    --model-name AttentionMILModel \
    --embedding-size 512 \
    --num-epochs 20 \
    --seed 0 \
    -L 512 \
    -D 384 \
    --lr 1e-4
```

Explanation of arguments:

- `--features-dir` : this contains the `.pt` (PyTorch) files for the whole slide image patch embeddings.
- `--csv` : the CSV spreadsheet containing the case IDs, slide IDs, and ground truth labels.
- `--label-col` : the column in `--csv` that contains the labels. For classification, this should be a single number in [0, n_classes).
- `--split-json` : a JSON file with the Kfold splits. The splits should be by case (not by slides). Here is an example:
    ```json
    {
    "patient_ids": [
        {
            "train": ["12340", "12341", "12342"],
            "val": ["22340", "22341", "22342"],
            "test": ["32340", "32341", "32342"]
        },
        {
            "train": ["22340", "22341", "22342"],
            "val": ["12340", "12341", "12342"],
            "test": ["32340", "32341", "32342"]
        },
        ]
    }
    ```
- `--fold` : the fold to train. This will choose splits from the `--fold` index of `patient_ids` in `--split-json`.
