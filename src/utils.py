"""
utils.py — Shared utilities for the ACA Butterfly Generative Augmentation project.

Split strategy
--------------
The Kaggle competition already provides a separate unlabelled `test/` folder
for the final submission. Therefore the labelled `train.csv` data is split
into only TWO parts:

  Train : 80 %  — update model weights
  Val   : 20 %  — monitor training, compare models

Calling get_splits(df, seed=GLOBAL_SEED) in every notebook guarantees
identical subsets without saving any CSV files to the repository.
"""

from sklearn.model_selection import train_test_split

# Project-wide seed — never change this so results stay reproducible
GLOBAL_SEED = 42


def get_splits(df, seed=GLOBAL_SEED):
    """
    Split a labelled DataFrame into Train (80%) and Validation (20%) subsets.

    Splits are **stratified** on the 'label' column so every class appears
    in both subsets in proportion to its frequency.

    Parameters
    ----------
    df   : pd.DataFrame with at least columns ['filename', 'label']
    seed : int, random state (default: GLOBAL_SEED = 42)

    Returns
    -------
    train_df, val_df : pd.DataFrames (index reset)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_class_mapping(df):
    """
    Build a sorted class → index mapping from the 'label' column.

    Parameters
    ----------
    df : pd.DataFrame with a 'label' column

    Returns
    -------
    class_to_idx : dict  {class_name: int}
    idx_to_class : dict  {int: class_name}
    classes      : list  sorted list of class names
    """
    classes = sorted(df["label"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    return class_to_idx, idx_to_class, classes

