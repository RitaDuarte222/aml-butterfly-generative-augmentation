"""
utils.py — Shared utilities for the ACA Butterfly Generative Augmentation project.

The canonical split function lives here so that every notebook uses
exactly the same splits by calling get_splits() with the same seed.
No CSV files need to be saved or committed to the repository.
"""

from sklearn.model_selection import train_test_split

# Project-wide seed — never change this so results stay reproducible
GLOBAL_SEED = 42


def get_splits(df, seed=GLOBAL_SEED):
    """
    Split a labelled DataFrame into Train / Validation / Test subsets.

    Strategy (80 / 20 / 20 style):
      1. Hold out 20 % of the full dataset as the **test** set.
      2. From the remaining 80 %, hold out a further 20 % as the **validation** set.

    This yields approximately:
      - Train      : 64 % of the full dataset
      - Validation : 16 % of the full dataset
      - Test        : 20 % of the full dataset

    Splits are **stratified** on the 'label' column so every class appears
    in all three subsets in proportion to its frequency.

    Parameters
    ----------
    df   : pd.DataFrame with at least columns ['filename', 'label']
    seed : int, random state (default: GLOBAL_SEED = 42)

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrames (index reset)
    """
    # Step 1 — hold out test set (20 % of total)
    trainval_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=seed,
    )

    # Step 2 — split remaining 80 % into train (80 %) and val (20 %)
    # 20 % of 80 % = 16 % of total → train ≈ 64 %, val ≈ 16 %
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.20,
        stratify=trainval_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


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
