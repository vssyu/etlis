import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from typing import Dict, Tuple


def load_label_map(
    excel_path: str | Path,
    name_col: str = 'label_name',
) -> Dict[str, int]:
    """
    Read clause category names from the Excel file and return a label→id mapping.
    The 'other/irrelevant' class is appended automatically by callers.

    Expected Excel columns (configurable via name_col):
        label_name  — the clause category string used in annotated examples
    """
    df = pd.read_excel(excel_path)
    if name_col not in df.columns:
        raise ValueError(
            f"Column '{name_col}' not found. Available columns: {df.columns.tolist()}"
        )
    labels = sorted(df[name_col].dropna().unique().tolist())
    return {label: idx for idx, label in enumerate(labels)}


def load_examples(
    examples_path: str | Path,
    label_map: Dict[str, int],
    text_col: str = 'text',
    label_col: str = 'label',
    other_label: str = 'other',
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load annotated examples and split into train / val / test HuggingFace Datasets.

    Args:
        examples_path: Path to .xlsx / .xls / .csv file.
        label_map:     Mapping returned by load_label_map().
        text_col:      Column name containing the segment text.
        label_col:     Column name containing the clause category string.
        other_label:   Name used for segments that don't match any known category.
        val_size:      Fraction of total data for validation.
        test_size:     Fraction of total data for test.
        seed:          Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    path = Path(examples_path)
    if path.suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {df.columns.tolist()}"
            )

    # Assign 'other' class id for any label not in label_map
    extended_map = {**label_map}
    if other_label not in extended_map:
        extended_map[other_label] = len(label_map)
    other_id = extended_map[other_label]

    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col])
    df['label_id'] = df[label_col].map(extended_map).fillna(other_id).astype(int)
    df = df[[text_col, 'label_id']].rename(columns={text_col: 'text'})

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df['label_id'],
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df['label_id'],
    )

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )
