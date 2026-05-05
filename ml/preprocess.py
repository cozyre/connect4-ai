"""
Checkpoint 3: Data Preprocessing
- Load UCI or generated dataset
- Encode labels
- Split train/test
- Save processed files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RAW_UCI = 'data/connect-4.data'       # UCI dataset (if downloaded)
RAW_GEN = 'data/generated_data.csv'   # Self-generated


def load_uci(path):
    """UCI Connect-4 dataset: 42 categorical columns + class."""
    df = pd.read_csv(path, header=None)
    # Columns 0-41: board cells (b=blank, x=player1, o=player2)
    # Column 42: outcome (win, loss, draw)
    feature_cols = list(range(42))
    label_col = 42
    mapping = {'b': 0, 'x': 1, 'o': 2}
    for c in feature_cols:
        df[c] = df[c].map(mapping)
    df[label_col] = df[label_col].str.strip()
    df.columns = [f'cell_{i}' for i in range(42)] + ['label']
    return df


def load_generated(path):
    return pd.read_csv(path)


def preprocess(df, test_size=0.2, random_state=42):
    df = df.dropna()
    df = df.drop_duplicates()

    X = df[[f'cell_{i}' for i in range(42)]].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    if os.path.exists(RAW_UCI):
        print("Loading UCI dataset...")
        df = load_uci(RAW_UCI)
    elif os.path.exists(RAW_GEN):
        print("Loading generated dataset...")
        df = load_generated(RAW_GEN)
    else:
        raise FileNotFoundError("No dataset found. Run ml/generate_data.py first.")

    X_train, X_test, y_train, y_test = preprocess(df)

    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print("Saved to data/")