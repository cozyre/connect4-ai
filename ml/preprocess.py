"""
Checkpoint 3: Data Preprocessing
- Load UCI, generated, or Kaggle dataset
- Encode labels
- Split train/test
- Save processed files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Data source paths
UCI = 'data/generated_data2.csv'       # UCI dataset
KAGGLE = 'data/generated_data1.csv'   # Kaggle (already processed CSV)


def load_dataset(path):
    """Load standardized Connect-4 dataset (42 cell columns + label)."""
    df = pd.read_csv(path, dtype={'label': 'object'})
    
    # Ensure label column exists and is normalized
    if 'label' not in df.columns:
        if 'target' in df.columns:
            df.rename(columns={'target': 'label'}, inplace=True)
        else:
            raise ValueError("Dataset must have 'label' or 'target' column")
    
    # Normalize label values
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    return df


def detect_and_load_dataset():
    """Auto-detect and load datasets, combining both if available."""
    dfs = []
    
    # Load Kaggle dataset if exists
    if os.path.exists(KAGGLE):
        print(f"✓ Found dataset: {KAGGLE}")
        dfs.append(load_dataset(KAGGLE))
    
    # Load UCI dataset if exists
    if os.path.exists(UCI):
        print(f"✓ Found dataset: {UCI}")
        dfs.append(load_dataset(UCI))
    
    if not dfs:
        raise FileNotFoundError(
            f"No dataset found. Expected one of:\n"
            f"  - {UCI}\n"
            f"  - {KAGGLE}\n"
            f"Run ml/generate_data.py first or provide a dataset CSV."
        )
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✓ Combined {len(dfs)} dataset(s): {len(combined)} total rows")
    return combined


def preprocess(df, test_size=0.2, random_state=42): # 80/20 data split
    """Clean, encode, and split dataset."""
    df = df.dropna()
    df = df.drop_duplicates()

    # Ensure label values are normalized
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower().str.strip()

    X = df[[f'cell_{i}' for i in range(42)]].values
    y = np.array(df['label'].tolist(), dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Auto-detect and load dataset
    df = detect_and_load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess(df)

    # Save
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Saved to data/ (X_train.npy, X_test.npy, y_train.npy, y_test.npy)")