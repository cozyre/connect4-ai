"""
Checkpoint 3: Data Preprocessing
- Load from: Kaggle CSV, UCI package, or self-generated
- Encode labels
- Split train/test (80/20)
- Save processed files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Dataset paths
KAGGLE_CSV = 'data/c4_game_database.csv'
RAW_GEN = 'data/generated_data.csv'

# Split ratio: 80% train, 20% test
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2


def load_kaggle(path):
    """
    Load Kaggle dataset: 42 columns (pos_01 to pos_42) + winner
    Values: 1 (player), -1 (opponent), 0 (empty)
    Winner: 1/-1/0
    """
    df = pd.read_csv(path)
    
    # Get feature columns (first 42)
    feature_cols = [col for col in df.columns if col.startswith('pos_')]
    label_col = 'winner'
    
    # Rename to standard format
    df.columns = [f'cell_{i}' for i in range(42)] + ['label']
    
    # Convert cell values: -1 → 2 (AI), keep 1 (player), 0 (empty)
    for c in range(42):
        df[f'cell_{c}'] = df[f'cell_{c}'].replace(-1, 2)
    
    # Convert labels: 1 → 'win', -1 → 'loss', 0 → 'draw'
    df['label'] = df['label'].map({1: 'win', -1: 'loss', 0: 'draw'})
    
    return df


def load_uci():
    from ucimlrepo import fetch_ucirepo
    # Fetch dataset (id=26 is Connect-4)
    connect_4 = fetch_ucirepo(id=26)
    
    # Get features and target
    X = connect_4.data.features
    y = connect_4.data.targets
    
    # Combine into one dataframe
    df = X.copy()
    df['label'] = y
    
    # Rename columns to standard format
    df.columns = [f'cell_{i}' for i in range(42)] + ['label']
    
    # The UCI dataset already has string labels (b, x, o) which need mapping
    # Actually, check your dataset format—adjust if needed
    # For now, assume labels are already ('win', 'loss', 'draw')
    
    return df


def load_generated(path):
    """Load self-generated dataset from ml/generate_data.py"""
    return pd.read_csv(path)


def preprocess(df, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=42):
    """
    Clean, split, and prepare data.
    
    Split ratio: 80% train, 20% test (stratified)
    """
    # Remove missing values and duplicates
    df = df.dropna()
    df = df.drop_duplicates()
    
    print(f"Cleaned dataset: {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Extract features and labels
    X = df[[f'cell_{i}' for i in range(42)]].values
    y = df['label'].values
    
    # Split: 80/20 with stratification (keep label proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Train/Test split:")
    print(f"  Train: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)\n")
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Choose which dataset to load
    print("=" * 60)
    print("DATASET LOADING OPTIONS")
    print("=" * 60)
    
    df = None
    
    # Priority 1: Kaggle CSV (if file exists)
    if os.path.exists(KAGGLE_CSV):
        print(f"\n✓ Found Kaggle dataset: {KAGGLE_CSV}")
        print("  Loading...")
        df = load_kaggle(KAGGLE_CSV)
    
    # Priority 2: UCI package
    elif False:  # Set to True if you want to use UCI
        print("\n✓ Fetching UCI Connect-4 dataset...")
        df = load_uci()
    
    # Priority 3: Self-generated
    elif os.path.exists(RAW_GEN):
        print(f"\n✓ Found generated dataset: {RAW_GEN}")
        df = load_generated(RAW_GEN)
    
    else:
        print("\n✗ No dataset found!")
        print(f"  Options:")
        print(f"    - Place Kaggle CSV at: {KAGGLE_CSV}")
        print(f"    - Run: python ml/generate_data.py")
        print(f"    - Uncomment UCI loading in preprocess.py")
        raise FileNotFoundError("No dataset available")
    
    # Preprocess
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    X_train, X_test, y_train, y_test = preprocess(df)
    
    # Save numpy arrays
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    print("=" * 60)
    print("SAVED TO data/")
    print("=" * 60)
    print("  ✓ X_train.npy")
    print("  ✓ X_test.npy")
    print("  ✓ y_train.npy")
    print("  ✓ y_test.npy")
    print("\nNext: python ml/train_model.py")