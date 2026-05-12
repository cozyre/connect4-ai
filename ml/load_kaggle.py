"""
Adapter: Convert Kaggle c4_game_database.csv to project format.
- Values: 1/-1/0  →  1/2/0  (player/AI/empty)
- Winner: 1/-1/0  →  win/loss/draw
Outputs: data/generated_data.csv (compatible with preprocess.py)
"""

import pandas as pd
import os

INPUT = 'data/c4_game_database.csv'
OUTPUT = 'data/generated_data.csv'


def convert():
    df = pd.read_csv(INPUT)

    # Rename columns to match project format
    df.columns = [f'cell_{i}' for i in range(42)] + ['label']

    # Convert cell values: -1 → 2 (AI), 0 stays 0, 1 stays 1
    cell_cols = [f'cell_{i}' for i in range(42)]
    df[cell_cols] = df[cell_cols].replace(-1, 2)

    # Convert labels
    df['label'] = df['label'].map({1: 'win', -1: 'loss', 0: 'draw'})
    df = df.dropna(subset=['label'])

    os.makedirs('data', exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(df['label'].value_counts())


if __name__ == '__main__':
    convert()