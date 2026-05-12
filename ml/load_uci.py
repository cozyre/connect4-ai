"""
Adapter: Fetch UCI Connect-4 dataset and convert to project format.
- Cell values: 'b'/'x'/'o'  →  0/1/2  (empty/player/AI)
- Labels: 'win'/'loss'/'draw' (already correct strings, just normalized)
Outputs: data/generated_data.csv (compatible with preprocess.py)

UCI dataset has 42 cells: 6 rows x 7 cols, labeled a1..a6, b1..b6, ..., g1..g6
Ordering: column-major (a1,a2,...,a6, b1,...,g6)
We remap to row-major left-to-right, top-to-bottom: row0col0..row5col6
"""

import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

OUTPUT = 'data/generated_data2.csv'

# UCI column naming: a1-a6 = col A rows 1-6, b1-b6 = col B rows 1-6, etc.
# Row-major order (top=row0): for each row r (0..5), iterate cols a..g
UCI_COLS = [f'{col}{row}' for row in range(1, 7) for col in 'abcdefg']
# That gives row-major: a1,b1,c1,d1,e1,f1,g1, a2,b2,...,g6 (row 1 = top)

CELL_MAP = {'b': 0, 'x': 1, 'o': 2}
LABEL_MAP = {'win': 'win', 'loss': 'loss', 'draw': 'draw'}


def convert():
    print("Fetching UCI Connect-4 dataset...")
    connect_4 = fetch_ucirepo(id=26)

    X = connect_4.data.features  # shape (67557, 42), columns a1..g6
    y = connect_4.data.targets   # shape (67557, 1), column 'class'

    print(f"Fetched {len(X)} rows. Columns sample: {list(X.columns[:7])}")
    print(f"Label values: {y.iloc[:, 0].unique()}")

    # Reorder X columns to row-major (top-to-bottom, left-to-right)
    X = X[UCI_COLS]

    # Rename to cell_0..cell_41
    X.columns = [f'cell_{i}' for i in range(42)]

    # Map cell values
    for col in X.columns:
        X[col] = X[col].map(CELL_MAP)

    # Attach and map labels
    label_col = y.iloc[:, 0].str.strip().str.lower().map(LABEL_MAP)
    X['label'] = label_col.values

    # Drop rows with unmapped values (shouldn't happen, but safety net)
    before = len(X)
    X = X.dropna()
    if len(X) < before:
        print(f"Warning: dropped {before - len(X)} rows with unmapped values")

    # Convert cell columns to int
    cell_cols = [f'cell_{i}' for i in range(42)]
    X[cell_cols] = X[cell_cols].astype(int)

    os.makedirs('data', exist_ok=True)
    X.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(X)} rows to {OUTPUT}")
    print(X['label'].value_counts())
    print(f"\nSample row:\n{X.iloc[0].to_dict()}")


if __name__ == '__main__':
    convert()