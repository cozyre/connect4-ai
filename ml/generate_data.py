"""
Generate self-play dataset using random vs random AI games.
Outputs: data/generated_data.csv
"""

import numpy as np
import pandas as pd
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.board import (
    create_board, drop_piece, get_valid_locations,
    get_next_open_row, winning_move, board_to_features,
    PLAYER, AI
)

COLS = 7


def simulate_game():
    board = create_board()
    turn = PLAYER
    records = []

    while True:
        valid = get_valid_locations(board)
        if not valid:
            # Draw — record all states
            for features, _ in records:
                records[records.index((features, _))] = (features, 'draw')
            break

        col = random.choice(valid)
        row = get_next_open_row(board, col)
        drop_piece(board, row, col, turn)

        features = board_to_features(board)

        if winning_move(board, turn):
            label = 'win' if turn == AI else 'loss'
            records.append((features, label))
            # Back-label all previous states
            final_label = label
            labeled = []
            for f, _ in records[:-1]:
                labeled.append((f, final_label))
            labeled.append((features, label))
            return labeled

        records.append((features, None))
        turn = AI if turn == PLAYER else PLAYER

    return [(f, 'draw') for f, _ in records]


def generate_dataset(n_games=5000, output_path='data/generated_data.csv'):
    all_records = []
    for i in range(n_games):
        if i % 500 == 0:
            print(f"  Simulating game {i}/{n_games}...")
        records = simulate_game()
        all_records.extend(records)

    cols = [f'cell_{i}' for i in range(42)] + ['label']
    df = pd.DataFrame([f + [l] for f, l in all_records], columns=cols)
    df.drop_duplicates(inplace=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path} ({len(df)} rows)")
    return df


if __name__ == '__main__':
    generate_dataset()