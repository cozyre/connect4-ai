import math
from game.board import (
    drop_piece, get_valid_locations, get_next_open_row,
    winning_move, is_terminal_node, board_to_features,
    PLAYER, AI, EMPTY, ROWS, COLS
)

WINDOW_LENGTH = 4


def score_window(window, piece):
    opp = PLAYER if piece == AI else AI
    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score


def score_position(board, piece):
    import numpy as np
    score = 0

    # Center column preference
    center = [int(i) for i in list(board[:, COLS // 2])]
    score += center.count(piece) * 3

    # Horizontal
    for r in range(ROWS):
        row = [int(i) for i in list(board[r, :])]
        for c in range(COLS - 3):
            score += score_window(row[c:c + 4], piece)

    # Vertical
    for c in range(COLS):
        col = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            score += score_window(col[r:r + 4], piece)

    # Diagonal /
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += score_window(window, piece)

    # Diagonal \
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r - i][c + i] for i in range(4)]
            score += score_window(window, piece)

    return score


def minimax(board, depth, alpha, beta, maximizing, ml_model=None):
    import numpy as np
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if terminal:
        if winning_move(board, AI):
            return (None, 1_000_000)
        elif winning_move(board, PLAYER):
            return (None, -1_000_000)
        else:
            return (None, 0)

    if depth == 0:
        if ml_model is not None:
            features = [board_to_features(board)]
            proba = ml_model.predict_proba(features)[0]
            classes = list(ml_model.classes_)
            score = 0
            if 'win' in classes:
                score += proba[classes.index('win')] * 100
            if 'loss' in classes:
                score -= proba[classes.index('loss')] * 100
            return (None, score)
        return (None, score_position(board, AI))

    best_col = valid_locations[0]

    if maximizing:
        value = -math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp = board.copy()
            drop_piece(temp, row, col, AI)
            _, new_score = minimax(temp, depth - 1, alpha, beta, False, ml_model)
            if new_score > value:
                value, best_col = new_score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp = board.copy()
            drop_piece(temp, row, col, PLAYER)
            _, new_score = minimax(temp, depth - 1, alpha, beta, True, ml_model)
            if new_score < value:
                value, best_col = new_score, col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value


def get_best_move(board, depth=5, ml_model=None):
    col, _ = minimax(board, depth, -math.inf, math.inf, True, ml_model)
    return col