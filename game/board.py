import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 1
AI = 2


def create_board():
    return np.zeros((ROWS, COLS), dtype=int)


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROWS - 1][col] == EMPTY


def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None


def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]


def winning_move(board, piece):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(4)):
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i][c] == piece for i in range(4)):
                return True
    # Diagonal /
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    # Diagonal \
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True
    return False


def is_terminal_node(board):
    return (
        winning_move(board, PLAYER)
        or winning_move(board, AI)
        or len(get_valid_locations(board)) == 0
    )


def board_to_features(board):
    """Flatten board to 1D feature vector for ML model."""
    return board.flatten().tolist()


def print_board(board):
    print(np.flip(board, 0))