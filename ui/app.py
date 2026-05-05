"""
Checkpoint 6: Streamlit UI
Run: streamlit run ui/app.py
"""

import streamlit as st
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.board import (
    create_board, drop_piece, get_next_open_row,
    is_valid_location, winning_move, get_valid_locations,
    board_to_features, PLAYER, AI, ROWS, COLS
)
from ai.minimax import get_best_move

try:
    from ml.train_model import load_model
    ml_model = load_model()
except Exception:
    ml_model = None

# ── Session state ──────────────────────────────────────────────────────────────
if 'board' not in st.session_state:
    st.session_state.board = create_board()
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None

st.title("🔴 Connect 4 AI")
st.caption("You are 🔴 (Player 1). AI is 🟡.")

# ── Board display ──────────────────────────────────────────────────────────────
ICONS = {0: "⚫", PLAYER: "🔴", AI: "🟡"}

board = st.session_state.board
flipped = np.flip(board, 0)

for row in flipped:
    st.write("  ".join(ICONS[cell] for cell in row))

# ── Confidence (optional) ──────────────────────────────────────────────────────
if ml_model is not None:
    features = [board_to_features(board)]
    proba = ml_model.predict_proba(features)[0]
    classes = list(ml_model.classes_)
    st.markdown("**ML Prediction confidence:**")
    for cls, p in zip(classes, proba):
        st.progress(float(p), text=f"{cls}: {p:.1%}")

# ── Player move ────────────────────────────────────────────────────────────────
if not st.session_state.game_over:
    st.write("---")
    col_choice = st.selectbox("Choose a column (0–6):", list(range(COLS)))

    if st.button("Drop piece"):
        if is_valid_location(board, col_choice):
            row = get_next_open_row(board, col_choice)
            drop_piece(board, row, col_choice, PLAYER)

            if winning_move(board, PLAYER):
                st.session_state.game_over = True
                st.session_state.winner = "Player"
            elif len(get_valid_locations(board)) == 0:
                st.session_state.game_over = True
                st.session_state.winner = "Draw"
            else:
                # AI move
                depth = 5
                ai_col = get_best_move(board, depth=depth, ml_model=ml_model)
                if ai_col is not None:
                    ai_row = get_next_open_row(board, ai_col)
                    drop_piece(board, ai_row, ai_col, AI)
                    if winning_move(board, AI):
                        st.session_state.game_over = True
                        st.session_state.winner = "AI"
            st.rerun()
        else:
            st.warning("Column full! Choose another.")

# ── Game over ──────────────────────────────────────────────────────────────────
if st.session_state.game_over:
    winner = st.session_state.winner
    if winner == "Player":
        st.success("🎉 You win!")
    elif winner == "AI":
        st.error("🤖 AI wins!")
    else:
        st.info("It's a draw!")

    if st.button("Play again"):
        st.session_state.board = create_board()
        st.session_state.game_over = False
        st.session_state.winner = None
        st.rerun()