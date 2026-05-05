# Connect 4 AI — Minimax + Decision Tree
> Academic project: hybrid AI using Minimax with Alpha-Beta Pruning and a Decision Tree classifier for board state evaluation.

---

## Project Structure

```
connect4/
├── game/           # Board logic (rules, win detection, state encoding)
├── ai/             # Minimax + Alpha-Beta Pruning
├── ml/             # Data generation, preprocessing, model training
├── data/           # Datasets (raw + processed)
├── models/         # Saved .pkl model
├── results/        # Metrics, confusion matrix output
└── ui/             # Streamlit interface
```

---

## Setup

```bash
git clone https://github.com/cozyre/connect4-ai.git
cd connect4-ai
pip install -r requirements.txt
```

---

## Checkpoints

### ✅ Checkpoint 3 — Data Preprocessing

**Option A: Use UCI Dataset**
1. Download from https://archive.ics.uci.edu/dataset/25/connect+4
2. Place `connect-4.data` in `data/`
3. Run:
```bash
python ml/preprocess.py
```

**Option B: Generate your own data**
```bash
python ml/generate_data.py   # generates data/generated_data.csv
python ml/preprocess.py
```

---

### ✅ Checkpoint 4 — Model Training

```bash
python ml/train_model.py
```

Outputs:
- `models/decision_tree.pkl`
- `results/metrics.json` (accuracy, classification report, confusion matrix)

---

### ✅ Checkpoint 5 — Evaluation

Check `results/metrics.json` for:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix (win / loss / draw)

---

### ✅ Checkpoint 6 — UI

```bash
streamlit run ui/app.py
```

Features:
- Play against AI
- Board rendered in browser
- ML confidence display (if model is trained)

---

## Architecture

```
User Input
    ↓
Game State (6×7 board)
    ↓
Minimax + Alpha-Beta Pruning (depth=5)
    ↓ (at leaf nodes)
Decision Tree Evaluation (win/loss/draw probability)
    ↓
Best Move Output
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| NumPy / Pandas | Board representation, data handling |
| Scikit-learn | Decision Tree training & evaluation |
| Streamlit | Web UI |

---

## Dataset

- **UCI Connect-4**: https://archive.ics.uci.edu/dataset/25/connect+4  
- **Self-generated**: `ml/generate_data.py` (random AI vs AI simulations)

---

## Notes

- Minimax depth is set to 5 by default (tunable in `ui/app.py`)
- ML model is optional — AI falls back to heuristic scoring if no model is found
- Keep `data/` and `models/` in `.gitignore` if files are large