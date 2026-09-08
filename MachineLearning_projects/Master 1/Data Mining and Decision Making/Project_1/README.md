# Snakes & Ladders — Markov Decision Process (Value Iteration)

**Course:** LINFO2275 – Data Mining and Decision Making, UCLouvain (Master 1)

## 🎯 Objective

Find the dice-choice policy that **minimizes the expected number of turns** to finish a Snakes & Ladders variant board, where the board includes traps (restart, move back 3, prison, bonus) and a fast/slow line choice, and dice size (1, 2 or 3) trades off speed against trap exposure.

## 🧩 Approach

- Modeled the game as a **Markov Decision Process**: 14 states, actions = dice choice (1/2/3)
- Solved with **Value Iteration** (Bellman optimality updates) until convergence (`epsilon = 1e-6`)
- Handled the board's special mechanics:
  - circular vs. non-circular layout
  - branching to the fast or slow line at state 2
  - trap probabilities that depend on the dice choice (rolling a 1 gives trap immunity)

## 🛠️ Tech stack

- Python
- NumPy

## 📂 Repository contents

| File | Description |
|---|---|
| `LINFO2275_Code_Final.py` | Full implementation — `BoardGame` class, Bellman/value-iteration solver, `markovDecision(layout, circle)` entry point |
| `LINFO2275_Project_1.pdf` | Final report: methodology, validation against baselines, results |
| `Projet_Data_Mining_LINFO2275_2024_2025.pdf` | Original assignment statement |

## ▶️ How to run

```python
from LINFO2275_Code_Final import markovDecision

Expec, Dice = markovDecision(layout, circle=True)
```

`layout` is a list encoding each cell's trap type (`0`=normal, `1`=restart, `2`=move back 3, `3`=bonus, `4`=prison); `circle` toggles whether the board wraps around after the last cell.
