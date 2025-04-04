====================================================
           AI Board Games: Tic Tac Toe & Connect 4
                  Developed by Aryan Singh
====================================================

This project implements intelligent agents (Minimax, Alpha-Beta Pruning, and Q-Learning)
for playing Tic Tac Toe and Connect 4 with a graphical interface using Pygame.

Both games support:
- Human vs Human
- Human vs AI
- AI vs AI
- Headless Self-Play for Q-Learning training

----------------------------------------------------
📦 Requirements:
----------------------------------------------------
- Python 3.7+
- pygame
- argparse
- numpy (optional, for performance)
- pickle (standard lib)

To install dependencies (if needed):
> pip install pygame

----------------------------------------------------
🚀 How to Run the Games (GUI)
----------------------------------------------------
Run the main launcher:

> python main.py --game [tictactoe|connect4] --ai_mode_1 [ai_type] --ai_mode_2 [ai_type]

Examples:
> python main.py --game tictactoe --ai_mode_1 minimax --ai_mode_2 none  
  → Minimax AI (X) vs Human (O)

> python main.py --game connect4 --ai_mode_1 qlearning --ai_mode_2 qlearning  
  → Q-Learning vs Q-Learning in Connect 4

AI Types:
- none       → Human player
- random     → Random moves
- minimax    → Classic Minimax algorithm
- alphabeta  → Minimax with Alpha-Beta pruning
- qlearning  → Trained reinforcement learning agent

----------------------------------------------------
🧠 How to Train Q-Learning Agents (Headless Mode)
----------------------------------------------------
You can run self-play simulations to train Q-Learning agents using:

> python src/gui/headless_selfplay_tictactoe.py  
> python src/gui/headless_selfplay_connect4.py

These scripts will:
- Run thousands of self-play episodes
- Save learned Q-tables in `data/` folder
- Print win stats after training

To customize training episodes, edit the `main()` function in those files.

----------------------------------------------------
📁 Project Structure (Relevant Files)
----------------------------------------------------
- `main.py` → Entry point to play games via CLI
- `src/gui/tictactoe.py` → Tic Tac Toe GUI logic
- `src/gui/connect4.py` → Connect 4 GUI logic
- `src/algorithm/` → AI agents (Minimax, Alpha-Beta, Q-Learning)
- `src/state/` → Cached state managers
- `data/` → Q-tables and precomputed state dictionaries

----------------------------------------------------
📌 Notes:
----------------------------------------------------
- For best Q-Learning performance, train agents using headless scripts before playing.
- Make sure to save your Q-table in the correct file path if you modify the training.

----------------------------------------------------
🧑‍💻 Author:
----------------------------------------------------
Aryan Singh  
MSc in Data Science, Trinity College Dublin
