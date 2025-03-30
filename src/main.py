"""
Created on 30/03/2025

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""


from gui.tictactoe import run_tictactoe
from gui.headless_selfplay_tictactoe import main

if __name__ == "__main__":
    # main(episode_count=10000)
    # Example calls:
    #  1) run_tictactoe(ai_mode_1="none", ai_mode_2="none") => Human vs Human
    # => Minimax (X) vs Human (O)
    run_tictactoe(ai_mode_1="qlearning", ai_mode_2="minimax")
    #  3) run_tictactoe(ai_mode_1="minimax", ai_mode_2="qlearning") => Minimax (X) vs Q-Learning (O)
    #  4) run_tictactoe(ai_mode_1="qlearning", ai_mode_2="qlearning") => Q-Learning X vs Q-Learning O
    # run_tictactoe(ai_mode_1="qlearning", ai_mode_2="alpha-beta")
#
