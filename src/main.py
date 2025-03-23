"""
Created on 23/03/2025

@author: Aryan

Filename: main.py

Relative Path: main.py
"""

"""
Filename: main.py
Path: main.py

Example usage:
    python main.py tictactoe minimax
    python main.py tictactoe alpha-beta
    python main.py tictactoe qlearning
    python main.py connect4 minimax
    ...
"""

from ai.connect4 import run_connect4
from ai.tictactoe import run_tictactoe
import sys
import argparse

# Import the GUIs (which we assume are updated to accept an AI "hook").
# Make sure these imports match your directory structure.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "game", choices=["tictactoe", "connect4"], help="Which game to play.")
    parser.add_argument("ai", choices=["minimax", "alpha-beta", "qlearning", "none"],
                        help="Which AI to use. If 'none', it's 2-player human.")
    return parser.parse_args()


def main():
    args = parse_args()
    game = args.game
    ai = args.ai

    if game == "tictactoe":
        run_tictactoe(ai)
    else:
        run_connect4(ai)


if __name__ == "__main__":
    main()
