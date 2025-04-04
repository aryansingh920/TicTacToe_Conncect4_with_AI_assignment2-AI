import argparse

from gui.tictactoe import run_tictactoe
from gui.connect4 import run_connect4


def main():
    parser = argparse.ArgumentParser(
        description="Play a board game with AI agents.")
    parser.add_argument(
        "--game",
        type=str,
        choices=["tictactoe", "connect4"],
        required=True,
        help="Choose the game to play: tictactoe or connect4"
    )
    parser.add_argument(
        "--ai_mode_1",
        type=str,
        default="none",
        choices=["none", "random", "minimax", "alphabeta", "qlearning"],
        help="AI mode for Player 1 (X)"
    )
    parser.add_argument(
        "--ai_mode_2",
        type=str,
        default="none",
        choices=["none", "random", "minimax", "alphabeta", "qlearning"],
        help="AI mode for Player 2 (O)"
    )

    args = parser.parse_args()

    if args.game == "tictactoe":
        run_tictactoe(ai_mode_1=args.ai_mode_1, ai_mode_2=args.ai_mode_2)
    elif args.game == "connect4":
        run_connect4(ai_mode_1=args.ai_mode_1, ai_mode_2=args.ai_mode_2)


if __name__ == "__main__":
    main()
