# # """
# # Created on 30/03/2025

# # @author: Aryan

# # Filename: main.py

# # Relative Path: src/main.py
# # """


# # # from gui.tictactoe import run_tictactoe
# # # from gui import headless_selfplay_tictactoe
# # # if __name__ == "__main__":
# # #     # headless_selfplay_tictactoe.main(episode_count=10000)
# # #     # Example calls:
# # #     run_tictactoe(ai_mode_1="none", ai_mode_2="minimax")
# # #     #  => Human vs Human
# # #     # => Minimax (X) vs Human (O)
# # #     # run_tictactoe(ai_mode_1="qlearning", ai_mode_2="minimax")
# # #     #  3) run_tictactoe(ai_mode_1="minimax", ai_mode_2="qlearning") => Minimax (X) vs Q-Learning (O)
# # #     #  4) run_tictactoe(ai_mode_1="qlearning", ai_mode_2="qlearning") => Q-Learning X vs Q-Learning O
# # #     # run_tictactoe(ai_mode_1="qlearning", ai_mode_2="alpha-beta")
# # #     #


# from gui.connect4 import run_connect4
# # from gui import headless_selfplay_connect4
# if __name__ == "__main__":
#     # headless_selfplay_connect4.main(episode_count=10000)
#     # Examples:
#     run_connect4(ai_mode_1="minimax", ai_mode_2="none")
#     #   run_connect4(ai_mode_1="minimax", ai_mode_2="none") => Minimax (Red) vs Human (Yellow)
#     #   run_connect4(ai_mode_1="minimax", ai_mode_2="alpha-beta") => Minimax vs Alpha-Beta
#     # run_connect4(ai_mode_1="none", ai_mode_2="qlearning")
