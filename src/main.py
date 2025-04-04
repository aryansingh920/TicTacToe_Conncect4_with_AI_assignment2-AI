# # ---------------------------------------------------------------------
# # If you run this file directly:
# # ---------------------------------------------------------------------
# from gui.tictactoe import run_tictactoe


# if __name__ == "__main__":
#     # Example calls:
#     #  1) run_tictactoe(ai_mode_1="none", ai_mode_2="none")
#     # => Human vs Human
#     #  2) run_tictactoe(ai_mode_1="minimax", ai_mode_2="none")
#     # => Minimax (X) vs Human (O)
#     #  3) run_tictactoe(ai_mode_1="minimax", ai_mode_2="qlearning")
#     # => Q-:Learning (X) vs Human (O)
#     run_tictactoe(ai_mode_1="qlearning", ai_mode_2="none")
#     # => Minimax (X) vs Q-Learning (O)
#     #  4) run_tictactoe(ai_mode_1="qlearning", ai_mode_2="qlearning")
#     # => Q-Learning X vs Q-Learning O
#     # run_tictactoe(ai_mode_1="qlearning", ai_mode_2="qlearning")


from gui.connect4 import run_connect4
if __name__ == "__main__":
    # Example calls:
    #  1) run_connect4(ai_mode_1="none", ai_mode_2="none")
    # => Human vs Human
    #  2) run_connect4(ai_mode_1="minimax", ai_mode_2="none")
    # => Minimax (X) vs Human (O)
    #  3) run_connect4(ai_mode_1="minimax", ai_mode_2="qlearning")
    # => Q-:Learning (X) vs Human (O)
    run_connect4(ai_mode_1="qlearning", ai_mode_2="none")
    # => Minimax (X) vs Q-Learning (O)
    #  4) run_connect4(ai_mode_1="qlearning", ai_mode_2="qlearning")
    # => Q-Learning X vs Q-Learning O
    # run_connect4(ai_mode_1="qlearning", ai_mode_2="qlearning")
