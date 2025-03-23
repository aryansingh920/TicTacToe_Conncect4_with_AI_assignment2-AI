"""
Filename: minimax.py
Path: algorithm/minimax.py

This file implements a simple Minimax approach (no alpha-beta) for
TicTacToe and Connect4. We assume you pass in a game state along with a
few helper functions so that the same code can be used for both.
"""

import math


def minimax_tictactoe(state, maximizing_player, depth=0):
    """
    Simple minimax for TicTacToe.
    :param state: dictionary or object that holds:
                  - 'board': 2D list
                  - 'current_player': integer (1 or 2)
                  - helper methods: is_terminal, evaluate, get_children
    :param maximizing_player: integer indicating which player we are maximizing for
    :param depth: recursion depth
    :return: (best_score, best_move)
    """

    # Check if terminal
    if state["is_terminal"](state["board"]):
        # Evaluate final board
        return state["evaluate"](state["board"], depth), None

    best_move = None

    if state["current_player"] == maximizing_player:
        best_score = -math.inf
        # Generate all possible child states
        for move, child_state in state["get_children"](state):
            score, _ = minimax_tictactoe(
                child_state, maximizing_player, depth+1)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move
    else:
        best_score = math.inf
        for move, child_state in state["get_children"](state):
            score, _ = minimax_tictactoe(
                child_state, maximizing_player, depth+1)
            if score < best_score:
                best_score = score
                best_move = move
        return best_score, best_move
