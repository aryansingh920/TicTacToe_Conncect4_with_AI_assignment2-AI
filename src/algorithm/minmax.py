"""
Filename: minimax.py
Path: algorithm/minimax.py

This file implements a simple Minimax approach (no alpha-beta) for
TicTacToe and Connect4. We assume you pass in a game state along with a
few helper functions so that the same code can be used for both.
"""

import math


def minimax_tictactoe(state, maximizing_player, depth=0, max_depth=9):
    """
    Simple minimax for TicTacToe with depth limit to prevent infinite recursion.
    :param state: dictionary containing board state and helper functions
    :param maximizing_player: integer indicating which player we are maximizing for (1 or 2)
    :param depth: current recursion depth
    :param max_depth: maximum recursion depth to prevent infinite loops
    :return: (best_score, best_move)
    """
    current_board = state["board"]
    current_player = state["current_player"]

    # Check if terminal or max depth reached
    if state["is_terminal"](current_board) or depth >= max_depth:
        return state["evaluate"](current_board, depth), None

    # Get all possible moves
    children = state["get_children"](state)

    # If no valid moves, it's a draw
    if not children:
        return 0, None

    best_move = None

    if current_player == maximizing_player:
        best_score = float('-inf')
        for move, child_state in children:
            score, _ = minimax_tictactoe(
                child_state, maximizing_player, depth+1, max_depth)
            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = float('inf')
        for move, child_state in children:
            score, _ = minimax_tictactoe(
                child_state, maximizing_player, depth+1, max_depth)
            if score < best_score:
                best_score = score
                best_move = move

    return best_score, best_move
