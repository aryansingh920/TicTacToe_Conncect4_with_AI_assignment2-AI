"""
Filename: minimax_alpha_beta.py
Path: algorithm/minimax_alpha_beta.py

This file implements Minimax with alpha-beta pruning for TicTacToe
and Connect4, similar to the plain Minimax approach but with alpha, beta cuts.
"""

import math


def minimax_alpha_beta_tictactoe(state, maximizing_player, alpha=-math.inf, beta=math.inf, depth=0):
    """
    Minimax with alpha-beta pruning for TicTacToe.
    :param state: dictionary or object that holds:
                  - 'board': 2D list
                  - 'current_player': integer (1 or 2)
                  - helper methods: is_terminal, evaluate, get_children
    :param maximizing_player: which player we are maximizing for
    :param alpha: best value the maximizer can guarantee so far
    :param beta: best value the minimizer can guarantee so far
    :param depth: recursion depth
    :return: (best_score, best_move)
    """

    if state["is_terminal"](state["board"]):
        return state["evaluate"](state["board"], depth), None

    best_move = None

    if state["current_player"] == maximizing_player:
        best_score = -math.inf
        for move, child_state in state["get_children"](state):
            score, _ = minimax_alpha_beta_tictactoe(child_state, maximizing_player,
                                                    alpha, beta, depth+1)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # beta cut-off
        return best_score, best_move
    else:
        best_score = math.inf
        for move, child_state in state["get_children"](state):
            score, _ = minimax_alpha_beta_tictactoe(child_state, maximizing_player,
                                                    alpha, beta, depth+1)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # alpha cut-off
        return best_score, best_move
