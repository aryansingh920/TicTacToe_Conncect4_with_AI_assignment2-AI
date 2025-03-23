"""
Filename: minimax_alpha_beta.py
Path: algorithm/minimax_alpha_beta.py

This file implements Minimax with alpha-beta pruning for TicTacToe
and Connect4, similar to the plain Minimax approach but with alpha, beta cuts.
"""

import math


def minimax_alpha_beta_tictactoe(state, maximizing_player, alpha=float('-inf'), beta=float('inf'), depth=0, max_depth=9):
    """
    Minimax with alpha-beta pruning for TicTacToe.
    :param state: dictionary containing board state and helper functions
    :param maximizing_player: which player we are maximizing for (1 or 2)
    :param alpha: best value the maximizer can guarantee so far
    :param beta: best value the minimizer can guarantee so far
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
            score, _ = minimax_alpha_beta_tictactoe(
                child_state, maximizing_player, alpha, beta, depth+1, max_depth
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # beta cut-off
    else:
        best_score = float('inf')
        for move, child_state in children:
            score, _ = minimax_alpha_beta_tictactoe(
                child_state, maximizing_player, alpha, beta, depth+1, max_depth
            )
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # alpha cut-off

    return best_score, best_move


def minimax_alpha_beta_connect4(state, maximizing_player, alpha=float('-inf'), beta=float('inf'), depth=0, max_depth=5):
    """
    Minimax with alpha-beta pruning for Connect4.
    
    :param state: dictionary containing board state and helper functions
    :param maximizing_player: which player we are maximizing for (1 or 2)
    :param alpha: best value the maximizer can guarantee so far
    :param beta: best value the minimizer can guarantee so far
    :param depth: current recursion depth
    :param max_depth: maximum recursion depth to prevent infinite loops
    :return: (best_score, best_move)
    """
    current_board = state["board"]
    current_player = state["current_player"]

    # Check if terminal state or max depth reached
    if state["is_terminal"](current_board) or depth >= max_depth:
        return state["evaluate"](current_board, depth), None

    # Get all possible moves
    children = state["get_children"](state)

    # If no valid moves, it's a draw
    if not children:
        return 0, None

    best_move = children[0][0]  # Default to first move

    if current_player == maximizing_player:
        best_score = float('-inf')
        for move, child_state in children:
            score, _ = minimax_alpha_beta_connect4(
                child_state, maximizing_player, alpha, beta, depth+1, max_depth
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # beta cut-off
    else:
        best_score = float('inf')
        for move, child_state in children:
            score, _ = minimax_alpha_beta_connect4(
                child_state, maximizing_player, alpha, beta, depth+1, max_depth
            )
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # alpha cut-off

    return best_score, best_move
