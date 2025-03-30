"""
Created on 30/03/2025

@author: Aryan

Filename: minmax_alpha_beta.py

Relative Path: src/algorithm/minmax_alpha_beta.py
"""


def minimax_alpha_beta_tictactoe(state, maximizing_player,
                                 alpha=float('-inf'), beta=float('inf'),
                                 depth=0, max_depth=9):
    """
    Alpha-Beta for TicTacToe.
    :param state: dict with "board", "current_player", etc.
    :param maximizing_player: 1 or 2, whichever we treat as "max"
    :return: (best_score, best_move)
    """
    current_board = state["board"]
    current_player = state["current_player"]

    # Terminal or depth limit?
    if state["is_terminal"](current_board) or depth >= max_depth:
        return state["evaluate"](current_board, depth), None

    children = state["get_children"](state)
    if not children:
        return 0, None

    best_move = None

    if current_player == maximizing_player:
        best_score = float('-inf')
        for move, child_state in children:
            score, _ = minimax_alpha_beta_tictactoe(
                child_state, maximizing_player, alpha, beta,
                depth+1, max_depth
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Beta cutoff
    else:
        best_score = float('inf')
        for move, child_state in children:
            score, _ = minimax_alpha_beta_tictactoe(
                child_state, maximizing_player, alpha, beta,
                depth+1, max_depth
            )
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # Alpha cutoff

    return best_score, best_move
