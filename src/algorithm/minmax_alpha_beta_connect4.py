"""
Created on 30/03/2025

@author: Aryan

Filename: minmax_alpha_beta_connect4.py
Relative Path: src/algorithm/minmax_alpha_beta_connect4.py
"""


def minimax_alpha_beta_connect4(
    state,
    maximizing_player,
    alpha=float("-inf"),
    beta=float("inf"),
    depth=0,
    max_depth=6
):
    """
    Alpha-Beta pruning for Connect 4.

    :param state: dict with "board", "current_player", "is_terminal", "evaluate", "get_children"
    :param maximizing_player: 1 or 2
    :return: (best_score, best_move)
    """
    board = state["board"]
    current_player = state["current_player"]
    is_terminal = state["is_terminal"]
    evaluate = state["evaluate"]
    get_children = state["get_children"]

    if is_terminal(board) or depth >= max_depth:
        return evaluate(board, depth), None

    children = get_children(state)
    if not children:
        return 0, None

    best_move = None

    if current_player == maximizing_player:
        best_score = float("-inf")
        for (move, child) in children:
            score, _ = minimax_alpha_beta_connect4(
                child,
                maximizing_player,
                alpha, beta,
                depth+1,
                max_depth
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # beta cutoff
        return best_score, best_move
    else:
        best_score = float("inf")
        for (move, child) in children:
            score, _ = minimax_alpha_beta_connect4(
                child,
                maximizing_player,
                alpha, beta,
                depth+1,
                max_depth
            )
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # alpha cutoff
        return best_score, best_move
