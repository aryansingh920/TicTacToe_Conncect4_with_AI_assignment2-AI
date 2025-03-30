"""
Created on 30/03/2025

@author: Aryan

Filename: minmax_connect4.py
Relative Path: src/algorithm/minmax_connect4.py
"""


def minimax_connect4(state, maximizing_player, depth=0, max_depth=6):
    """
    Minimax for Connect 4 with optional depth limit.

    :param state: dict with keys:
        - "board": current 2D board
        - "current_player": int 1 or 2
        - "is_terminal": callable(board)->bool
        - "evaluate": callable(board, depth)->int
        - "get_children": callable(state)->list[(move, child_state)]
    :param maximizing_player: 1 or 2, whichever we treat as the "maximizing" side
    :param depth: current recursion depth
    :param max_depth: maximum recursion depth
    :return: (best_score, best_move) 
        best_move is the column index to drop the piece
    """
    board = state["board"]
    current_player = state["current_player"]
    is_terminal = state["is_terminal"]
    evaluate = state["evaluate"]
    get_children = state["get_children"]

    if is_terminal(board) or depth >= max_depth:
        return evaluate(board, depth), None

    children = get_children(state)  # list of (move, child_state)
    if not children:
        # No possible moves => treat as a draw
        return 0, None

    best_move = None

    if current_player == maximizing_player:
        best_score = float("-inf")
        for (move, child) in children:
            score, _ = minimax_connect4(
                child, maximizing_player, depth+1, max_depth)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move
    else:
        best_score = float("inf")
        for (move, child) in children:
            score, _ = minimax_connect4(
                child, maximizing_player, depth+1, max_depth)
            if score < best_score:
                best_score = score
                best_move = move
        return best_score, best_move
