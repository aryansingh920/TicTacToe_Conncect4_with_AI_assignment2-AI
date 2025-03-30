"""
Created on 30/03/2025

@author: Aryan

Filename: minmax.py

Relative Path: src/algorithm/minmax.py
"""



def minimax_tictactoe(state, maximizing_player, depth=0, max_depth=9):
    """
    Simple minimax for TicTacToe with depth limit.
    :param state: dict with:
        - "board": 2D list
        - "current_player": int (1 or 2)
        - "is_terminal": function(board)->bool
        - "evaluate": function(board, depth)->int
        - "get_children": function(state)->list of (move, child_state)
    :param maximizing_player: which player we consider the "max" side (usually X=1)
    :param depth: current recursion depth
    :param max_depth: cut-off
    :return: (best_score, best_move)
    """
    current_board = state["board"]
    current_player = state["current_player"]

    # Terminal or depth limit?
    if state["is_terminal"](current_board) or depth >= max_depth:
        return state["evaluate"](current_board, depth), None

    # Generate children
    children = state["get_children"](state)
    if not children:
        return 0, None  # Draw if no moves

    best_move = None

    if current_player == maximizing_player:
        best_score = float('-inf')
        for move, child_state in children:
            score, _ = minimax_tictactoe(child_state, maximizing_player,
                                         depth+1, max_depth)
            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = float('inf')
        for move, child_state in children:
            score, _ = minimax_tictactoe(child_state, maximizing_player,
                                         depth+1, max_depth)
            if score < best_score:
                best_score = score
                best_move = move

    return best_score, best_move
