"""
Created on 30/03/2025

@author: Aryan

Filename: tictactoe.py

Relative Path: src/gui/tictactoe.py
"""

import sys
import pygame
import pickle
import os
import random
import collections
import copy

# ---------------------------------------------------------------------
# From your existing code: imports of alpha-beta, minimax, QLearningAgent, etc.
from algorithm.minmax import minimax_tictactoe
from algorithm.minmax_alpha_beta import minimax_alpha_beta_tictactoe
from algorithm.qLearning import QLearningAgent
from state.tictactoe import TicTacToeStateManager
# ---------------------------------------------------------------------

pygame.init()

# -----------------------------
# CONFIG
# -----------------------------
WIDTH, HEIGHT = 600, 650
BOARD_OFFSET = 50
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = 600 // BOARD_ROWS

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
TEXT_COLOR = (255, 255, 255)
BUTTON_BG_COLOR = (40, 100, 90)

LINE_WIDTH = 10
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
font = pygame.font.SysFont(None, 40)

BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
reset_button_rect = pygame.Rect(
    WIDTH - (BUTTON_WIDTH + 20), 5, BUTTON_WIDTH, BUTTON_HEIGHT)

# Global game state
board = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]  # 3x3
player = 1  # 1 = X, 2 = O
game_over = False
x_wins = 0
o_wins = 0
winning_line = None
anim_progress = 0.0

# Create a global TicTacToeStateManager (optional usage)
ttt_manager = TicTacToeStateManager()

# Create a SINGLE Q agent with shared table
q_agent = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon=0.1, save_file='data/tictactoe/qtable.pkl')

# We'll store separate game trajectories for X and O
q_game_trajectory_X = []
q_game_trajectory_O = []

# -----------------------------
# DRAWING FUNCTIONS
# -----------------------------


def draw_board():
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, (10, 130, 120), (0, 0, WIDTH, BOARD_OFFSET))
    # Lines
    for i in range(1, BOARD_ROWS):
        start_pos = (0, BOARD_OFFSET + i*SQUARE_SIZE)
        end_pos = (WIDTH, BOARD_OFFSET + i*SQUARE_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, LINE_WIDTH)
    for i in range(1, BOARD_COLS):
        start_pos = (i*SQUARE_SIZE, BOARD_OFFSET)
        end_pos = (i*SQUARE_SIZE, BOARD_OFFSET + BOARD_ROWS*SQUARE_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, LINE_WIDTH)


def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = BOARD_OFFSET + row * SQUARE_SIZE + SQUARE_SIZE // 2
            if board[row][col] == 1:  # X
                pygame.draw.line(screen, CROSS_COLOR,
                                 (center_x - SPACE, center_y - SPACE),
                                 (center_x + SPACE, center_y + SPACE),
                                 CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR,
                                 (center_x - SPACE, center_y + SPACE),
                                 (center_x + SPACE, center_y - SPACE),
                                 CROSS_WIDTH)
            elif board[row][col] == 2:  # O
                pygame.draw.circle(screen, CIRCLE_COLOR,
                                   (center_x, center_y),
                                   SQUARE_SIZE//3, CIRCLE_WIDTH)


def draw_score():
    score_text = f"X - {x_wins}   |   O - {o_wins}"
    text_surface = font.render(score_text, True, TEXT_COLOR)
    text_rect = text_surface.get_rect(midleft=(10, BOARD_OFFSET // 2))
    screen.blit(text_surface, text_rect)


def draw_reset_button():
    pygame.draw.rect(screen, BUTTON_BG_COLOR, reset_button_rect)
    button_text = font.render("Reset", True, TEXT_COLOR)
    button_text_rect = button_text.get_rect(center=reset_button_rect.center)
    screen.blit(button_text, button_text_rect)


def draw_winning_line():
    global anim_progress
    if winning_line is None:
        return
    start_x, start_y, end_x, end_y = winning_line
    current_x = start_x + (end_x - start_x) * anim_progress
    current_y = start_y + (end_y - start_y) * anim_progress
    pygame.draw.line(screen, (200, 50, 50), (start_x, start_y),
                     (current_x, current_y), 10)
    anim_progress += 0.02
    if anim_progress >= 1.0:
        anim_progress = 1.0

# -----------------------------
# LOGIC FUNCTIONS
# -----------------------------


def mark_square(row, col, player_id):
    board[row][col] = player_id


def is_available(row, col):
    return board[row][col] == 0


def check_win(player_id):
    # Rows
    for r in range(BOARD_ROWS):
        if all(board[r][c] == player_id for c in range(BOARD_COLS)):
            return ('row', r)
    # Cols
    for c in range(BOARD_COLS):
        if all(board[r][c] == player_id for r in range(BOARD_ROWS)):
            return ('col', c)
    # Diagonal
    if all(board[i][i] == player_id for i in range(BOARD_ROWS)):
        return ('diag', 0)
    # Anti-diag
    if all(board[i][BOARD_COLS - 1 - i] == player_id for i in range(BOARD_ROWS)):
        return ('diag', 1)
    return None


def check_draw():
    return all(cell != 0 for row in board for cell in row)


def restart():
    global board, game_over, player, winning_line, anim_progress
    global q_game_trajectory_X, q_game_trajectory_O
    board = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
    game_over = False
    player = 1
    winning_line = None
    anim_progress = 0.0
    q_game_trajectory_X = []
    q_game_trajectory_O = []


def set_winning_line(win_info):
    global winning_line
    if not win_info:
        return
    typ, idx = win_info
    if typ == 'row':
        y = BOARD_OFFSET + idx*SQUARE_SIZE + SQUARE_SIZE//2
        winning_line = (0, y, WIDTH, y)
    elif typ == 'col':
        x = idx*SQUARE_SIZE + SQUARE_SIZE//2
        winning_line = (x, BOARD_OFFSET, x, BOARD_OFFSET +
                        BOARD_ROWS*SQUARE_SIZE)
    elif typ == 'diag':
        if idx == 0:
            winning_line = (0, BOARD_OFFSET, BOARD_COLS *
                            SQUARE_SIZE, BOARD_OFFSET + BOARD_ROWS*SQUARE_SIZE)
        else:
            winning_line = (BOARD_COLS*SQUARE_SIZE, BOARD_OFFSET,
                            0, BOARD_OFFSET + BOARD_ROWS*SQUARE_SIZE)

# -----------------------------
# AI HELPER FUNCTIONS
# -----------------------------


def transform_board_for_O(bd):
    """
    Transform the board so that from O's perspective, it looks like X's perspective.
    This allows using the same Q-table for both players.
    X (1) becomes O (2) and vice versa.
    """
    transformed = copy.deepcopy(bd)
    for r in range(3):
        for c in range(3):
            if transformed[r][c] == 1:
                transformed[r][c] = 2
            elif transformed[r][c] == 2:
                transformed[r][c] = 1
    return transformed


def tictactoe_is_terminal(bd):
    # Reuse your manager or do manual checks
    return ttt_manager._is_terminal(bd)


def tictactoe_evaluate(bd, depth):
    # Identical to your existing scoring
    # +10 - depth if X wins, -10 + depth if O wins, 0 otherwise
    for i in range(3):
        if bd[i][0] == bd[i][1] == bd[i][2] == 1:
            return 10 - depth
        if bd[i][0] == bd[i][1] == bd[i][2] == 2:
            return depth - 10
    for i in range(3):
        if bd[0][i] == bd[1][i] == bd[2][i] == 1:
            return 10 - depth
        if bd[0][i] == bd[1][i] == bd[2][i] == 2:
            return depth - 10
    if bd[0][0] == bd[1][1] == bd[2][2] == 1:
        return 10 - depth
    if bd[0][0] == bd[1][1] == bd[2][2] == 2:
        return depth - 10
    if bd[0][2] == bd[1][1] == bd[2][0] == 1:
        return 10 - depth
    if bd[0][2] == bd[1][1] == bd[2][0] == 2:
        return depth - 10
    # no winner => 0
    return 0


def tictactoe_get_children(state):
    bd = state["board"]
    current_player = state["current_player"]
    children = []
    for r in range(3):
        for c in range(3):
            if bd[r][c] == 0:
                new_bd = [row[:] for row in bd]
                new_bd[r][c] = current_player
                next_player = 1 if current_player == 2 else 2
                child_state = {
                    "board": new_bd,
                    "current_player": next_player,
                    "is_terminal": state["is_terminal"],
                    "evaluate": state["evaluate"],
                    "get_children": state["get_children"]
                }
                children.append(((r, c), child_state))
    return children


def get_ai_move(ai_mode, current_player):
    """
    Decide the best move for the CURRENT player given ai_mode.
    Return (row, col).
    """
    global q_game_trajectory_X, q_game_trajectory_O

    # Build a 'state' dict for minimax-based methods
    state = {
        "board": [row[:] for row in board],
        "current_player": current_player,
        "is_terminal": tictactoe_is_terminal,
        "evaluate": tictactoe_evaluate,
        "get_children": tictactoe_get_children
    }

    if ai_mode == "minimax":
        score, move = minimax_tictactoe(state, maximizing_player=1)
        return move
    elif ai_mode == "alpha-beta":
        score, move = minimax_alpha_beta_tictactoe(state, maximizing_player=1)
        return move
    elif ai_mode == "qlearning":
        # Use the single Q-agent for both players, but transform the board for O
        valid_moves = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if board[r][c] == 0:
                    valid_moves.append((r, c))
        if not valid_moves:
            return None

        # Select trajectory based on current player
        q_trajectory = q_game_trajectory_X if current_player == 1 else q_game_trajectory_O

        # For player O, transform the board so the agent sees it as X
        if current_player == 2:
            transformed_board = transform_board_for_O(board)
            chosen_move = q_agent.choose_action(transformed_board, valid_moves)
        else:
            chosen_move = q_agent.choose_action(board, valid_moves)

        # Store transition
        board_copy = [row[:] for row in board]
        q_trajectory.append((board_copy, chosen_move))
        return chosen_move
    else:
        return None

# -----------------------------
# MAIN LOOP (MODIFIED FOR 2 AI)
# -----------------------------


def run_tictactoe(ai_mode_1="none", ai_mode_2="none"):
    """
    Run the TicTacToe game loop:
      - ai_mode_1 is the AI type for Player X (1)
      - ai_mode_2 is the AI type for Player O (2)
      - If ai_mode = "none", that player is human.
      - If ai_mode = "qlearning", that player uses QLearningAgent.
      - If ai_mode = "minimax"/"alpha-beta", that player uses minimax-based approach.
    """
    global player, game_over, x_wins, o_wins
    global q_game_trajectory_X, q_game_trajectory_O

    restart()  # start fresh

    clock = pygame.time.Clock()
    while True:
        clock.tick(60)

        # AI Move for Player 1 (X)
        if not game_over and ai_mode_1 != "none" and player == 1:
            move = get_ai_move(ai_mode_1, current_player=1)
            if move:
                r, c = move
                if is_available(r, c):
                    mark_square(r, c, 1)
                    win_info = check_win(1)
                    if win_info:
                        set_winning_line(win_info)
                        game_over = True
                        x_wins += 1
                        # Q-learning reward if ai_mode_1 is qlearning
                        if ai_mode_1 == "qlearning":
                            # X just won => final reward = +1 for X
                            q_agent.batch_update_from_game(
                                q_game_trajectory_X, final_reward=+1)
                        # If O was also Q-learning, O gets negative reward
                        if ai_mode_2 == "qlearning":
                            q_agent.batch_update_from_game(
                                q_game_trajectory_O, final_reward=-1)

                    elif check_draw():
                        game_over = True
                        # If Q-learning: reward=0 for both
                        if ai_mode_1 == "qlearning":
                            q_agent.batch_update_from_game(
                                q_game_trajectory_X, final_reward=0)
                        if ai_mode_2 == "qlearning":
                            q_agent.batch_update_from_game(
                                q_game_trajectory_O, final_reward=0)
                    else:
                        player = 2
                else:
                    # invalid move => pass
                    pass
            else:
                # No move => likely a draw
                game_over = True

        # AI Move for Player 2 (O)
        elif not game_over and ai_mode_2 != "none" and player == 2:
            move = get_ai_move(ai_mode_2, current_player=2)
            if move:
                r, c = move
                if is_available(r, c):
                    mark_square(r, c, 2)
                    win_info = check_win(2)
                    if win_info:
                        set_winning_line(win_info)
                        game_over = True
                        o_wins += 1
                        # Q-learning reward if ai_mode_2 is qlearning
                        if ai_mode_2 == "qlearning":
                            # O just won => final reward = +1 for O
                            q_agent.batch_update_from_game(
                                q_game_trajectory_O, final_reward=+1)
                        # If X was also Q-learning, X gets negative reward
                        if ai_mode_1 == "qlearning":
                            q_agent.batch_update_from_game(
                                q_game_trajectory_X, final_reward=-1)
                    elif check_draw():
                        game_over = True
                        # If Q-learning: reward=0
                        if ai_mode_2 == "qlearning":
                            q_agent.batch_update_from_game(
                                q_game_trajectory_O, final_reward=0)
                        if ai_mode_1 == "qlearning":
                            q_agent.batch_update_from_game(
                                q_game_trajectory_X, final_reward=0)
                    else:
                        player = 1
                else:
                    # invalid move => pass
                    pass
            else:
                # No move => presumably a draw
                game_over = True

        # Check Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                # Reset button?
                if reset_button_rect.collidepoint(mouseX, mouseY):
                    restart()
                else:
                    # If it's a human player's turn => place
                    # Player 1 is human if ai_mode_1=="none"
                    if not game_over and player == 1 and ai_mode_1 == "none":
                        if mouseY >= BOARD_OFFSET:
                            row = (mouseY - BOARD_OFFSET) // SQUARE_SIZE
                            col = mouseX // SQUARE_SIZE
                            if (0 <= row < BOARD_ROWS) and (0 <= col < BOARD_COLS):
                                if is_available(row, col):
                                    mark_square(row, col, 1)
                                    win_info = check_win(1)
                                    if win_info:
                                        set_winning_line(win_info)
                                        game_over = True
                                        x_wins += 1
                                        if ai_mode_2 == "qlearning":
                                            # O lost => final_reward = -1
                                            q_agent.batch_update_from_game(
                                                q_game_trajectory_O, final_reward=-1)
                                    elif check_draw():
                                        game_over = True
                                        if ai_mode_2 == "qlearning":
                                            q_agent.batch_update_from_game(
                                                q_game_trajectory_O, final_reward=0)
                                    else:
                                        player = 2

                    # Player 2 is human if ai_mode_2=="none"
                    elif not game_over and player == 2 and ai_mode_2 == "none":
                        if mouseY >= BOARD_OFFSET:
                            row = (mouseY - BOARD_OFFSET) // SQUARE_SIZE
                            col = mouseX // SQUARE_SIZE
                            if (0 <= row < BOARD_ROWS) and (0 <= col < BOARD_COLS):
                                if is_available(row, col):
                                    mark_square(row, col, 2)
                                    win_info = check_win(2)
                                    if win_info:
                                        set_winning_line(win_info)
                                        game_over = True
                                        o_wins += 1
                                        if ai_mode_1 == "qlearning":
                                            # X lost => final_reward = -1
                                            q_agent.batch_update_from_game(
                                                q_game_trajectory_X, final_reward=-1)
                                    elif check_draw():
                                        game_over = True
                                        if ai_mode_1 == "qlearning":
                                            q_agent.batch_update_from_game(
                                                q_game_trajectory_X, final_reward=0)
                                    else:
                                        player = 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart()

        # Drawing
        draw_board()
        draw_score()
        draw_reset_button()
        draw_figures()

        if game_over and winning_line is not None and anim_progress < 1.0:
            draw_winning_line()

        pygame.display.update()


# ---------------------------------------------------------------------
# If you run this file directly:
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     # Example calls:
#     #  1) run_tictactoe(ai_mode_1="none", ai_mode_2="none") => Human vs Human
#     #  2) run_tictactoe(ai_mode_1="minimax", ai_mode_2="none") => Minimax (X) vs Human (O)
#     #  3) run_tictactoe(ai_mode_1="minimax", ai_mode_2="qlearning") => Minimax (X) vs Q-Learning (O)
#     #  4) run_tictactoe(ai_mode_1="qlearning", ai_mode_2="qlearning") => Q-Learning X vs Q-Learning O
#     run_tictactoe(ai_mode_1="qlearning", ai_mode_2="qlearning")
