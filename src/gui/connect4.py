"""
Created on 30/03/2025

@author: Aryan

Filename: connect4.py
Relative Path: gui/connect4.py

This version supports:
  - Human vs Human
  - Human vs AI (Minimax, Alpha-Beta, Q-Learning)
  - AI vs AI (Minimax, Alpha-Beta, Q-Learning)
"""

import sys
import pygame
import os
import copy  # for copying board states in Q-Learning

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from state.connect4 import Connect4StateManager
from algorithm.minmax_connect4 import minimax_connect4
from algorithm.minmax_alpha_beta_connect4 import minimax_alpha_beta_connect4

# Import your QLearningAgent
from algorithm.qLearning import QLearningAgent

pygame.init()

# ----------------------------------------------------------------------
# GLOBAL CONFIG
# ----------------------------------------------------------------------
BOARD_N = 4  # If your Connect4StateManager is for a 4x4 board
SQUARE_SIZE = 100
BOARD_OFFSET = 60
WIDTH = BOARD_N * SQUARE_SIZE
HEIGHT = BOARD_OFFSET + (BOARD_N * SQUARE_SIZE)

BG_COLOR = (28, 170, 156)
HEADER_COLOR = (10, 130, 120)
LINE_COLOR = (23, 145, 135)
RED_COLOR = (255, 0, 0)
YELLOW_COLOR = (255, 255, 0)
EMPTY_COLOR = (40, 40, 40)
BUTTON_BG_COLOR = (40, 100, 90)
TEXT_COLOR = (255, 255, 255)

BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect 4")
font = pygame.font.SysFont(None, 40)

# ----------------------------------------------------------------------
# GLOBAL GAME STATE
# ----------------------------------------------------------------------
board = [[0]*BOARD_N for _ in range(BOARD_N)]  # 0=empty, 1=Red, 2=Yellow
player = 1  # 1=Red, 2=Yellow
game_over = False
red_wins = 0
yellow_wins = 0
winning_line = None
anim_progress = 0.0

# "Reset" button position
reset_button_rect = pygame.Rect(
    WIDTH - (BUTTON_WIDTH + 20), 5, BUTTON_WIDTH, BUTTON_HEIGHT
)

# Connect 4 State Manager (for next-move generation & terminal checks)
c4_manager = None


def initialize_c4_manager():
    """Lazy-load the Connect4StateManager if not already done."""
    global c4_manager
    if c4_manager is None:
        c4_manager = Connect4StateManager(
            n=BOARD_N,
            cache_file='data/connect4/connect4_states_4x4.pkl'
        )


# ----------------------------------------------------------------------
# Q-LEARNING AGENTS (one for Red, one for Yellow)
# ----------------------------------------------------------------------
# If you prefer to only have a single agent controlling both sides, you can,
# but typically we keep separate Q-tables so each color can learn distinctly.
q_agent_red = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon=0.1, save_file='data/connect4/qtable_red.pkl'
)
q_agent_yellow = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon=0.1, save_file='data/connect4/qtable_yellow.pkl'
)

# We store each agent's (state, action) pairs for a single game.
game_trajectory_red = []
game_trajectory_yellow = []


# ----------------------------------------------------------------------
# HELPER FUNCTIONS: Board Keys, Terminal Checks, Winners
# ----------------------------------------------------------------------
def board_to_key(bd):
    """Convert a 2D board into a string or tuple key (manager uses string or tuple)."""
    return ''.join(str(cell) for row in bd for cell in row)


def connect4_is_terminal(bd):
    """Use manager's stored data to check if the board is terminal."""
    key = board_to_key(bd)
    info = c4_manager.states.get(key, {})
    return info.get('is_terminal', False)


def connect4_winner(bd):
    """Return the winner from manager's data: 0, 1, or 2."""
    key = board_to_key(bd)
    info = c4_manager.states.get(key, {})
    return info.get('winner', 0)


# ----------------------------------------------------------------------
# EVALUATION FOR MINIMAX & ALPHA-BETA
# ----------------------------------------------------------------------
def connect4_evaluate(bd, depth):
    """
    Enhanced evaluation for minimax/alpha-beta:
      - Terminal states: use a high magnitude score.
      - Non-terminal states: score based on potential connect opportunities.
    """
    w = connect4_winner(bd)
    if w == 1:
        return 1000 - depth
    elif w == 2:
        return depth - 1000

    rows = len(bd)
    cols = len(bd[0]) if rows > 0 else 0
    score = 0

    def get_window(r, c, dr, dc, length=4):
        window = []
        for i in range(length):
            rr = r + i*dr
            cc = c + i*dc
            if 0 <= rr < rows and 0 <= cc < cols:
                window.append(bd[rr][cc])
            else:
                break
        return window

    def score_window(window):
        red_count = window.count(1)
        yellow_count = window.count(2)

        if red_count > 0 and yellow_count == 0:
            # Weighted heuristics
            if red_count == 3 and len(window) == 4:
                return 50
            elif red_count == 2 and len(window) == 4:
                return 5
            elif red_count == 2 and len(window) == 3:
                return 3
        elif yellow_count > 0 and red_count == 0:
            if yellow_count == 3 and len(window) == 4:
                return -50
            elif yellow_count == 2 and len(window) == 4:
                return -5
            elif yellow_count == 2 and len(window) == 3:
                return -3
        return 0

    # Horizontal (length 4)
    for r in range(rows):
        for c in range(cols - 3):
            window = get_window(r, c, 0, 1, 4)
            score += score_window(window)

    # Vertical (length 4)
    for r in range(rows - 3):
        for c in range(cols):
            window = get_window(r, c, 1, 0, 4)
            score += score_window(window)

    # Diagonal \ (length 4)
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = get_window(r, c, 1, 1, 4)
            score += score_window(window)

    # Diagonal / (length 4)
    for r in range(3, rows):
        for c in range(cols - 3):
            window = get_window(r, c, -1, 1, 4)
            score += score_window(window)

    # Also check 3-length windows if you like to refine scoring:
    # (Below is optional / for further refinement)
    for r in range(rows):
        for c in range(cols - 2):
            window = get_window(r, c, 0, 1, 3)
            score += score_window(window)
    for r in range(rows - 2):
        for c in range(cols):
            window = get_window(r, c, 1, 0, 3)
            score += score_window(window)
    for r in range(rows - 2):
        for c in range(cols - 2):
            window = get_window(r, c, 1, 1, 3)
            score += score_window(window)
    for r in range(2, rows):
        for c in range(cols - 2):
            window = get_window(r, c, -1, 1, 3)
            score += score_window(window)

    return score


# ----------------------------------------------------------------------
# GENERATE CHILD STATES (for minimax / alpha-beta)
# ----------------------------------------------------------------------
def connect4_get_children(state):
    """
    Returns a list of (move, child_state) pairs for the state.
    'move' will be the column index. 'child_state' is the next board/player.
    """
    bd = state["board"]
    current_player = state["current_player"]
    children = []

    next_moves = c4_manager.get_next_possible_moves(bd)
    for move_info in next_moves:
        col = move_info['move']
        new_bd = move_info['new_board']
        next_player = 1 if current_player == 2 else 2

        child_state = {
            "board": new_bd,
            "current_player": next_player,
            "is_terminal": connect4_is_terminal,
            "evaluate": connect4_evaluate,
            "get_children": connect4_get_children
        }
        children.append((col, child_state))
    return children


# ----------------------------------------------------------------------
# GET AI MOVE (for Minimax, Alpha-Beta, or Q-Learning)
# ----------------------------------------------------------------------
def get_ai_move(ai_mode, current_player):
    """
    Return a column index or None if no moves.
    If ai_mode is "minimax" or "alpha-beta", we call the minimax family.
    If ai_mode is "qlearning", we do Q-Learning.
    """
    # If Q-Learning, we'll do it differently below.
    if ai_mode == "qlearning":
        return get_qlearning_move(current_player)

    # Build a state dict for minimax-based approaches
    state = {
        "board": [row[:] for row in board],
        "current_player": current_player,
        "is_terminal": connect4_is_terminal,
        "evaluate": connect4_evaluate,
        "get_children": connect4_get_children
    }

    if connect4_is_terminal(board):
        return None
    next_moves = c4_manager.get_next_possible_moves(board)
    if not next_moves:
        return None

    if ai_mode == "minimax":
        score, best_col = minimax_connect4(
            state, maximizing_player=1, depth=0, max_depth=6
        )
        return best_col
    elif ai_mode == "alpha-beta":
        score, best_col = minimax_alpha_beta_connect4(
            state, maximizing_player=1, depth=0, max_depth=6
        )
        return best_col

    # default fallback
    return None


def get_qlearning_move(current_player):
    """
    Choose a move (column) via Q-Learning agent (epsilon-greedy).
    We'll get valid moves from c4_manager as well, then pick from Q-Table.
    """
    next_moves = c4_manager.get_next_possible_moves(board)
    if not next_moves:
        return None

    # List out valid column moves
    valid_moves = [move_info['move'] for move_info in next_moves]

    # Distinguish which Q-Agent is playing
    if current_player == 1:
        agent = q_agent_red
    else:
        agent = q_agent_yellow

    # QLearningAgent expects:
    #   choose_action(board, valid_actions)
    # So we pass the current board and the list of valid columns.
    chosen_move = agent.choose_action(board, valid_moves)
    return chosen_move


# ----------------------------------------------------------------------
# GAME LOGIC
# ----------------------------------------------------------------------
def is_valid_column(col):
    return board[0][col] == 0  # top cell is empty => valid move


def get_next_open_row(col):
    for r in range(BOARD_N-1, -1, -1):
        if board[r][col] == 0:
            return r
    return None


def drop_piece(row, col, piece):
    board[row][col] = piece


def check_draw():
    """If no empty cell exists, it's a draw."""
    return all(board[r][c] != 0 for r in range(BOARD_N) for c in range(BOARD_N))


def find_winning_positions(piece):
    """
    Return a list of 4 consecutive (r,c) positions if 'piece' forms a connect-4, else None.
    We use this to animate the line on the board.
    """
    # Horizontal
    for r in range(BOARD_N):
        for c in range(BOARD_N - 3):
            if (board[r][c] == piece and
                board[r][c+1] == piece and
                board[r][c+2] == piece and
                    board[r][c+3] == piece):
                return [(r, c), (r, c+1), (r, c+2), (r, c+3)]

    # Vertical
    for c in range(BOARD_N):
        for r in range(BOARD_N - 3):
            if (board[r][c] == piece and
                board[r+1][c] == piece and
                board[r+2][c] == piece and
                    board[r+3][c] == piece):
                return [(r, c), (r+1, c), (r+2, c), (r+3, c)]

    # Diagonal down-right
    for r in range(BOARD_N - 3):
        for c in range(BOARD_N - 3):
            if (board[r][c] == piece and
                board[r+1][c+1] == piece and
                board[r+2][c+2] == piece and
                    board[r+3][c+3] == piece):
                return [(r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3)]

    # Diagonal up-right
    for r in range(3, BOARD_N):
        for c in range(BOARD_N - 3):
            if (board[r][c] == piece and
                board[r-1][c+1] == piece and
                board[r-2][c+2] == piece and
                    board[r-3][c+3] == piece):
                return [(r, c), (r-1, c+1), (r-2, c+2), (r-3, c+3)]
    return None


def restart():
    global board, game_over, player, winning_line, anim_progress
    global game_trajectory_red, game_trajectory_yellow
    board = [[0]*BOARD_N for _ in range(BOARD_N)]
    game_over = False
    player = 1
    winning_line = None
    anim_progress = 0.0

    # Reset the game trajectories
    game_trajectory_red = []
    game_trajectory_yellow = []


# ----------------------------------------------------------------------
# ANIMATION (Winning Line)
# ----------------------------------------------------------------------
def set_winning_line(win_positions):
    """Create an animation line from the first position to the last in the list."""
    global winning_line, anim_progress
    if not win_positions:
        return

    (start_r, start_c) = win_positions[0]
    (end_r, end_c) = win_positions[-1]

    sx = start_c * SQUARE_SIZE + (SQUARE_SIZE // 2)
    sy = BOARD_OFFSET + start_r * SQUARE_SIZE + (SQUARE_SIZE // 2)
    ex = end_c * SQUARE_SIZE + (SQUARE_SIZE // 2)
    ey = BOARD_OFFSET + end_r * SQUARE_SIZE + (SQUARE_SIZE // 2)

    winning_line = (sx, sy, ex, ey)
    anim_progress = 0.0


def draw_winning_line():
    global anim_progress
    if winning_line is None:
        return
    (sx, sy, ex, ey) = winning_line
    cx = sx + (ex - sx) * anim_progress
    cy = sy + (ey - sy) * anim_progress
    pygame.draw.line(screen, (200, 50, 50), (sx, sy), (cx, cy), 10)
    anim_progress += 0.02
    if anim_progress > 1.0:
        anim_progress = 1.0


# ----------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------
def draw_header():
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, HEADER_COLOR, (0, 0, WIDTH, BOARD_OFFSET))


def draw_reset_button():
    pygame.draw.rect(screen, BUTTON_BG_COLOR, reset_button_rect)
    text = font.render("Reset", True, TEXT_COLOR)
    text_rect = text.get_rect(center=reset_button_rect.center)
    screen.blit(text, text_rect)


def draw_score():
    global red_wins, yellow_wins
    score_text = f"Red - {red_wins}   |   Yellow - {yellow_wins}"
    text_surface = font.render(score_text, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 10))


def draw_pieces():
    """
    row=0 at top visually, row=BOARD_N-1 at bottom.
    So we simply loop row-wise to draw circles.
    """
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            center_x = c * SQUARE_SIZE + (SQUARE_SIZE // 2)
            center_y = BOARD_OFFSET + r * SQUARE_SIZE + (SQUARE_SIZE // 2)
            color = EMPTY_COLOR
            if board[r][c] == 1:
                color = RED_COLOR
            elif board[r][c] == 2:
                color = YELLOW_COLOR
            pygame.draw.circle(
                screen, color, (center_x, center_y), SQUARE_SIZE//2 - 5)


# ----------------------------------------------------------------------
# FINAL REWARD ASSIGNMENT FOR Q-LEARNING
# ----------------------------------------------------------------------
def assign_qlearning_rewards(winner):
    """
    If Red is winner => red reward = +1, yellow reward = -1.
    If Yellow is winner => red reward = -1, yellow reward = +1.
    If draw => 0 for both.
    Then call batch_update_from_game(trajectory, final_reward).
    """
    if winner == 1:
        red_final = 1.0
        yellow_final = -1.0
    elif winner == 2:
        red_final = -1.0
        yellow_final = 1.0
    else:
        # draw or no winner => 0 for both
        red_final = 0.0
        yellow_final = 0.0

    # Update the Q-tables in one go
    if game_trajectory_red:
        q_agent_red.batch_update_from_game(game_trajectory_red, red_final)
    if game_trajectory_yellow:
        q_agent_yellow.batch_update_from_game(
            game_trajectory_yellow, yellow_final)


# ----------------------------------------------------------------------
# MAIN LOOP for Connect 4
# ----------------------------------------------------------------------
def run_connect4(ai_mode_1="none", ai_mode_2="none"):
    """
    ai_mode_1 => how Red (player=1) plays
    ai_mode_2 => how Yellow (player=2) plays

    Possible values: "none", "minimax", "alpha-beta", "qlearning"
    """
    initialize_c4_manager()
    global player, game_over, red_wins, yellow_wins
    global game_trajectory_red, game_trajectory_yellow

    restart()
    clock = pygame.time.Clock()

    while True:
        clock.tick(60)

        # If it's Red's turn and Red is AI
        if not game_over and ai_mode_1 != "none" and player == 1:
            # 1) Q-Learning or Minim/AlphaBeta => get the column
            col = get_ai_move(ai_mode_1, current_player=1)
            if col is not None:
                row = get_next_open_row(col)
                if row is not None:
                    # Before dropping, store (current_board, action) for Q-learning
                    if ai_mode_1 == "qlearning":
                        # We store a *copy* of the board plus the chosen move
                        game_trajectory_red.append(
                            (copy.deepcopy(board), col)
                        )

                    drop_piece(row, col, 1)
                    win_pos = find_winning_positions(1)
                    if win_pos:
                        set_winning_line(win_pos)
                        game_over = True
                        red_wins += 1
                        # Q-Learning final reward
                        if ai_mode_1 == "qlearning" or ai_mode_2 == "qlearning":
                            assign_qlearning_rewards(winner=1)
                    elif check_draw():
                        game_over = True
                        # Q-Learning final reward
                        if ai_mode_1 == "qlearning" or ai_mode_2 == "qlearning":
                            assign_qlearning_rewards(winner=0)
                    else:
                        player = 2

        # If it's Yellow's turn and Yellow is AI
        elif not game_over and ai_mode_2 != "none" and player == 2:
            col = get_ai_move(ai_mode_2, current_player=2)
            if col is not None:
                row = get_next_open_row(col)
                if row is not None:
                    if ai_mode_2 == "qlearning":
                        game_trajectory_yellow.append(
                            (copy.deepcopy(board), col)
                        )

                    drop_piece(row, col, 2)
                    win_pos = find_winning_positions(2)
                    if win_pos:
                        set_winning_line(win_pos)
                        game_over = True
                        yellow_wins += 1
                        # Q-Learning final reward
                        if ai_mode_1 == "qlearning" or ai_mode_2 == "qlearning":
                            assign_qlearning_rewards(winner=2)
                    elif check_draw():
                        game_over = True
                        # Q-Learning final reward
                        if ai_mode_1 == "qlearning" or ai_mode_2 == "qlearning":
                            assign_qlearning_rewards(winner=0)
                    else:
                        player = 1

        # Pygame events (for human moves, reset, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                # Check if "Reset" was clicked
                if reset_button_rect.collidepoint(mouseX, mouseY):
                    restart()
                    continue

                # Human's turn if game not over and player is "none" AI
                if not game_over and mouseY >= BOARD_OFFSET:
                    col = mouseX // SQUARE_SIZE
                    if col < BOARD_N:
                        # If it's Red's turn and Red is Human
                        if (player == 1 and ai_mode_1 == "none") and is_valid_column(col):
                            row = get_next_open_row(col)
                            drop_piece(row, col, 1)
                            win_pos = find_winning_positions(1)
                            if win_pos:
                                set_winning_line(win_pos)
                                game_over = True
                                red_wins += 1
                                # Q-Learning final reward if the *other* player was Q-learning
                                if ai_mode_2 == "qlearning":
                                    assign_qlearning_rewards(winner=1)
                            elif check_draw():
                                game_over = True
                                if ai_mode_2 == "qlearning":
                                    assign_qlearning_rewards(winner=0)
                            else:
                                player = 2

                        # If it's Yellow's turn and Yellow is Human
                        elif (player == 2 and ai_mode_2 == "none") and is_valid_column(col):
                            row = get_next_open_row(col)
                            drop_piece(row, col, 2)
                            win_pos = find_winning_positions(2)
                            if win_pos:
                                set_winning_line(win_pos)
                                game_over = True
                                yellow_wins += 1
                                if ai_mode_1 == "qlearning":
                                    assign_qlearning_rewards(winner=2)
                            elif check_draw():
                                game_over = True
                                if ai_mode_1 == "qlearning":
                                    assign_qlearning_rewards(winner=0)
                            else:
                                player = 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart()

        # RENDERING
        draw_header()
        draw_reset_button()
        draw_score()
        draw_pieces()

        # Animate winning line if game_over
        if game_over and winning_line is not None and anim_progress < 1.0:
            draw_winning_line()

        pygame.display.update()
