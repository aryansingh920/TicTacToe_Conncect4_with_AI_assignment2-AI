"""
Created on 30/03/2025

@author: Aryan

Filename: connect4.py
Relative Path: gui/connect4.py
"""

import sys
import pygame
import os

# Import your generated states manager
# Make sure the path and class name match where you placed Connect4StateManager.
from state.connect4 import Connect4StateManager

# If you stored your minimax code for Connect 4 in a separate file,
# import them as needed. Below we assume you have:
from algorithm.minmax_connect4 import minimax_connect4
from algorithm.minmax_alpha_beta_connect4 import minimax_alpha_beta_connect4

pygame.init()

# ----------------------------------------------------------------------
# GLOBAL CONFIG
# ----------------------------------------------------------------------
# If your board is NxN, set N here to match your Connect4StateManager.
# If using classic 6-rows x 7-cols Connect 4, adapt as needed.
BOARD_N = 4

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

# Create a single instance of Connect4StateManager
# Adjust file name to your pickled states file
c4_manager = None


def initialize_c4_manager():
    global c4_manager
    if c4_manager is None:
        c4_manager = Connect4StateManager(
            n=BOARD_N,
            cache_file='data/connect4/connect4_states_4x4.pkl'
        )


# ----------------------------------------------------------------------
# CONNECT 4 AI HELPER FUNCTIONS
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


def connect4_evaluate(bd, depth):
    """
    Enhanced evaluation for minimax/alpha-beta:
      - Terminal states: use a high magnitude score.
      - Non-terminal states: score based on potential connect opportunities.
      - Includes horizontal, vertical, and diagonal (3 and 4-length) evaluations.
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

    # Horizontal (length 3)
    for r in range(rows):
        for c in range(cols - 2):
            window = get_window(r, c, 0, 1, 3)
            score += score_window(window)

    # Vertical (length 3)
    for r in range(rows - 2):
        for c in range(cols):
            window = get_window(r, c, 1, 0, 3)
            score += score_window(window)

    # Diagonal \ (length 3)
    for r in range(rows - 2):
        for c in range(cols - 2):
            window = get_window(r, c, 1, 1, 3)
            score += score_window(window)

    # Diagonal / (length 3)
    for r in range(2, rows):
        for c in range(cols - 2):
            window = get_window(r, c, -1, 1, 3)
            score += score_window(window)

    return score


def connect4_get_children(state):
    """
    Generate all child states from the current state.
    Each child is (move, child_state).
    Where 'move' is the column index, and child_state is the next board+player.
    """
    bd = state["board"]
    current_player = state["current_player"]
    children = []

    # The manager's get_next_possible_moves returns a list of dict with
    # 'move', 'new_board', 'state_details'
    next_moves = c4_manager.get_next_possible_moves(bd)
    for move_info in next_moves:
        col = move_info['move']
        new_bd = move_info['new_board']
        next_player = 1 if current_player == 2 else 2

        child_state = {
            "board": new_bd,
            "current_player": next_player,
            # IMPORTANT: store the function reference, NOT a bool
            "is_terminal": connect4_is_terminal,
            "evaluate": connect4_evaluate,
            "get_children": connect4_get_children
        }

        children.append((col, child_state))
    return children


def get_ai_move(ai_mode, current_player):
    """
    Decide the best move for the CURRENT player, given ai_mode.
    Return a column index or None if no moves available.
    """
    # Build state dict for minimax-based approaches
    state = {
        "board": [row[:] for row in board],
        "current_player": current_player,
        "is_terminal": connect4_is_terminal,
        "evaluate": connect4_evaluate,
        "get_children": connect4_get_children
    }

    # If terminal or no moves => None
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

    # If ai_mode is something else or "none", return None
    return None

# ----------------------------------------------------------------------
# GAME LOGIC
# ----------------------------------------------------------------------


def is_valid_column(col):
    return board[0][col] == 0  # top cell is empty


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
    We can rely on c4_manager too, but we want the exact positions to animate the line.
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
    board = [[0]*BOARD_N for _ in range(BOARD_N)]
    game_over = False
    player = 1
    winning_line = None
    anim_progress = 0.0

# ----------------------------------------------------------------------
# ANIMATION
# ----------------------------------------------------------------------


def set_winning_line(win_positions):
    """
    Create an animation line from the first position to the last in the list.
    """
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
    # row=0 at top visually, row=BOARD_N-1 at bottom
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
# MAIN LOOP for Connect 4 (similar to TTT)
# ----------------------------------------------------------------------


def run_connect4(ai_mode_1="none", ai_mode_2="none"):
    """
    Connect 4 main loop:
      - ai_mode_1 => how Red (player=1) plays
      - ai_mode_2 => how Yellow (player=2) plays
      - "none" => human
      - "minimax" => minimax
      - "alpha-beta" => alpha-beta
    """
    initialize_c4_manager()
    global player, game_over, red_wins, yellow_wins

    restart()
    clock = pygame.time.Clock()

    while True:
        clock.tick(60)

        # If it's Red's turn and Red is AI
        if not game_over and ai_mode_1 != "none" and player == 1:
            col = get_ai_move(ai_mode_1, current_player=1)
            if col is not None:
                row = get_next_open_row(col)
                if row is not None:
                    drop_piece(row, col, 1)
                    win_pos = find_winning_positions(1)
                    if win_pos:
                        set_winning_line(win_pos)
                        game_over = True
                        red_wins += 1
                    elif check_draw():
                        game_over = True
                    else:
                        player = 2

        # If it's Yellow's turn and Yellow is AI
        elif not game_over and ai_mode_2 != "none" and player == 2:
            col = get_ai_move(ai_mode_2, current_player=2)
            if col is not None:
                row = get_next_open_row(col)
                if row is not None:
                    drop_piece(row, col, 2)
                    win_pos = find_winning_positions(2)
                    if win_pos:
                        set_winning_line(win_pos)
                        game_over = True
                        yellow_wins += 1
                    elif check_draw():
                        game_over = True
                    else:
                        player = 1

        # Pygame events (for human moves, reset, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                # Check reset
                if reset_button_rect.collidepoint(mouseX, mouseY):
                    restart()
                    continue
                # Human's turn if game not over
                if not game_over and mouseY >= BOARD_OFFSET:
                    col = mouseX // SQUARE_SIZE
                    if col < BOARD_N:
                        if (player == 1 and ai_mode_1 == "none") and is_valid_column(col):
                            row = get_next_open_row(col)
                            drop_piece(row, col, 1)
                            win_pos = find_winning_positions(1)
                            if win_pos:
                                set_winning_line(win_pos)
                                game_over = True
                                red_wins += 1
                            elif check_draw():
                                game_over = True
                            else:
                                player = 2
                        elif (player == 2 and ai_mode_2 == "none") and is_valid_column(col):
                            row = get_next_open_row(col)
                            drop_piece(row, col, 2)
                            win_pos = find_winning_positions(2)
                            if win_pos:
                                set_winning_line(win_pos)
                                game_over = True
                                yellow_wins += 1
                            elif check_draw():
                                game_over = True
                            else:
                                player = 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart()

        # RENDER
        draw_header()
        draw_reset_button()
        draw_score()
        draw_pieces()

        # Animate winning line if game_over
        if game_over and winning_line is not None and anim_progress < 1.0:
            draw_winning_line()

        pygame.display.update()
