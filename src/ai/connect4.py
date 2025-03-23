"""
Created on 23/03/2025

@author: Aryan

Filename: connect4.py
Relative Path: gui/connect4.py

Updated to allow AI play (minimax, alpha-beta, or qlearning).
"""

import pygame
import sys

# We'll rename or create a specialized minimax for Connect4
from algorithm.minmax import minimax_tictactoe
from algorithm.minmax_alpha_beta import minimax_alpha_beta_tictactoe
from algorithm.qLearning import QLearningAgent

pygame.init()

# -----------------------------
# CONFIG AND CONSTANTS
# -----------------------------
BOARD_COLS = 7
BOARD_ROWS = 6
SQUARE_SIZE = 100
BOARD_OFFSET = 60
WIDTH = BOARD_COLS * SQUARE_SIZE
HEIGHT = BOARD_OFFSET + (BOARD_ROWS * SQUARE_SIZE)

BG_COLOR = (28, 170, 156)
BOARD_COLOR = (10, 130, 120)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
EMPTY_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
BUTTON_BG_COLOR = (40, 100, 90)

BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
reset_button_rect = pygame.Rect(WIDTH - (BUTTON_WIDTH + 20), 10,
                                BUTTON_WIDTH, BUTTON_HEIGHT)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect 4")
font = pygame.font.SysFont(None, 40)

board = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
player = 1
game_over = False
red_wins = 0
yellow_wins = 0

winning_line = None
anim_progress = 0.0

q_agent_connect4 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

# -----------------------------
# HELPER / LOGIC (original + minor changes)
# -----------------------------


def is_valid_location(col):
    return board[0][col] == 0


def get_next_open_row(col):
    for r in range(BOARD_ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None


def drop_piece(row, col, piece):
    board[row][col] = piece


def find_winning_positions(piece):
    # Horizontal
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if (board[r][c] == piece and
                board[r][c+1] == piece and
                board[r][c+2] == piece and
                    board[r][c+3] == piece):
                return [(r, c), (r, c+1), (r, c+2), (r, c+3)]
    # Vertical
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            if (board[r][c] == piece and
                board[r+1][c] == piece and
                board[r+2][c] == piece and
                    board[r+3][c] == piece):
                return [(r, c), (r+1, c), (r+2, c), (r+3, c)]
    # Positive diagonal
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            if (board[r][c] == piece and
                board[r+1][c+1] == piece and
                board[r+2][c+2] == piece and
                    board[r+3][c+3] == piece):
                return [(r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3)]
    # Negative diagonal
    for r in range(3, BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if (board[r][c] == piece and
                board[r-1][c+1] == piece and
                board[r-2][c+2] == piece and
                    board[r-3][c+3] == piece):
                return [(r, c), (r-1, c+1), (r-2, c+2), (r-3, c+3)]
    return None


def check_draw():
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == 0:
                return False
    return True


def reset_board():
    global board, game_over, player, winning_line, anim_progress
    board = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
    game_over = False
    player = 1
    winning_line = None
    anim_progress = 0.0

# For animation


def set_winning_line(positions):
    global winning_line, anim_progress
    start_r, start_c = positions[0]
    end_r, end_c = positions[-1]
    start_x = start_c * SQUARE_SIZE + (SQUARE_SIZE // 2)
    start_y = BOARD_OFFSET + start_r * SQUARE_SIZE + (SQUARE_SIZE // 2)
    end_x = end_c * SQUARE_SIZE + (SQUARE_SIZE // 2)
    end_y = BOARD_OFFSET + end_r * SQUARE_SIZE + (SQUARE_SIZE // 2)
    winning_line = (start_x, start_y, end_x, end_y)
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

# -----------------------------
# DRAWING FUNCTIONS (same as original)
# -----------------------------


def draw_board():
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, BOARD_COLOR, (0, 0, WIDTH, BOARD_OFFSET))


def draw_reset_button():
    pygame.draw.rect(screen, BUTTON_BG_COLOR, reset_button_rect)
    button_text = font.render("Reset", True, TEXT_COLOR)
    text_rect = button_text.get_rect(center=reset_button_rect.center)
    screen.blit(button_text, text_rect)


def draw_score():
    score_text = f"Red - {red_wins}   |   Yellow - {yellow_wins}"
    text_surface = font.render(score_text, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 10))


def draw_pieces():
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            center_x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = BOARD_OFFSET + r * SQUARE_SIZE + SQUARE_SIZE // 2
            color = EMPTY_COLOR
            if board[r][c] == 1:
                color = RED
            elif board[r][c] == 2:
                color = YELLOW
            pygame.draw.circle(
                screen, color, (center_x, center_y), SQUARE_SIZE//2 - 5)

# -----------------------------
# AI HOOKS FOR CONNECT4 (DEMO ONLY)
# -----------------------------


def connect4_is_terminal(bd):
    if find_winning_positions(1) or find_winning_positions(2):
        return True
    # check draw
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if bd[r][c] == 0:
                return False
    return True


def connect4_evaluate(bd, depth):
    # Very naive. +10 if Red wins, -10 if Yellow wins, 0 otherwise.
    # For a more sophisticated approach, you'd count potential 3 in a row, etc.
    if find_winning_positions(1):
        return 10 - depth
    elif find_winning_positions(2):
        return depth - 10
    return 0


def connect4_get_children(state):
    from copy import deepcopy
    children = []
    bd = state["board"]
    for c in range(BOARD_COLS):
        if bd[0][c] == 0:
            # valid
            new_bd = deepcopy(bd)
            # find open row
            for r in range(BOARD_ROWS-1, -1, -1):
                if new_bd[r][c] == 0:
                    new_bd[r][c] = state["current_player"]
                    break
            next_player = 2 if state["current_player"] == 1 else 1
            child_state = {
                "board": new_bd,
                "current_player": next_player,
                "is_terminal": connect4_is_terminal,
                "evaluate": connect4_evaluate,
                "get_children": connect4_get_children
            }
            children.append(((r, c), child_state))
    return children


def get_ai_move_connect4(ai_mode):
    # Build a state
    st = {
        "board": [row[:] for row in board],
        "current_player": player,
        "is_terminal": connect4_is_terminal,
        "evaluate": connect4_evaluate,
        "get_children": connect4_get_children
    }
    if ai_mode == "minimax":
        score, move = minimax_tictactoe(st, maximizing_player=1)
        return move  # (row, col)
    elif ai_mode == "alpha-beta":
        score, move = minimax_alpha_beta_tictactoe(st, maximizing_player=1)
        return move
    elif ai_mode == "qlearning":
        valid_actions = []
        for col in range(BOARD_COLS):
            if is_valid_location(col):
                # row is the next open row
                valid_actions.append((get_next_open_row(col), col))
        if not valid_actions:
            return None
        return q_agent_connect4.choose_action(board, valid_actions)
    else:
        return None

# -----------------------------
# MAIN LOOP WRAPPED
# -----------------------------


def run_connect4(ai_mode="none"):
    global player, game_over, red_wins, yellow_wins

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        # If it's AI's turn and not game over
        if not game_over and ai_mode != "none" and player == 2:
            move = get_ai_move_connect4(ai_mode)
            if move is not None:
                row, col = move
                if row is not None and col is not None and is_valid_location(col):
                    drop_piece(row, col, player)
                    win_pos = find_winning_positions(player)
                    if win_pos:
                        set_winning_line(win_pos)
                        game_over = True
                        if player == 1:
                            red_wins += 1
                        else:
                            yellow_wins += 1
                    elif check_draw():
                        game_over = True
                    else:
                        player = 1 if player == 2 else 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if reset_button_rect.collidepoint(mouse_x, mouse_y):
                    reset_board()
                    continue
                if not game_over and player == 1:  # Player 1 is human
                    if mouse_y >= BOARD_OFFSET:
                        col = mouse_x // SQUARE_SIZE
                        if col < BOARD_COLS and is_valid_location(col):
                            row = get_next_open_row(col)
                            drop_piece(row, col, player)
                            win_pos = find_winning_positions(player)
                            if win_pos:
                                set_winning_line(win_pos)
                                game_over = True
                                red_wins += 1
                            elif check_draw():
                                game_over = True
                            else:
                                player = 2 if player == 1 else 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_board()

        # RENDER
        draw_board()
        draw_reset_button()
        draw_score()
        draw_pieces()

        if game_over and winning_line is not None and anim_progress < 1.0:
            draw_winning_line()

        pygame.display.update()


# If you want direct execution:
if __name__ == "__main__":
    run_connect4(ai_mode="none")
