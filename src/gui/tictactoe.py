"""
Created on 23/03/2025

@author: Aryan

Filename: tictactoe.py

Relative Path: gui/tictactoe.py
"""

import pygame
import sys

pygame.init()

# -----------------------------
# CONFIG AND CONSTANTS
# -----------------------------
# Window dimensions
WIDTH, HEIGHT = 600, 650  # Extra space at bottom/top for score
BOARD_OFFSET = 50         # Space on top for scoreboard

BOARD_ROWS, BOARD_COLS = 3, 3
LINE_WIDTH = 10
SQUARE_SIZE = 600 // BOARD_ROWS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
TEXT_COLOR = (255, 255, 255)
BUTTON_BG_COLOR = (40, 100, 90)

# Pygame setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
font = pygame.font.SysFont(None, 40)

# Button rect (for "Reset")
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
reset_button_rect = pygame.Rect(
    WIDTH - (BUTTON_WIDTH + 20), 5, BUTTON_WIDTH, BUTTON_HEIGHT)

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
# Board representation: 0=empty, 1=X, 2=O
board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

player = 1  # Start with X
game_over = False
x_wins = 0
o_wins = 0

# Animation data for winning line
winning_line = None  # (start_x, start_y, end_x, end_y)
anim_progress = 0.0  # From 0.0 to 1.0

# -----------------------------
# DRAWING FUNCTIONS
# -----------------------------


def draw_board():
    screen.fill(BG_COLOR)
    # Draw scoreboard background
    pygame.draw.rect(screen, (10, 130, 120), (0, 0, WIDTH, BOARD_OFFSET))

    # Draw lines (horizontal and vertical)
    for i in range(1, BOARD_ROWS):
        start_pos = (0, BOARD_OFFSET + i * SQUARE_SIZE)
        end_pos = (WIDTH, BOARD_OFFSET + i * SQUARE_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, LINE_WIDTH)
    for i in range(1, BOARD_COLS):
        start_pos = (i * SQUARE_SIZE, BOARD_OFFSET)
        end_pos = (i * SQUARE_SIZE, BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, LINE_WIDTH)


def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = BOARD_OFFSET + row * SQUARE_SIZE + SQUARE_SIZE // 2

            if board[row][col] == 1:
                # Draw Cross
                pygame.draw.line(screen, CROSS_COLOR,
                                 (center_x - SPACE, center_y - SPACE),
                                 (center_x + SPACE, center_y + SPACE),
                                 CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR,
                                 (center_x - SPACE, center_y + SPACE),
                                 (center_x + SPACE, center_y - SPACE),
                                 CROSS_WIDTH)
            elif board[row][col] == 2:
                # Draw Circle
                pygame.draw.circle(screen, CIRCLE_COLOR,
                                   (center_x, center_y),
                                   CIRCLE_RADIUS, CIRCLE_WIDTH)


def draw_score():
    # "X - {x_wins}  |  O - {o_wins}"
    score_text = f"X - {x_wins}   |   O - {o_wins}"
    text_surface = font.render(score_text, True, TEXT_COLOR)
    # Center it in scoreboard area
    text_rect = text_surface.get_rect(midleft=(10, BOARD_OFFSET // 2))
    screen.blit(text_surface, text_rect)


def draw_reset_button():
    """
    Draws the "Reset" button on the scoreboard area.
    """
    pygame.draw.rect(screen, BUTTON_BG_COLOR, reset_button_rect)
    button_text = font.render("Reset", True, TEXT_COLOR)
    button_text_rect = button_text.get_rect(center=reset_button_rect.center)
    screen.blit(button_text, button_text_rect)


def draw_winning_line():
    """
    Draws an animated winning line from start to end using anim_progress [0..1].
    """
    global anim_progress
    if winning_line is None:
        return

    start_x, start_y, end_x, end_y = winning_line
    # Calculate current point based on progress
    current_x = start_x + (end_x - start_x) * anim_progress
    current_y = start_y + (end_y - start_y) * anim_progress

    pygame.draw.line(screen, (200, 50, 50), (start_x, start_y),
                     (current_x, current_y), 10)
    # Increment progress
    anim_progress += 0.02
    if anim_progress >= 1.0:
        anim_progress = 1.0

# -----------------------------
# GAME/BOARD FUNCTIONS
# -----------------------------


def mark_square(row, col, player_id):
    board[row][col] = player_id


def is_available(row, col):
    return board[row][col] == 0


def check_win(player_id):
    # Rows
    for r in range(BOARD_ROWS):
        if all([board[r][c] == player_id for c in range(BOARD_COLS)]):
            return ('row', r)

    # Cols
    for c in range(BOARD_COLS):
        if all([board[r][c] == player_id for r in range(BOARD_ROWS)]):
            return ('col', c)

    # Diagonal
    if all([board[i][i] == player_id for i in range(BOARD_ROWS)]):
        return ('diag', 0)

    # Anti-diagonal
    if all([board[i][BOARD_COLS - 1 - i] == player_id for i in range(BOARD_ROWS)]):
        return ('diag', 1)

    return None  # No win


def check_draw():
    return all([cell != 0 for row in board for cell in row])


def restart():
    global board, game_over, player, winning_line, anim_progress
    board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    game_over = False
    player = 1
    winning_line = None
    anim_progress = 0.0


def set_winning_line(win_info):
    """
    Sets the global 'winning_line' (start_x, start_y, end_x, end_y)
    based on the winning row/col/diag.
    """
    global winning_line
    if not win_info:
        return

    typ, idx = win_info
    # Each cell is SQUARE_SIZE high, offset by BOARD_OFFSET vertically
    if typ == 'row':
        # Horizontal line across row idx
        start_x = 0
        start_y = BOARD_OFFSET + idx * SQUARE_SIZE + SQUARE_SIZE // 2
        end_x = WIDTH
        end_y = start_y
        winning_line = (start_x, start_y, end_x, end_y)
    elif typ == 'col':
        # Vertical line across column idx
        start_x = idx * SQUARE_SIZE + SQUARE_SIZE // 2
        start_y = BOARD_OFFSET
        end_x = start_x
        end_y = BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE
        winning_line = (start_x, start_y, end_x, end_y)
    elif typ == 'diag':
        # Diagonal
        if idx == 0:
            # Main diagonal
            start_x, start_y = (0, BOARD_OFFSET)
            end_x = BOARD_COLS * SQUARE_SIZE
            end_y = BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE
        else:
            # Anti-diagonal
            start_x = BOARD_COLS * SQUARE_SIZE
            start_y = BOARD_OFFSET
            end_x = 0
            end_y = BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE
        winning_line = (start_x, start_y, end_x, end_y)

# -----------------------------
# MAIN LOOP
# -----------------------------


def main():
    global player, game_over, x_wins, o_wins

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)  # 60 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle mouse events
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos

                # Check if clicked Reset button
                if reset_button_rect.collidepoint(mouseX, mouseY):
                    restart()

                # If not game_over, process board clicks
                elif not game_over and mouseY >= BOARD_OFFSET:
                    row = (mouseY - BOARD_OFFSET) // SQUARE_SIZE
                    col = mouseX // SQUARE_SIZE

                    if row < BOARD_ROWS and col < BOARD_COLS and is_available(row, col):
                        mark_square(row, col, player)
                        win_res = check_win(player)

                        if win_res:  # We have a winner
                            set_winning_line(win_res)
                            game_over = True
                            if player == 1:
                                x_wins += 1
                            else:
                                o_wins += 1
                        elif check_draw():
                            game_over = True
                        else:
                            player = 2 if player == 1 else 1

            # Handle keyboard for quick reset
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart()

        # Render everything
        draw_board()
        draw_score()
        draw_reset_button()
        draw_figures()

        # Animate winning line if exists
        if game_over and winning_line is not None and anim_progress < 1.0:
            draw_winning_line()

        pygame.display.update()


if __name__ == "__main__":
    main()
