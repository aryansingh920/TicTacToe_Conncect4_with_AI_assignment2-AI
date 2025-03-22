import pygame
import sys

pygame.init()

# -----------------------------
# CONFIG AND CONSTANTS
# -----------------------------
BOARD_COLS = 7
BOARD_ROWS = 6

# Layout
SQUARE_SIZE = 100
BOARD_OFFSET = 60  # space at top for scoreboard
WIDTH = BOARD_COLS * SQUARE_SIZE
HEIGHT = BOARD_OFFSET + (BOARD_ROWS * SQUARE_SIZE)

# Colors
BG_COLOR = (28, 170, 156)      # background
BOARD_COLOR = (10, 130, 120)   # scoreboard background
LINE_COLOR = (23, 145, 135)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
EMPTY_COLOR = (30, 30, 30)     # empty cells
TEXT_COLOR = (255, 255, 255)
BUTTON_BG_COLOR = (40, 100, 90)

# Button (for "Reset")
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
reset_button_rect = pygame.Rect(WIDTH - (BUTTON_WIDTH + 20), 10,
                                BUTTON_WIDTH, BUTTON_HEIGHT)

# Window setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect 4")
font = pygame.font.SysFont(None, 40)

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
# Board: 0 = empty, 1 = Red, 2 = Yellow
board = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]

player = 1            # 1 = Red's turn, 2 = Yellow's turn
game_over = False
red_wins = 0
yellow_wins = 0

# -----------------------------
# HELPER / LOGIC FUNCTIONS
# -----------------------------


def is_valid_location(col):
    """
    A column is valid if the topmost cell (row 0) is empty (0).
    This means there's room to place a piece in that column.
    """
    return board[0][col] == 0


def get_next_open_row(col):
    """
    Returns the lowest available row in the given column.
    That is, starting from the bottom row, which is row = BOARD_ROWS - 1.
    """
    for r in range(BOARD_ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None  # Shouldn't happen if we check is_valid_location first


def drop_piece(row, col, piece):
    board[row][col] = piece


def check_win(piece):
    """Return True if 'piece' (1 or 2) has four in a row."""
    # 1) Check horizontal
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if (board[r][c] == piece and
                board[r][c+1] == piece and
                board[r][c+2] == piece and
                    board[r][c+3] == piece):
                return True

    # 2) Check vertical
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            if (board[r][c] == piece and
                board[r+1][c] == piece and
                board[r+2][c] == piece and
                    board[r+3][c] == piece):
                return True

    # 3) Check positively sloped diagonals (\)
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            if (board[r][c] == piece and
                board[r+1][c+1] == piece and
                board[r+2][c+2] == piece and
                    board[r+3][c+3] == piece):
                return True

    # 4) Check negatively sloped diagonals (/)
    for r in range(3, BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if (board[r][c] == piece and
                board[r-1][c+1] == piece and
                board[r-2][c+2] == piece and
                    board[r-3][c+3] == piece):
                return True

    return False


def check_draw():
    """Returns True if the board is full (no zeros)."""
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == 0:
                return False
    return True


def reset_board():
    global board, game_over, player
    board = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
    game_over = False
    player = 1

# -----------------------------
# DRAWING FUNCTIONS
# -----------------------------


def draw_board():
    """Draw background + scoreboard region."""
    screen.fill(BG_COLOR)
    # Scoreboard area
    pygame.draw.rect(screen, BOARD_COLOR, (0, 0, WIDTH, BOARD_OFFSET))


def draw_reset_button():
    """Draws the 'Reset' button at the top-right area."""
    pygame.draw.rect(screen, BUTTON_BG_COLOR, reset_button_rect)
    button_text = font.render("Reset", True, TEXT_COLOR)
    text_rect = button_text.get_rect(center=reset_button_rect.center)
    screen.blit(button_text, text_rect)


def draw_score():
    """Displays Red and Yellow win counts."""
    score_text = f"Red - {red_wins}   |   Yellow - {yellow_wins}"
    text_surface = font.render(score_text, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 10))


def draw_pieces():
    """
    Draw the Connect 4 grid of circles:
    - row 0 is near top of screen, but we'll visually place row 0 at the top
      and row BOARD_ROWS-1 at the bottom for a typical Connect 4 look.
    """
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            # 'x' coordinate depends on c
            # 'y' coordinate depends on r, plus offset for scoreboard
            center_x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = BOARD_OFFSET + r * SQUARE_SIZE + SQUARE_SIZE // 2

            color = EMPTY_COLOR
            if board[r][c] == 1:
                color = RED
            elif board[r][c] == 2:
                color = YELLOW

            pygame.draw.circle(screen, color,
                               (center_x, center_y),
                               SQUARE_SIZE//2 - 5)

# -----------------------------
# MAIN LOOP
# -----------------------------


def main():
    global player, game_over, red_wins, yellow_wins

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouse_x, mouse_y = event.pos

                # Check if clicked "Reset" button
                if reset_button_rect.collidepoint(mouse_x, mouse_y):
                    reset_board()
                    continue  # Skip further board logic

                # Check if we clicked on the board area
                if mouse_y >= BOARD_OFFSET:
                    col = mouse_x // SQUARE_SIZE
                    if col < BOARD_COLS and is_valid_location(col):
                        row = get_next_open_row(col)
                        drop_piece(row, col, player)

                        # Check for winner
                        if check_win(player):
                            game_over = True
                            if player == 1:
                                red_wins += 1
                            else:
                                yellow_wins += 1

                        elif check_draw():
                            game_over = True

                        else:
                            # Switch player
                            player = 2 if player == 1 else 1

            # Keyboard: R to reset
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_board()

        # DRAW
        draw_board()
        draw_reset_button()
        draw_score()
        draw_pieces()

        pygame.display.update()


if __name__ == "__main__":
    main()
