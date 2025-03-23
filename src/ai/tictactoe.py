"""
Created on 23/03/2025

@author: Aryan

Filename: tictactoe.py

Relative Path: ai/tictactoe.py
"""

"""
Created on 23/03/2025

@author: Aryan

Filename: tictactoe.py
Relative Path: gui/tictactoe.py

Updated to allow playing with AI (minimax, alpha-beta, or qlearning).
"""

import sys
import pygame
from algorithm.minmax import minimax_tictactoe
from algorithm.qLearning import QLearningAgent
from algorithm.minmax_alpha_beta import minimax_alpha_beta_tictactoe

# Import your AI modules

pygame.init()

# -----------------------------
# CONFIG AND CONSTANTS
# -----------------------------
WIDTH, HEIGHT = 600, 650
BOARD_OFFSET = 50
BOARD_ROWS, BOARD_COLS = 3, 3
LINE_WIDTH = 10
SQUARE_SIZE = 600 // BOARD_ROWS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
TEXT_COLOR = (255, 255, 255)
BUTTON_BG_COLOR = (40, 100, 90)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
font = pygame.font.SysFont(None, 40)

BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
reset_button_rect = pygame.Rect(
    WIDTH - (BUTTON_WIDTH + 20), 5, BUTTON_WIDTH, BUTTON_HEIGHT)

board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
player = 1  # Start with X
game_over = False
x_wins = 0
o_wins = 0

winning_line = None
anim_progress = 0.0

# For Q-learning demonstration
q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

# -----------------------------
# DRAWING FUNCTIONS (unchanged)
# -----------------------------


def draw_board():
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, (10, 130, 120), (0, 0, WIDTH, BOARD_OFFSET))
    # Lines
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
                pygame.draw.line(screen, CROSS_COLOR,
                                 (center_x - SPACE, center_y - SPACE),
                                 (center_x + SPACE, center_y + SPACE),
                                 CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR,
                                 (center_x - SPACE, center_y + SPACE),
                                 (center_x + SPACE, center_y - SPACE),
                                 CROSS_WIDTH)
            elif board[row][col] == 2:
                pygame.draw.circle(screen, CIRCLE_COLOR,
                                   (center_x, center_y),
                                   CIRCLE_RADIUS, CIRCLE_WIDTH)


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
# LOGIC FUNCTIONS (slightly updated)
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
    # Anti-diag
    if all([board[i][BOARD_COLS - 1 - i] == player_id for i in range(BOARD_ROWS)]):
        return ('diag', 1)
    return None


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
    global winning_line
    if not win_info:
        return
    typ, idx = win_info
    if typ == 'row':
        start_x = 0
        start_y = BOARD_OFFSET + idx * SQUARE_SIZE + SQUARE_SIZE // 2
        end_x = WIDTH
        end_y = start_y
        winning_line = (start_x, start_y, end_x, end_y)
    elif typ == 'col':
        start_x = idx * SQUARE_SIZE + SQUARE_SIZE // 2
        start_y = BOARD_OFFSET
        end_x = start_x
        end_y = BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE
        winning_line = (start_x, start_y, end_x, end_y)
    elif typ == 'diag':
        if idx == 0:
            start_x, start_y = (0, BOARD_OFFSET)
            end_x = BOARD_COLS * SQUARE_SIZE
            end_y = BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE
        else:
            start_x = BOARD_COLS * SQUARE_SIZE
            start_y = BOARD_OFFSET
            end_x = 0
            end_y = BOARD_OFFSET + BOARD_ROWS * SQUARE_SIZE
        winning_line = (start_x, start_y, end_x, end_y)

# -----------------------------
# AI HOOKS - HELPER FUNCTIONS
# -----------------------------


def tictactoe_is_terminal(bd):
    # Return True if we have a winner or draw
    if check_win(1) or check_win(2):
        return True
    if check_draw():
        return True
    return False


def tictactoe_evaluate(bd, depth):
    # A simplistic scoring approach
    # If X wins, return positive; if O wins, return negative
    # Depth is used so that faster wins get a bigger score.
    # Example scoring:
    #    +10 - depth if X wins
    #    -10 + depth if O wins
    #     0 if draw
    if check_win(1):
        return 10 - depth
    elif check_win(2):
        return depth - 10
    else:
        return 0


def tictactoe_get_children(state):
    """
    Generate all possible next states for the current player.
    Return list of (move, child_state).
    'move' can be (row, col).
    """
    bd = state["board"]
    children = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if bd[r][c] == 0:
                # Copy board
                new_board = [row[:] for row in bd]
                new_board[r][c] = state["current_player"]
                # Next player
                next_player = 1 if state["current_player"] == 2 else 2
                child_state = {
                    "board": new_board,
                    "current_player": next_player,
                    "is_terminal": tictactoe_is_terminal,
                    "evaluate": tictactoe_evaluate,
                    "get_children": tictactoe_get_children
                }
                children.append(((r, c), child_state))
    return children


def get_ai_move(ai_mode):
    """
    Decide the best move for the current player given ai_mode.
    Return (row, col).
    """
    # Build state dictionary for minimax
    state = {
        "board": [row[:] for row in board],
        "current_player": player,
        "is_terminal": tictactoe_is_terminal,
        "evaluate": tictactoe_evaluate,
        "get_children": tictactoe_get_children
    }

    if ai_mode == "minimax":
        # Let X=1 be the maximizing player
        score, move = minimax_tictactoe(state, maximizing_player=1)
        return move
    elif ai_mode == "alpha-beta":
        score, move = minimax_alpha_beta_tictactoe(state, maximizing_player=1)
        return move
    elif ai_mode == "qlearning":
        # We do a single-step Q lookup.
        valid_moves = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if board[r][c] == 0:
                    valid_moves.append((r, c))
        if not valid_moves:
            return None
        return q_agent.choose_action(board, valid_moves)
    else:
        return None  # "none" means no AI


# -----------------------------
# MAIN LOOP WRAPPED IN A FUNCTION
# -----------------------------
def run_tictactoe(ai_mode="none"):
    """
    Run the TicTacToe game loop. If ai_mode != 'none',
    then player 2 is replaced with an AI.
    """
    global player, game_over, x_wins, o_wins

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        # If it's the AI's turn and the game isn't over
        if not game_over and ai_mode != "none" and player == 2:
            move = get_ai_move(ai_mode)
            if move is not None:
                r, c = move
                if is_available(r, c):
                    mark_square(r, c, player)
                    win_res = check_win(player)
                    if win_res:
                        set_winning_line(win_res)
                        game_over = True
                        if player == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    elif check_draw():
                        game_over = True
                    else:
                        player = 1 if player == 2 else 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                if reset_button_rect.collidepoint(mouseX, mouseY):
                    restart()
                elif not game_over and player == 1:
                    # Human is X
                    if mouseY >= BOARD_OFFSET:
                        row = (mouseY - BOARD_OFFSET) // SQUARE_SIZE
                        col = mouseX // SQUARE_SIZE
                        if row < BOARD_ROWS and col < BOARD_COLS and is_available(row, col):
                            mark_square(row, col, player)
                            win_res = check_win(player)
                            if win_res:
                                set_winning_line(win_res)
                                game_over = True
                                x_wins += 1
                            elif check_draw():
                                game_over = True
                            else:
                                player = 2 if player == 1 else 1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart()

        # Draw
        draw_board()
        draw_score()
        draw_reset_button()
        draw_figures()

        if game_over and winning_line is not None and anim_progress < 1.0:
            draw_winning_line()

        pygame.display.update()


# If you still want to run it directly:
if __name__ == "__main__":
    run_tictactoe(ai_mode="none")
