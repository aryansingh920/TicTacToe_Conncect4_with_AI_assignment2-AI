"""
Created on 30/03/2025

@author: Aryan

Headless Connect 4 self-play training script for Q-Learning.
Board size: 4x4
Single Q-learning agent that alternates between player 1 (Red=1) and player 2 (Yellow=2),
with a transform step for Yellow's perspective.
"""

import random
import collections
import copy
import sys

# ----------------------------------------------------------------------
#  Import the QLearningAgent from your "algorithm.qLearning" module
#  which should have the same interface you used for TicTacToe.
# ----------------------------------------------------------------------
from algorithm.qLearning import QLearningAgent


# ----------------------------------------------------------------------
#  GAME CONSTANTS & HELPERS
# ----------------------------------------------------------------------
ROWS = 4
COLS = 4


def get_valid_moves(board):
    """
    Return a list of valid 'moves' for the current board.
    In Connect 4, a 'move' is the column index where a piece can drop.
    """
    valid_moves = []
    for col in range(COLS):
        # If the top cell is not occupied, that column is valid
        if board[0][col] == 0:
            valid_moves.append(col)
    return valid_moves


def get_next_open_row(board, col):
    """
    For the given column, find the lowest empty row in that column.
    Return that row index or None if column is full (should not happen if validated).
    """
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None


def drop_piece(board, row, col, player):
    """
    Place the player's piece in the board at (row, col).
    player is 1 or 2.
    """
    board[row][col] = player


def check_win(player, board):
    """
    Check if 'player' (1 or 2) has a Connect-4 in the 4x4 board.
    Returns True if found, else False.
    """
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if (board[r][c] == player and
                board[r][c+1] == player and
                board[r][c+2] == player and
                    board[r][c+3] == player):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if (board[r][c] == player and
                board[r+1][c] == player and
                board[r+2][c] == player and
                    board[r+3][c] == player):
                return True

    # Diagonal (down-right)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if (board[r][c] == player and
                board[r+1][c+1] == player and
                board[r+2][c+2] == player and
                    board[r+3][c+3] == player):
                return True

    # Diagonal (up-right)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if (board[r][c] == player and
                board[r-1][c+1] == player and
                board[r-2][c+2] == player and
                    board[r-3][c+3] == player):
                return True

    return False


def check_draw(board):
    """
    Return True if board is completely full (no zeros), else False.
    """
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 0:
                return False
    return True


def transform_board_for_player2(board):
    """
    Transform the board so that from Player 2's perspective,
    it looks like Player 1. 
    We swap all 1s and 2s:
      1 -> 2
      2 -> 1
    This is so we can use the same Q-values for both players in one Q-table.
    """
    transformed = copy.deepcopy(board)
    for r in range(ROWS):
        for c in range(COLS):
            if transformed[r][c] == 1:
                transformed[r][c] = 2
            elif transformed[r][c] == 2:
                transformed[r][c] = 1
    return transformed


# ----------------------------------------------------------------------
#  PLAY ONE SELF-PLAY GAME: Q vs. Q (single Q-table)
# ----------------------------------------------------------------------
def play_game_q_vs_q(agent, transform_state_for_O=True):
    """
    One self-play game on a 4x4 board:
      - Player 1 uses direct board
      - Player 2 optionally transforms the board so it "looks like P1" 
        (that's what transform_state_for_O means)
    Return (winner, final_board)
      winner=0 => draw
      winner=1 => Player 1 wins
      winner=2 => Player 2 wins
    """
    board = [[0]*COLS for _ in range(ROWS)]
    player = 1  # Start with Player 1
    q_game_traj_p1 = []
    q_game_traj_p2 = []

    while True:
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            # No moves => it's a draw
            # Give final reward 0 to both trajectories
            agent.batch_update_from_game(q_game_traj_p1, final_reward=0)
            agent.batch_update_from_game(q_game_traj_p2, final_reward=0)
            return (0, board)

        # If player=1, we do direct board
        if player == 1:
            # Choose an action from Q
            action_col = agent.choose_action(board, valid_moves)
            # Record trajectory
            board_copy = copy.deepcopy(board)
            q_game_traj_p1.append((board_copy, action_col))

            # Execute the move
            row = get_next_open_row(board, action_col)
            drop_piece(board, row, action_col, 1)

            # Check win/draw
            if check_win(1, board):
                # Player 1 wins => +1 for P1, -1 for P2
                agent.batch_update_from_game(q_game_traj_p1, final_reward=+1)
                agent.batch_update_from_game(q_game_traj_p2, final_reward=-1)
                return (1, board)
            if check_draw(board):
                agent.batch_update_from_game(q_game_traj_p1, final_reward=0)
                agent.batch_update_from_game(q_game_traj_p2, final_reward=0)
                return (0, board)

            player = 2

        else:
            # Player=2
            if transform_state_for_O:
                transformed_board = transform_board_for_player2(board)
                action_col = agent.choose_action(
                    transformed_board, valid_moves)
            else:
                # If not transforming, you'd need to modify your QLearningAgent
                # to accept a "player" argument or do separate Q-tables.
                # We'll assume transform approach here.
                action_col = agent.choose_action(board, valid_moves)

            # Record trajectory
            board_copy = copy.deepcopy(board)
            q_game_traj_p2.append((board_copy, action_col))

            # Execute the move
            row = get_next_open_row(board, action_col)
            drop_piece(board, row, action_col, 2)

            # Check win/draw
            if check_win(2, board):
                agent.batch_update_from_game(q_game_traj_p2, final_reward=+1)
                agent.batch_update_from_game(q_game_traj_p1, final_reward=-1)
                return (2, board)
            if check_draw(board):
                agent.batch_update_from_game(q_game_traj_p1, final_reward=0)
                agent.batch_update_from_game(q_game_traj_p2, final_reward=0)
                return (0, board)

            player = 1


# ----------------------------------------------------------------------
#  MAIN TRAINING LOOP
# ----------------------------------------------------------------------
def main(episode_count=50000):
    """
    Train the Q-learning agent by self-play on a 4x4 Connect Four board.
    Defaults to 50,000 episodes. Adjust as needed for your environment.
    """
    # Create a single Q agent for both players
    agent = QLearningAgent(
        alpha=0.1,         # Learning rate
        gamma=0.9,         # Discount factor
        epsilon=0.2,       # Exploration rate
        save_file='data/connect4/qtable_single.pkl'  # Single Q-table for both players
    )

    wins_for_p1 = 0
    wins_for_p2 = 0
    draws = 0

    for episode in range(episode_count):
        winner, final_board = play_game_q_vs_q(
            agent, transform_state_for_O=True)
        if winner == 1:
            wins_for_p1 += 1
        elif winner == 2:
            wins_for_p2 += 1
        else:
            draws += 1

        # Optional: decay epsilon over time
        # agent.epsilon = max(agent.epsilon * 0.999995, 0.01)

        # Save progress every 5,000 games (for example)
        if (episode+1) % 5000 == 0:
            agent.save_qtable()
            print(
                f"Episode {episode+1}: P1 wins = {wins_for_p1}, P2 wins = {wins_for_p2}, Draws = {draws}")

    # Final save
    agent.save_qtable()
    print("Training finished.")
    print(f"P1 wins: {wins_for_p1}, P2 wins: {wins_for_p2}, Draws: {draws}")


# If you want to make it executable:
# if __name__ == "__main__":
#     main(episode_count=50000)
