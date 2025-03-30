"""
Created on 30/03/2025

@author: Aryan

Filename: headless_selfplay_tictactoe.py

Relative Path: src/gui/headless_selfplay_tictactoe.py
"""

import random
import collections
import copy
import sys

# Assuming we import the same QLearningAgent from your "algorithm.qLearning" file
from algorithm.qLearning import QLearningAgent


def check_win(player, board):
    # Quick check for 3x3 TTT
    # Rows
    for r in range(3):
        if board[r][0] == board[r][1] == board[r][2] == player:
            return True
    # Cols
    for c in range(3):
        if board[0][c] == board[1][c] == board[2][c] == player:
            return True
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False


def check_draw(board):
    for row in board:
        for cell in row:
            if cell == 0:
                return False
    return True


def get_valid_moves(board):
    valid_moves = []
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                valid_moves.append((r, c))
    return valid_moves


def play_game_q_vs_q(agentX, agentO):
    """
    One self-play game: X=1, O=2
    Return (winner, final_board)
    """
    board = [[0]*3 for _ in range(3)]
    player = 1
    q_game_trajectory_X = []
    q_game_trajectory_O = []

    while True:
        if player == 1:
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                # It's a draw => update Q
                agentX.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agentO.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            action = agentX.choose_action(board, valid_moves)
            board_copy = copy.deepcopy(board)
            q_game_trajectory_X.append((board_copy, action))

            r, c = action
            board[r][c] = 1

            if check_win(1, board):
                agentX.batch_update_from_game(
                    q_game_trajectory_X, final_reward=+1)
                agentO.batch_update_from_game(
                    q_game_trajectory_O, final_reward=-1)
                return (1, board)
            if check_draw(board):
                agentX.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agentO.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            player = 2
        else:
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                agentX.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agentO.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            action = agentO.choose_action(board, valid_moves)
            board_copy = copy.deepcopy(board)
            q_game_trajectory_O.append((board_copy, action))

            r, c = action
            board[r][c] = 2

            if check_win(2, board):
                agentO.batch_update_from_game(
                    q_game_trajectory_O, final_reward=+1)
                agentX.batch_update_from_game(
                    q_game_trajectory_X, final_reward=-1)
                return (2, board)
            if check_draw(board):
                agentX.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agentO.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            player = 1


def main(episode_count=10000):
    # Create two Q agents
    # You can share the same Q-table if you want, but typically we give each player its own.
    agentX = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2,
                            save_file='data/tictactoe/qtable_X.pkl')
    agentO = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2,
                            save_file='data/tictactoe/qtable_O.pkl')

    num_episodes = episode_count

    wins_for_X = 0
    wins_for_O = 0
    draws = 0

    for episode in range(num_episodes):
        winner, final_board = play_game_q_vs_q(agentX, agentO)
        if winner == 1:
            wins_for_X += 1
        elif winner == 2:
            wins_for_O += 1
        else:
            draws += 1

        # Decay epsilon if desired
        # e.g.: agentX.epsilon *= 0.9999

        # Save progress every 1000 games
        if (episode+1) % 1000 == 0:
            agentX.save_qtable()
            agentO.save_qtable()
            print(
                f"Episode {episode+1} - X wins: {wins_for_X}, O wins: {wins_for_O}, Draws: {draws}")

    # Final save
    agentX.save_qtable()
    agentO.save_qtable()
    print("Training finished.")
    print(f"X wins: {wins_for_X}, O wins: {wins_for_O}, Draws: {draws}")


# if __name__ == "__main__":
#     main()
