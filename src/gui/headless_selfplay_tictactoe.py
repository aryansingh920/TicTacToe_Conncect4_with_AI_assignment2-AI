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


def play_game_q_vs_q(agent, transform_state_for_O=True):
    """
    One self-play game: X=1, O=2
    Uses a single agent that acts for both players
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
                agent.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agent.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            action = agent.choose_action(board, valid_moves)
            board_copy = copy.deepcopy(board)
            q_game_trajectory_X.append((board_copy, action))

            r, c = action
            board[r][c] = 1

            if check_win(1, board):
                agent.batch_update_from_game(
                    q_game_trajectory_X, final_reward=+1)
                agent.batch_update_from_game(
                    q_game_trajectory_O, final_reward=-1)
                return (1, board)
            if check_draw(board):
                agent.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agent.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            player = 2
        else:
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                agent.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agent.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            # For player O, we can either:
            # 1. Transform the board to look like X's perspective
            # 2. Keep the original board but have the agent handle different players
            if transform_state_for_O:
                # Option 1: Transform board so O looks like X to the agent
                transformed_board = transform_board_for_O(board)
                action = agent.choose_action(transformed_board, valid_moves)
            else:
                # Option 2: Pass the player information to the agent
                action = agent.choose_action(board, valid_moves, player=2)

            board_copy = copy.deepcopy(board)
            q_game_trajectory_O.append((board_copy, action))

            r, c = action
            board[r][c] = 2

            if check_win(2, board):
                agent.batch_update_from_game(
                    q_game_trajectory_O, final_reward=+1)
                agent.batch_update_from_game(
                    q_game_trajectory_X, final_reward=-1)
                return (2, board)
            if check_draw(board):
                agent.batch_update_from_game(
                    q_game_trajectory_X, final_reward=0)
                agent.batch_update_from_game(
                    q_game_trajectory_O, final_reward=0)
                return (0, board)

            player = 1


def transform_board_for_O(board):
    """
    Transform the board so that from O's perspective, it looks like X's perspective.
    This allows using the same Q-table for both players.
    X (1) becomes O (2) and vice versa.
    """
    transformed = copy.deepcopy(board)
    for r in range(3):
        for c in range(3):
            if transformed[r][c] == 1:
                transformed[r][c] = 2
            elif transformed[r][c] == 2:
                transformed[r][c] = 1
    return transformed


def main(episode_count=10000):
    # Create a single Q agent for both players
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2,
                           save_file='data/tictactoe/qtable.pkl')

    num_episodes = episode_count

    wins_for_X = 0
    wins_for_O = 0
    draws = 0

    for episode in range(num_episodes):
        winner, final_board = play_game_q_vs_q(agent)
        if winner == 1:
            wins_for_X += 1
        elif winner == 2:
            wins_for_O += 1
        else:
            draws += 1

        # Decay epsilon if desired
        # e.g.: agent.epsilon *= 0.9999

        # Save progress every 1000 games
        if (episode+1) % 1000 == 0:
            agent.save_qtable()
            print(
                f"Episode {episode+1} - X wins: {wins_for_X}, O wins: {wins_for_O}, Draws: {draws}")

    # Final save
    agent.save_qtable()
    print("Training finished.")
    print(f"X wins: {wins_for_X}, O wins: {wins_for_O}, Draws: {draws}")


# if __name__ == "__main__":
#     main()
