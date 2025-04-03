
"""
Created on 31/03/2025

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

import copy
import random
import pickle
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the actual algorithm implementations
from algorithm.minmax import minimax_tictactoe
from algorithm.minmax_alpha_beta import minimax_alpha_beta_tictactoe
from algorithm.qLearning import QLearningAgent

# Define a simple TicTacToe state manager if not provided elsewhere


class TicTacToeStateManager:
    def __init__(self):
        pass

    def _is_terminal(self, board):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return True
            if board[0][i] != 0 and board[0][i] == board[1][i] == board[2][i]:
                return True
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return True
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return True
        # Check for draw
        return all(cell != 0 for row in board for cell in row)

# ---------------------------------------------------------------------
# HEADLESS SIMULATION CODE
# ---------------------------------------------------------------------


class HeadlessTicTacToe:
    def __init__(self):
        self.board = [[0]*3 for _ in range(3)]
        self.player = 1  # 1 = X, 2 = O
        self.game_over = False
        self.winner = None
        self.ttt_manager = TicTacToeStateManager()
        self.q_game_trajectory_X = []
        self.q_game_trajectory_O = []

    def reset(self):
        self.board = [[0]*3 for _ in range(3)]
        self.player = 1
        self.game_over = False
        self.winner = None
        self.q_game_trajectory_X = []
        self.q_game_trajectory_O = []

    def mark_square(self, row, col, player_id):
        self.board[row][col] = player_id

    def is_available(self, row, col):
        return self.board[row][col] == 0

    def check_win(self, player_id):
        for r in range(3):
            if all(self.board[r][c] == player_id for c in range(3)):
                return True
        for c in range(3):
            if all(self.board[r][c] == player_id for r in range(3)):
                return True
        if all(self.board[i][i] == player_id for i in range(3)):
            return True
        if all(self.board[i][2 - i] == player_id for i in range(3)):
            return True
        return False

    def check_draw(self):
        return all(cell != 0 for row in self.board for cell in row)

    def transform_board_for_O(self, bd):
        transformed = copy.deepcopy(bd)
        for r in range(3):
            for c in range(3):
                if transformed[r][c] == 1:
                    transformed[r][c] = 2
                elif transformed[r][c] == 2:
                    transformed[r][c] = 1
        return transformed

    def tictactoe_is_terminal(self, bd):
        return self.ttt_manager._is_terminal(bd)

    def tictactoe_evaluate(self, bd, depth):
        for i in range(3):
            if bd[i][0] == bd[i][1] == bd[i][2] == 1:
                return 10 - depth
            if bd[i][0] == bd[i][1] == bd[i][2] == 2:
                return depth - 10
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
        return 0

    def tictactoe_get_children(self, state):
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

    def get_ai_move(self, ai_mode, current_player, q_agent=None):
        state = {
            "board": [row[:] for row in self.board],
            "current_player": current_player,
            "is_terminal": self.tictactoe_is_terminal,
            "evaluate": self.tictactoe_evaluate,
            "get_children": self.tictactoe_get_children
        }

        if ai_mode == "random":
            valid_moves = [(r, c) for r in range(3)
                           for c in range(3) if self.is_available(r, c)]
            return random.choice(valid_moves) if valid_moves else None

        elif ai_mode == "minimax":
            _, move = minimax_tictactoe(state, maximizing_player=1)
            return move

        elif ai_mode == "alpha-beta":
            _, move = minimax_alpha_beta_tictactoe(state, maximizing_player=1)
            return move

        elif ai_mode == "qlearning":
            if q_agent is None:
                raise ValueError(
                    "Q-learning agent is required for qlearning mode")

            valid_moves = [(r, c) for r in range(3)
                           for c in range(3) if self.board[r][c] == 0]
            if not valid_moves:
                return None

            q_trajectory = self.q_game_trajectory_X if current_player == 1 else self.q_game_trajectory_O
            board_to_use = self.transform_board_for_O(
                self.board) if current_player == 2 else self.board
            chosen_move = q_agent.choose_action(board_to_use, valid_moves)
            board_copy = [row[:] for row in self.board]
            q_trajectory.append((board_copy, chosen_move))
            return chosen_move

        return None

    def run_single_game(self, ai_mode_1, ai_mode_2, q_agent=None, verbose=False):
        self.reset()
        while not self.game_over:
            current_ai_mode = ai_mode_1 if self.player == 1 else ai_mode_2
            move = self.get_ai_move(current_ai_mode, self.player, q_agent)

            if move:
                r, c = move
                if self.is_available(r, c):
                    self.mark_square(r, c, self.player)
                    if verbose:
                        print(
                            f"Player {'X' if self.player == 1 else 'O'} placed at {r}, {c}")
                        self.print_board()

                    if self.check_win(self.player):
                        self.game_over = True
                        self.winner = self.player
                        if verbose:
                            print(
                                f"Player {'X' if self.player == 1 else 'O'} wins!")
                        if ai_mode_1 == "qlearning" and self.player == 1:
                            q_agent.batch_update_from_game(
                                self.q_game_trajectory_X, final_reward=1)
                        elif ai_mode_2 == "qlearning" and self.player == 2:
                            q_agent.batch_update_from_game(
                                self.q_game_trajectory_O, final_reward=1)
                        if ai_mode_1 == "qlearning" and self.player == 2:
                            q_agent.batch_update_from_game(
                                self.q_game_trajectory_X, final_reward=-1)
                        elif ai_mode_2 == "qlearning" and self.player == 1:
                            q_agent.batch_update_from_game(
                                self.q_game_trajectory_O, final_reward=-1)

                    elif self.check_draw():
                        self.game_over = True
                        self.winner = 0
                        if verbose:
                            print("Game is a draw!")
                        if ai_mode_1 == "qlearning":
                            q_agent.batch_update_from_game(
                                self.q_game_trajectory_X, final_reward=0)
                        if ai_mode_2 == "qlearning":
                            q_agent.batch_update_from_game(
                                self.q_game_trajectory_O, final_reward=0)

                    else:
                        # Switch player (1->2 or 2->1)
                        self.player = 3 - self.player
            else:
                self.game_over = True
                self.winner = 0

        return self.winner

    def print_board(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for i, row in enumerate(self.board):
            print(f"{i} {symbols[row[0]]}|{symbols[row[1]]}|{symbols[row[2]]}")
            if i < 2:
                print("  -+-+-")
        print()

# ---------------------------------------------------------------------
# SIMULATION RUNNER
# ---------------------------------------------------------------------


def run_simulations(num_games, save_plots=True, verbose=False):
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                             save_file='data/tictactoe/headless_qtable.pkl')
    game = HeadlessTicTacToe()
    results = {}
    start_time = time.time()
    total_combinations = len(ai_modes) * len(ai_modes)
    current_combo = 0

    for ai_mode_1 in ai_modes:
        for ai_mode_2 in ai_modes:
            current_combo += 1
            combo_key = f"{ai_mode_1} vs {ai_mode_2}"
            results[combo_key] = {"x_wins": 0, "o_wins": 0, "draws": 0}
            print(
                f"Running {combo_key} ({current_combo}/{total_combinations})...")

            for i in range(num_games):
                if verbose and i % (num_games // 10) == 0:
                    print(f"  Game {i+1}/{num_games}")
                winner = game.run_single_game(
                    ai_mode_1, ai_mode_2, q_agent, verbose=False)
                if winner == 1:
                    results[combo_key]["x_wins"] += 1
                elif winner == 2:
                    results[combo_key]["o_wins"] += 1
                else:
                    results[combo_key]["draws"] += 1

            if "qlearning" in (ai_mode_1, ai_mode_2):
                q_agent.save_qtable()  # Corrected method name

            print(f"  Results for {combo_key}:")
            print(
                f"    X wins: {results[combo_key]['x_wins']} ({results[combo_key]['x_wins']/num_games*100:.1f}%)")
            print(
                f"    O wins: {results[combo_key]['o_wins']} ({results[combo_key]['o_wins']/num_games*100:.1f}%)")
            print(
                f"    Draws: {results[combo_key]['draws']} ({results[combo_key]['draws']/num_games*100:.1f}%)")
            print()

    elapsed_time = time.time() - start_time
    print(
        f"Total time: {elapsed_time:.2f} seconds for {total_combinations} combinations of {num_games} games each")

    with open(f"tictactoe_results_{num_games}_games.txt", "w") as f:
        f.write(
            f"TicTacToe AI Simulation Results ({num_games} games per combination)\n")
        f.write(f"Total time: {elapsed_time:.2f} seconds\n\n")
        for combo_key, stats in results.items():
            f.write(f"Results for {combo_key}:\n")
            f.write(
                f"  X wins: {stats['x_wins']} ({stats['x_wins']/num_games*100:.1f}%)\n")
            f.write(
                f"  O wins: {stats['o_wins']} ({stats['o_wins']/num_games*100:.1f}%)\n")
            f.write(
                f"  Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)\n\n")

    if save_plots:
        generate_performance_plots(results, num_games)

    return results


def generate_performance_plots(results, num_games):
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    x_win_rates = []
    o_win_rates = []
    draw_rates = []
    labels = []

    for x_ai in ai_modes:
        for o_ai in ai_modes:
            combo_key = f"{x_ai} vs {o_ai}"
            stats = results[combo_key]
            x_win_rates.append(stats["x_wins"] / num_games * 100)
            o_win_rates.append(stats["o_wins"] / num_games * 100)
            draw_rates.append(stats["draws"] / num_games * 100)
            labels.append(combo_key)

    fig, ax = plt.subplots(figsize=(15, 8))
    barWidth = 0.25
    r1 = range(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    ax.bar(r1, x_win_rates, width=barWidth, label='X Wins')
    ax.bar(r2, o_win_rates, width=barWidth, label='O Wins')
    ax.bar(r3, draw_rates, width=barWidth, label='Draws')
    plt.xlabel('AI Combinations')
    plt.ylabel('Percentage (%)')
    plt.title(f'TicTacToe AI Performance Comparison ({num_games} games each)')
    plt.xticks([r + barWidth for r in range(len(labels))],
               labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'tictactoe_performance_{num_games}_games.png')
    plt.close()

    matrix_data = defaultdict(dict)
    for x_ai in ai_modes:
        for o_ai in ai_modes:
            combo_key = f"{x_ai} vs {o_ai}"
            stats = results[combo_key]
            matrix_data[x_ai][o_ai] = stats["x_wins"] / num_games * 100

    matrix_values = [[matrix_data[x_ai][o_ai]
                      for o_ai in ai_modes] for x_ai in ai_modes]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_values, cmap='YlGnBu')
    ax.set_xticks(range(len(ai_modes)))
    ax.set_yticks(range(len(ai_modes)))
    ax.set_xticklabels(ai_modes)
    ax.set_yticklabels(ai_modes)
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    for i in range(len(ai_modes)):
        for j in range(len(ai_modes)):
            ax.text(j, i, f"{matrix_values[i][j]:.1f}%",
                    ha="center", va="center", color="black")
    ax.set_title(f"X Win Rate (Player 1) - {num_games} games each")
    ax.set_xlabel("O (Player 2) Strategy")
    ax.set_ylabel("X (Player 1) Strategy")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Win Percentage", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig(f'tictactoe_x_winrate_matrix_{num_games}_games.png')
    plt.close()


if __name__ == "__main__":
    num_games = 100  # Adjust this number as needed
    print(
        f"Starting TicTacToe AI simulation with {num_games} games per combination...")
    results = run_simulations(num_games, save_plots=True, verbose=False)
