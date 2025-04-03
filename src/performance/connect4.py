"""
Created on 31/03/2025

@author: Aryan

Filename: main_connect4.py
Relative Path: src/main_connect4.py

Headless simulation for Connect 4:
  - Runs N games for every combination of AI strategies:
      "random", "minimax", "alpha-beta", "qlearning"
  - Records win/loss/draw statistics
  - Generates performance plots (bar chart and heatmap)
  
Note: This file uses your existing Connect 4 modules (state manager, algorithms,
and QLearningAgent) without modifying them.
"""

import copy
import random
import time
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Import Connect4 algorithms and state manager
from algorithm.minmax_connect4 import minimax_connect4
from algorithm.minmax_alpha_beta_connect4 import minimax_alpha_beta_connect4
from algorithm.qLearning import QLearningAgent
from state.connect4 import Connect4StateManager

# -----------------------------
# Helper Functions for Evaluation
# -----------------------------


def connect4_winner(bd):
    n = len(bd)
    # Horizontal check
    for r in range(n):
        for c in range(n - 3):
            if bd[r][c] != 0 and bd[r][c] == bd[r][c+1] == bd[r][c+2] == bd[r][c+3]:
                return bd[r][c]
    # Vertical check
    for c in range(n):
        for r in range(n - 3):
            if bd[r][c] != 0 and bd[r][c] == bd[r+1][c] == bd[r+2][c] == bd[r+3][c]:
                return bd[r][c]
    # Diagonal down-right
    for r in range(n - 3):
        for c in range(n - 3):
            if bd[r][c] != 0 and bd[r][c] == bd[r+1][c+1] == bd[r+2][c+2] == bd[r+3][c+3]:
                return bd[r][c]
    # Diagonal up-right
    for r in range(3, n):
        for c in range(n - 3):
            if bd[r][c] != 0 and bd[r][c] == bd[r-1][c+1] == bd[r-2][c+2] == bd[r-3][c+3]:
                return bd[r][c]
    return 0


def connect4_evaluate(bd, depth):
    n = len(bd)
    w = connect4_winner(bd)
    if w == 1:
        return 1000 - depth
    elif w == 2:
        return depth - 1000

    score = 0
    rows = n
    cols = n

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

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            window = get_window(r, c, 0, 1, 4)
            score += score_window(window)
    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            window = get_window(r, c, 1, 0, 4)
            score += score_window(window)
    # Diagonal down-right
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = get_window(r, c, 1, 1, 4)
            score += score_window(window)
    # Diagonal up-right
    for r in range(3, rows):
        for c in range(cols - 3):
            window = get_window(r, c, -1, 1, 4)
            score += score_window(window)
    # Also check 3-length windows
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

# -----------------------------
# Headless Connect4 Simulation Class
# -----------------------------


class HeadlessConnect4:
    def __init__(self, n=4):
        self.n = n
        self.board = [[0]*n for _ in range(n)]
        self.player = 1  # 1 = Red, 2 = Yellow
        self.game_over = False
        self.winner = None
        # Initialize the Connect4 state manager (with your settings)
        self.c4_manager = Connect4StateManager(
            n=n,
            cache_file=f'data/connect4/connect4_states_{n}x{n}.pkl'
        )
        # Trajectories for Q-Learning updates
        self.q_game_trajectory_red = []
        self.q_game_trajectory_yellow = []

    def reset(self):
        self.board = [[0]*self.n for _ in range(self.n)]
        self.player = 1
        self.game_over = False
        self.winner = None
        self.q_game_trajectory_red = []
        self.q_game_trajectory_yellow = []

    def is_valid_column(self, col):
        return self.board[0][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.n-1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return None

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def find_winning_positions(self, piece):
        n = self.n
        # Horizontal
        for r in range(n):
            for c in range(n - 3):
                if (self.board[r][c] == piece and
                    self.board[r][c+1] == piece and
                    self.board[r][c+2] == piece and
                        self.board[r][c+3] == piece):
                    return [(r, c), (r, c+1), (r, c+2), (r, c+3)]
        # Vertical
        for c in range(n):
            for r in range(n - 3):
                if (self.board[r][c] == piece and
                    self.board[r+1][c] == piece and
                    self.board[r+2][c] == piece and
                        self.board[r+3][c] == piece):
                    return [(r, c), (r+1, c), (r+2, c), (r+3, c)]
        # Diagonal down-right
        for r in range(n - 3):
            for c in range(n - 3):
                if (self.board[r][c] == piece and
                    self.board[r+1][c+1] == piece and
                    self.board[r+2][c+2] == piece and
                        self.board[r+3][c+3] == piece):
                    return [(r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3)]
        # Diagonal up-right
        for r in range(3, n):
            for c in range(n - 3):
                if (self.board[r][c] == piece and
                    self.board[r-1][c+1] == piece and
                    self.board[r-2][c+2] == piece and
                        self.board[r-3][c+3] == piece):
                    return [(r, c), (r-1, c+1), (r-2, c+2), (r-3, c+3)]
        return None

    def check_win(self, piece):
        return self.find_winning_positions(piece) is not None

    def check_draw(self):
        return all(self.board[r][c] != 0 for r in range(self.n) for c in range(self.n))

    def get_ai_move(self, ai_mode, current_player, q_agent_red=None, q_agent_yellow=None):
        valid_moves = [col for col in range(
            self.n) if self.is_valid_column(col)]
        if not valid_moves:
            return None

        if ai_mode == "random":
            return random.choice(valid_moves)

        # For minimax/alpha-beta, build a state dictionary.
        # Define local helper functions that use our Connect4StateManager.
        def is_terminal(bd):
            key = ''.join(str(cell) for row in bd for cell in row)
            info = self.c4_manager.states.get(key, {})
            return info.get('is_terminal', False)

        def evaluate(bd, depth):
            return connect4_evaluate(bd, depth)

        def get_children(state):
            bd = state["board"]
            current = state["current_player"]
            moves = self.c4_manager.get_next_possible_moves(bd)
            children = []
            for move_info in moves:
                col = move_info['move']
                new_bd = move_info['new_board']
                next_player = 1 if current == 2 else 2
                child_state = {
                    "board": new_bd,
                    "current_player": next_player,
                    "is_terminal": is_terminal,
                    "evaluate": evaluate,
                    "get_children": get_children
                }
                children.append((col, child_state))
            return children

        state = {
            "board": [row[:] for row in self.board],
            "current_player": current_player,
            "is_terminal": is_terminal,
            "evaluate": evaluate,
            "get_children": get_children
        }

        if is_terminal(self.board):
            return None

        if ai_mode == "minimax":
            _, best_move = minimax_connect4(
                state, maximizing_player=1, depth=0, max_depth=6)
            return best_move
        elif ai_mode == "alpha-beta":
            _, best_move = minimax_alpha_beta_connect4(
                state, maximizing_player=1, depth=0, max_depth=6)
            return best_move
        elif ai_mode == "qlearning":
            agent = q_agent_red if current_player == 1 else q_agent_yellow
            chosen_move = agent.choose_action(self.board, valid_moves)
            if current_player == 1:
                self.q_game_trajectory_red.append(
                    (copy.deepcopy(self.board), chosen_move))
            else:
                self.q_game_trajectory_yellow.append(
                    (copy.deepcopy(self.board), chosen_move))
            return chosen_move
        else:
            return None

    def run_single_game(self, ai_mode_red, ai_mode_yellow, q_agent_red=None, q_agent_yellow=None, verbose=False):
        self.reset()
        while not self.game_over:
            current_mode = ai_mode_red if self.player == 1 else ai_mode_yellow
            move = self.get_ai_move(
                current_mode, self.player, q_agent_red, q_agent_yellow)
            if move is None:
                self.game_over = True
                self.winner = 0
                break
            if self.is_valid_column(move):
                row = self.get_next_open_row(move)
                if row is None:
                    self.game_over = True
                    self.winner = 0
                    break
                self.drop_piece(row, move, self.player)
                if verbose:
                    print(
                        f"Player {'Red' if self.player == 1 else 'Yellow'} drops in column {move}")
                    self.print_board()
                if self.check_win(self.player):
                    self.game_over = True
                    self.winner = self.player
                    # Q-Learning final reward assignment:
                    if ai_mode_red == "qlearning" and self.winner == 1:
                        q_agent_red.batch_update_from_game(
                            self.q_game_trajectory_red, final_reward=1)
                        if ai_mode_yellow == "qlearning":
                            q_agent_yellow.batch_update_from_game(
                                self.q_game_trajectory_yellow, final_reward=-1)
                    elif ai_mode_yellow == "qlearning" and self.winner == 2:
                        q_agent_yellow.batch_update_from_game(
                            self.q_game_trajectory_yellow, final_reward=1)
                        if ai_mode_red == "qlearning":
                            q_agent_red.batch_update_from_game(
                                self.q_game_trajectory_red, final_reward=-1)
                    break
                elif self.check_draw():
                    self.game_over = True
                    self.winner = 0
                    if ai_mode_red == "qlearning":
                        q_agent_red.batch_update_from_game(
                            self.q_game_trajectory_red, final_reward=0)
                    if ai_mode_yellow == "qlearning":
                        q_agent_yellow.batch_update_from_game(
                            self.q_game_trajectory_yellow, final_reward=0)
                    break
                else:
                    self.player = 2 if self.player == 1 else 1
            else:
                self.game_over = True
                self.winner = 0
                break
        return self.winner

    def print_board(self):
        symbols = {0: '.', 1: 'R', 2: 'Y'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

# -----------------------------
# Simulation Runner and Plotting
# -----------------------------


def run_simulations(num_games, save_plots=True, verbose=False):
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    # Create two separate Q-Learning agents (for Red and Yellow)
    q_agent_red = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                                 save_file='data/connect4/headless_qtable_red.pkl')
    q_agent_yellow = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                                    save_file='data/connect4/headless_qtable_yellow.pkl')
    game = HeadlessConnect4(n=4)
    results = {}
    start_time = time.time()
    total_combinations = len(ai_modes) * len(ai_modes)
    current_combo = 0

    for ai_mode_red in ai_modes:
        for ai_mode_yellow in ai_modes:
            current_combo += 1
            combo_key = f"{ai_mode_red} vs {ai_mode_yellow}"
            results[combo_key] = {"red_wins": 0, "yellow_wins": 0, "draws": 0}
            print(
                f"Running {combo_key} ({current_combo}/{total_combinations})...")

            for i in range(num_games):
                if verbose and i % (num_games // 10) == 0:
                    print(f"  Game {i+1}/{num_games}")
                winner = game.run_single_game(
                    ai_mode_red, ai_mode_yellow, q_agent_red, q_agent_yellow, verbose=False)
                if winner == 1:
                    results[combo_key]["red_wins"] += 1
                elif winner == 2:
                    results[combo_key]["yellow_wins"] += 1
                else:
                    results[combo_key]["draws"] += 1

            if "qlearning" in (ai_mode_red, ai_mode_yellow):
                q_agent_red.save_qtable()
                q_agent_yellow.save_qtable()

            print(f"Results for {combo_key}:")
            print(
                f"  Red wins: {results[combo_key]['red_wins']} ({results[combo_key]['red_wins']/num_games*100:.1f}%)")
            print(
                f"  Yellow wins: {results[combo_key]['yellow_wins']} ({results[combo_key]['yellow_wins']/num_games*100:.1f}%)")
            print(
                f"  Draws: {results[combo_key]['draws']} ({results[combo_key]['draws']/num_games*100:.1f}%)\n")

    elapsed_time = time.time() - start_time
    print(
        f"Total time: {elapsed_time:.2f} seconds for {total_combinations} combinations of {num_games} games each")

    with open(f"connect4_results_{num_games}_games.txt", "w") as f:
        f.write(
            f"Connect4 AI Simulation Results ({num_games} games per combination)\n")
        f.write(f"Total time: {elapsed_time:.2f} seconds\n\n")
        for combo_key, stats in results.items():
            f.write(f"Results for {combo_key}:\n")
            f.write(
                f"  Red wins: {stats['red_wins']} ({stats['red_wins']/num_games*100:.1f}%)\n")
            f.write(
                f"  Yellow wins: {stats['yellow_wins']} ({stats['yellow_wins']/num_games*100:.1f}%)\n")
            f.write(
                f"  Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)\n\n")

    if save_plots:
        generate_performance_plots(results, num_games)

    return results


def generate_performance_plots(results, num_games):
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    red_win_rates = []
    yellow_win_rates = []
    draw_rates = []
    labels = []

    for red_ai in ai_modes:
        for yellow_ai in ai_modes:
            combo_key = f"{red_ai} vs {yellow_ai}"
            stats = results[combo_key]
            red_win_rates.append(stats["red_wins"] / num_games * 100)
            yellow_win_rates.append(stats["yellow_wins"] / num_games * 100)
            draw_rates.append(stats["draws"] / num_games * 100)
            labels.append(combo_key)

    # Bar chart: Red wins, Yellow wins, and Draw percentages
    fig, ax = plt.subplots(figsize=(15, 8))
    barWidth = 0.25
    r1 = range(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    ax.bar(r1, red_win_rates, width=barWidth, label='Red Wins')
    ax.bar(r2, yellow_win_rates, width=barWidth, label='Yellow Wins')
    ax.bar(r3, draw_rates, width=barWidth, label='Draws')
    plt.xlabel('AI Combinations')
    plt.ylabel('Percentage (%)')
    plt.title(f'Connect4 AI Performance Comparison ({num_games} games each)')
    plt.xticks([r + barWidth for r in range(len(labels))],
               labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'connect4_performance_{num_games}_games.png')
    plt.close()

    # Heatmap: Red win rate matrix (Player 1)
    matrix_data = defaultdict(dict)
    for red_ai in ai_modes:
        for yellow_ai in ai_modes:
            combo_key = f"{red_ai} vs {yellow_ai}"
            stats = results[combo_key]
            matrix_data[red_ai][yellow_ai] = stats["red_wins"] / \
                num_games * 100

    matrix_values = [[matrix_data[red_ai][yellow_ai]
                      for yellow_ai in ai_modes] for red_ai in ai_modes]
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
    ax.set_title(f"Red Win Rate (Player 1) - {num_games} games each")
    ax.set_xlabel("Yellow (Player 2) Strategy")
    ax.set_ylabel("Red (Player 1) Strategy")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Win Percentage", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig(f'connect4_red_winrate_matrix_{num_games}_games.png')
    plt.close()


# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    num_games = 100  # Adjust the number of games per combination as needed
    print(
        f"Starting Connect4 AI simulation with {num_games} games per combination...")
    run_simulations(num_games, save_plots=True, verbose=False)
