"""
Created on 31/03/2025

@author: Aryan

Filename: main_connect4.py
Relative Path: src/main_connect4.py

Headless simulation for Connect 4 with enhanced evaluation:
  - Runs N games per run for every combination of AI strategies:
      "random", "minimax", "alpha-beta", "qlearning"
  - Repeats each matchup for multiple runs with different random seeds
  - Records win/loss/draw statistics and runtime (mean and std deviation)
  - Records Q-learning convergence (win rate over episodes)
  - Compares different Q-learning hyperparameters
  - Generates performance plots (bar chart with error bars, runtime comparison, Q-learning convergence plots, and hyperparameter variation)
  
Note: This file uses your existing Connect 4 modules (state manager, algorithms,
and QLearningAgent) without modifying them.
"""

import copy
import random
import time
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

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
# New Simulation Runner and Plotting Functions
# -----------------------------


def run_simulations_multiple_runs(num_games, num_runs=5, save_plots=True, verbose=False):
    """
    Runs each AI combination for multiple runs (with different random seeds)
    and records win statistics and runtime.
    Also records Q-learning convergence data (if applicable).
    """
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    aggregated_results = {}
    runtime_results = {}

    # For Q-learning convergence data: storing list of outcomes per game per run
    ql_convergence_data = defaultdict(lambda: {"red": [], "yellow": []})

    for red_ai in ai_modes:
        for yellow_ai in ai_modes:
            combo_key = f"{red_ai} vs {yellow_ai}"
            aggregated_results[combo_key] = {
                "red_wins": [], "yellow_wins": [], "draws": []}
            runtime_results[combo_key] = []

            for run in range(num_runs):
                seed = run + 42  # different seed per run
                random.seed(seed)
                # Initialize Q-learning agents only if needed:
                q_agent_red = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                                             save_file=f'data/connect4/headless_qtable_red_{combo_key}_{run}.pkl')
                q_agent_yellow = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                                                save_file=f'data/connect4/headless_qtable_yellow_{combo_key}_{run}.pkl')
                game = HeadlessConnect4(n=4)
                run_stats = {"red_wins": 0, "yellow_wins": 0, "draws": 0}
                # For Q-learning convergence, record per-game outcome (win=1, loss/draw=0)
                ql_red_outcomes = []
                ql_yellow_outcomes = []

                start_run = time.time()
                for i in range(num_games):
                    winner = game.run_single_game(
                        red_ai, yellow_ai, q_agent_red, q_agent_yellow, verbose=verbose)
                    if winner == 1:
                        run_stats["red_wins"] += 1
                        if red_ai == "qlearning":
                            ql_red_outcomes.append(1)
                        if yellow_ai == "qlearning":
                            ql_yellow_outcomes.append(0)
                    elif winner == 2:
                        run_stats["yellow_wins"] += 1
                        if red_ai == "qlearning":
                            ql_red_outcomes.append(0)
                        if yellow_ai == "qlearning":
                            ql_yellow_outcomes.append(1)
                    else:
                        run_stats["draws"] += 1
                        if red_ai == "qlearning":
                            ql_red_outcomes.append(0)
                        if yellow_ai == "qlearning":
                            ql_yellow_outcomes.append(0)
                end_run = time.time()
                runtime = end_run - start_run
                runtime_results[combo_key].append(runtime)

                aggregated_results[combo_key]["red_wins"].append(
                    run_stats["red_wins"])
                aggregated_results[combo_key]["yellow_wins"].append(
                    run_stats["yellow_wins"])
                aggregated_results[combo_key]["draws"].append(
                    run_stats["draws"])

                # Store convergence data if Q-learning is involved:
                if red_ai == "qlearning":
                    ql_convergence_data[combo_key]["red"].append(
                        ql_red_outcomes)
                if yellow_ai == "qlearning":
                    ql_convergence_data[combo_key]["yellow"].append(
                        ql_yellow_outcomes)

            # After all runs, calculate and print summary statistics:
            red_mean = np.mean(aggregated_results[combo_key]["red_wins"])
            red_std = np.std(aggregated_results[combo_key]["red_wins"])
            yellow_mean = np.mean(aggregated_results[combo_key]["yellow_wins"])
            yellow_std = np.std(aggregated_results[combo_key]["yellow_wins"])
            draws_mean = np.mean(aggregated_results[combo_key]["draws"])
            draws_std = np.std(aggregated_results[combo_key]["draws"])
            runtime_mean = np.mean(runtime_results[combo_key])
            runtime_std = np.std(runtime_results[combo_key])

            print(
                f"Results for {combo_key} over {num_runs} runs of {num_games} games each:")
            print(f"  Red wins: {red_mean:.2f} ± {red_std:.2f}")
            print(f"  Yellow wins: {yellow_mean:.2f} ± {yellow_std:.2f}")
            print(f"  Draws: {draws_mean:.2f} ± {draws_std:.2f}")
            print(
                f"  Average runtime: {runtime_mean:.2f}s ± {runtime_std:.2f}s\n")

    if save_plots:
        generate_error_bar_plots(
            aggregated_results, num_games, runtime_results)
        generate_qlearning_convergence_plots(ql_convergence_data)

    return aggregated_results, runtime_results, ql_convergence_data


def generate_error_bar_plots(aggregated_results, num_games, runtime_results):
    # Bar chart for win rates with error bars
    labels = list(aggregated_results.keys())
    red_means = [np.mean(aggregated_results[label]["red_wins"]
                         ) / num_games * 100 for label in labels]
    red_stds = [np.std(aggregated_results[label]["red_wins"]
                       ) / num_games * 100 for label in labels]
    yellow_means = [np.mean(
        aggregated_results[label]["yellow_wins"]) / num_games * 100 for label in labels]
    yellow_stds = [np.std(aggregated_results[label]["yellow_wins"]
                          ) / num_games * 100 for label in labels]
    draw_means = [np.mean(aggregated_results[label]["draws"]) /
                  num_games * 100 for label in labels]
    draw_stds = [np.std(aggregated_results[label]["draws"]) /
                 num_games * 100 for label in labels]

    x = np.arange(len(labels))
    barWidth = 0.25

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - barWidth, red_means, width=barWidth,
           yerr=red_stds, capsize=5, label='Red Wins')
    ax.bar(x, yellow_means, width=barWidth,
           yerr=yellow_stds, capsize=5, label='Yellow Wins')
    ax.bar(x + barWidth, draw_means, width=barWidth,
           yerr=draw_stds, capsize=5, label='Draws')
    ax.set_xlabel('AI Combination')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(
        f'Connect4 AI Performance with Variability ({num_games} games per run)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('connect4_performance_error_bars.png')
    plt.close()

    # Plot runtime error bars for each AI combination:
    runtime_means = [np.mean(runtime_results[label]) for label in labels]
    runtime_stds = [np.std(runtime_results[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x, runtime_means, yerr=runtime_stds, capsize=5)
    ax.set_xlabel('AI Combination')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title(
        f'Connect4 Simulation Runtime per AI Combination ({num_games} games per run)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('connect4_runtime_error_bars.png')
    plt.close()


def generate_qlearning_convergence_plots(ql_convergence_data):
    # For each combination that involves Q-learning, plot the moving average win rate over episodes.
    for combo_key, data in ql_convergence_data.items():
        # Plot for Red Q-learning convergence if data is available
        if data["red"]:
            # Assume all runs have the same number of episodes (games)
            num_runs = len(data["red"])
            num_episodes = len(data["red"][0])
            # Compute average win (1=win, 0 otherwise) per episode over runs:
            # percentage win rate per episode
            red_avg = np.mean(data["red"], axis=0) * 100
            episodes = np.arange(1, num_episodes+1)
            # Compute a moving average (window size = 10)
            window = 10
            red_moving_avg = np.convolve(
                red_avg, np.ones(window)/window, mode='valid')

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episodes[window-1:], red_moving_avg,
                    label='Red Q-learning Win Rate')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Win Rate (%)')
            ax.set_title(f'Q-learning Convergence for Red in {combo_key}')
            ax.legend()
            plt.tight_layout()
            filename = f'qlearning_convergence_red_{combo_key.replace(" ", "_")}.png'
            plt.savefig(filename)
            plt.close()

        # Similarly, plot for Yellow Q-learning convergence if available:
        if data["yellow"]:
            num_runs = len(data["yellow"])
            num_episodes = len(data["yellow"][0])
            yellow_avg = np.mean(data["yellow"], axis=0) * 100
            episodes = np.arange(1, num_episodes+1)
            window = 10
            yellow_moving_avg = np.convolve(
                yellow_avg, np.ones(window)/window, mode='valid')

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episodes[window-1:], yellow_moving_avg,
                    label='Yellow Q-learning Win Rate')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Win Rate (%)')
            ax.set_title(f'Q-learning Convergence for Yellow in {combo_key}')
            ax.legend()
            plt.tight_layout()
            filename = f'qlearning_convergence_yellow_{combo_key.replace(" ", "_")}.png'
            plt.savefig(filename)
            plt.close()


def run_qlearning_parameter_experiments(num_games, num_runs, alphas, gammas, epsilons):
    """
    Runs experiments varying Q-learning hyperparameters (alpha, gamma, epsilon)
    against a baseline opponent (here, using "random" for Player 2) and reports the win rate.
    """
    param_results = {}

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                key = f"alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}"
                wins = []
                for run in range(num_runs):
                    seed = run + 1000
                    random.seed(seed)
                    q_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon,
                                             save_file=f'data/connect4/qtable_param_{key}_{run}.pkl')
                    game = HeadlessConnect4(n=4)
                    win_count = 0
                    for i in range(num_games):
                        # Here, we pit Q-learning (Player 1) against a random opponent (Player 2)
                        winner = game.run_single_game(
                            "qlearning", "random", q_agent, None, verbose=False)
                        if winner == 1:
                            win_count += 1
                    # win rate in percentage
                    wins.append(win_count / num_games * 100)
                param_results[key] = wins
                print(
                    f"{key}: Mean win rate = {np.mean(wins):.2f}% ± {np.std(wins):.2f}% over {num_runs} runs")

    # Plot the hyperparameter variation results:
    keys = list(param_results.keys())
    win_means = [np.mean(param_results[k]) for k in keys]
    win_stds = [np.std(param_results[k]) for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x, win_means, yerr=win_stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_xlabel('Q-learning Hyperparameters')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(
        'Effect of Q-learning Hyperparameters on Win Rate (vs Random)')
    plt.tight_layout()
    plt.savefig('qlearning_parameter_variation.png')
    plt.close()

    return param_results


# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    num_games = 100  # Number of games per run (adjust as needed)
    num_runs = 5     # Run each experiment 5 times to compute variance/SD
    print(
        f"Starting Connect4 AI simulation with {num_games} games per run, over {num_runs} runs each...")
    aggregated_results, runtime_results, ql_convergence_data = run_simulations_multiple_runs(
        num_games, num_runs, save_plots=True, verbose=False)

    # Run experiments to test Q-learning hyperparameter variation.
    alphas = [0.1, 0.5]
    gammas = [0.7, 0.9]
    epsilons = [0.1, 0.3]
    run_qlearning_parameter_experiments(
        num_games, num_runs, alphas, gammas, epsilons)
