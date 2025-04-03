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
import statistics
import itertools

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
        self.move_times = []  # record per-move computation time

    def reset(self):
        self.board = [[0]*3 for _ in range(3)]
        self.player = 1
        self.game_over = False
        self.winner = None
        self.q_game_trajectory_X = []
        self.q_game_trajectory_O = []
        self.move_times = []

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
        # record the game start time
        game_start_time = time.time()
        while not self.game_over:
            # record move start time
            move_start_time = time.time()
            current_ai_mode = ai_mode_1 if self.player == 1 else ai_mode_2
            move = self.get_ai_move(current_ai_mode, self.player, q_agent)
            # record move end time and save move duration
            move_duration = time.time() - move_start_time
            self.move_times.append(move_duration)

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

        total_game_time = time.time() - game_start_time
        return self.winner, total_game_time

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
    # Use default Q-learning parameters here
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                             save_file='data/tictactoe/headless_qtable.pkl')
    game = HeadlessTicTacToe()
    results = {}
    start_time = time.time()
    total_combinations = len(ai_modes) * len(ai_modes)
    current_combo = 0

    # Initialize results dictionary with game time list for runtime measurements.
    for ai_mode_1 in ai_modes:
        for ai_mode_2 in ai_modes:
            combo_key = f"{ai_mode_1} vs {ai_mode_2}"
            results[combo_key] = {"x_wins": 0,
                                  "o_wins": 0, "draws": 0, "game_times": []}

    for ai_mode_1 in ai_modes:
        for ai_mode_2 in ai_modes:
            current_combo += 1
            combo_key = f"{ai_mode_1} vs {ai_mode_2}"
            print(
                f"Running {combo_key} ({current_combo}/{total_combinations})...")

            for i in range(num_games):
                if verbose and i % (num_games // 10) == 0:
                    print(f"  Game {i+1}/{num_games}")
                # Time each game to record runtime per game
                winner, game_time = game.run_single_game(
                    ai_mode_1, ai_mode_2, q_agent, verbose=False)
                results[combo_key]["game_times"].append(game_time)
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
            avg_time = statistics.mean(results[combo_key]["game_times"])
            std_time = statistics.stdev(
                results[combo_key]["game_times"]) if num_games > 1 else 0
            print(
                f"    Average game time: {avg_time:.4f} sec (std: {std_time:.4f} sec)")
            print()

    elapsed_time = time.time() - start_time
    print(
        f"Total simulation time: {elapsed_time:.2f} seconds for {total_combinations} combinations of {num_games} games each")

    # Save overall simulation results to a file.
    with open(f"tictactoe_results_{num_games}_games.txt", "w") as f:
        f.write(
            f"TicTacToe AI Simulation Results ({num_games} games per combination)\n")
        f.write(f"Total simulation time: {elapsed_time:.2f} seconds\n\n")
        for combo_key, stats in results.items():
            avg_time = statistics.mean(stats["game_times"])
            std_time = statistics.stdev(stats["game_times"]) if len(
                stats["game_times"]) > 1 else 0
            f.write(f"Results for {combo_key}:\n")
            f.write(
                f"  X wins: {stats['x_wins']} ({stats['x_wins']/num_games*100:.1f}%)\n")
            f.write(
                f"  O wins: {stats['o_wins']} ({stats['o_wins']/num_games*100:.1f}%)\n")
            f.write(
                f"  Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)\n")
            f.write(
                f"  Average game time: {avg_time:.4f} sec (std: {std_time:.4f} sec)\n\n")

    if save_plots:
        generate_performance_plots(results, num_games)

    return results


def generate_performance_plots(results, num_games):
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    x_win_rates = []
    o_win_rates = []
    draw_rates = []
    x_win_std = []
    o_win_std = []
    draw_std = []
    labels = []

    # For plotting with error bars, we assume that when running multiple simulation runs
    # you aggregate the average win rates from each run. Here, we only have one run.
    # (For multiple runs, see run_multiple_simulations below.)
    for x_ai in ai_modes:
        for o_ai in ai_modes:
            combo_key = f"{x_ai} vs {o_ai}"
            stats = results[combo_key]
            x_win_rate = stats["x_wins"] / num_games * 100
            o_win_rate = stats["o_wins"] / num_games * 100
            draw_rate = stats["draws"] / num_games * 100
            x_win_rates.append(x_win_rate)
            o_win_rates.append(o_win_rate)
            draw_rates.append(draw_rate)
            # In single-run plots, standard deviation is zero; when aggregating runs, include std.
            x_win_std.append(0)
            o_win_std.append(0)
            draw_std.append(0)
            labels.append(combo_key)

    fig, ax = plt.subplots(figsize=(15, 8))
    barWidth = 0.25
    r1 = range(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    ax.bar(r1, x_win_rates, width=barWidth,
           yerr=x_win_std, capsize=5, label='X Wins')
    ax.bar(r2, o_win_rates, width=barWidth,
           yerr=o_win_std, capsize=5, label='O Wins')
    ax.bar(r3, draw_rates, width=barWidth,
           yerr=draw_std, capsize=5, label='Draws')
    plt.xlabel('AI Combinations')
    plt.ylabel('Percentage (%)')
    plt.title(f'TicTacToe AI Performance Comparison ({num_games} games each)')
    plt.xticks([r + barWidth for r in range(len(labels))],
               labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'tictactoe_performance_{num_games}_games.png')
    plt.close()

    # Also create a heatmap for X win rates as a matrix
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

# ---------------------------------------------------------------------
# MULTIPLE SIMULATIONS FOR VARIANCE AND RUNTIME EVALUATION
# ---------------------------------------------------------------------


def run_multiple_simulations(num_runs, num_games):
    """
    Runs the entire simulation suite multiple times with different random seeds.
    Aggregates win counts and runtime, then prints mean and standard deviation.
    """
    ai_modes = ["random", "minimax", "alpha-beta", "qlearning"]
    # Initialize aggregated results dictionary.
    aggregated_results = {}
    for x_ai in ai_modes:
        for o_ai in ai_modes:
            key = f"{x_ai} vs {o_ai}"
            aggregated_results[key] = {
                "x_wins": [], "o_wins": [], "draws": [], "avg_game_time": []}

    for run in range(num_runs):
        # Set a unique seed per run for reproducibility.
        random_seed = run
        random.seed(random_seed)
        # Run simulation without plotting to speed up
        run_result = run_simulations(
            num_games, save_plots=False, verbose=False)
        for combo_key, stats in run_result.items():
            aggregated_results[combo_key]["x_wins"].append(stats["x_wins"])
            aggregated_results[combo_key]["o_wins"].append(stats["o_wins"])
            aggregated_results[combo_key]["draws"].append(stats["draws"])
            avg_game_time = statistics.mean(stats["game_times"])
            aggregated_results[combo_key]["avg_game_time"].append(
                avg_game_time)

    print("\n--- Aggregated Results over {} runs ({} games each) ---".format(num_runs, num_games))
    for combo_key, data in aggregated_results.items():
        x_win_mean = statistics.mean(data["x_wins"])
        x_win_std = statistics.stdev(data["x_wins"]) if len(
            data["x_wins"]) > 1 else 0
        o_win_mean = statistics.mean(data["o_wins"])
        o_win_std = statistics.stdev(data["o_wins"]) if len(
            data["o_wins"]) > 1 else 0
        draw_mean = statistics.mean(data["draws"])
        draw_std = statistics.stdev(data["draws"]) if len(
            data["draws"]) > 1 else 0
        time_mean = statistics.mean(data["avg_game_time"])
        time_std = statistics.stdev(data["avg_game_time"]) if len(
            data["avg_game_time"]) > 1 else 0

        print(f"Results for {combo_key}:")
        print(f"  X wins: {x_win_mean} ± {x_win_std}")
        print(f"  O wins: {o_win_mean} ± {o_win_std}")
        print(f"  Draws:  {draw_mean} ± {draw_std}")
        print(f"  Average game time: {time_mean:.4f} sec ± {time_std:.4f} sec")
        print()

# ---------------------------------------------------------------------
# Q-LEARNING CONVERGENCE PLOT
# ---------------------------------------------------------------------


def run_qlearning_convergence(num_episodes=1000, window=100):
    """
    Runs Q-learning (as player 1 against a random opponent) for a given number
    of episodes, records win rate, and plots a moving average to show convergence.
    """
    game = HeadlessTicTacToe()
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                             save_file='data/tictactoe/headless_qtable.pkl')
    win_history = []
    for episode in range(num_episodes):
        # Q-learning (P1) vs random (P2)
        winner, _ = game.run_single_game(
            "qlearning", "random", q_agent, verbose=False)
        win_history.append(1 if winner == 1 else 0)
    # Compute moving average of win rate
    moving_avg = [statistics.mean(win_history[i:i+window])
                  for i in range(len(win_history)-window+1)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(moving_avg)), moving_avg,
             label='Q-learning Win Rate (P1)')
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (Moving Average)")
    plt.title("Q-learning Convergence over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qlearning_convergence.png")
    plt.close()
    print("Q-learning convergence plot saved as 'qlearning_convergence.png'.")

# ---------------------------------------------------------------------
# Q-LEARNING HYPERPARAMETER VARIATION EXPERIMENT
# ---------------------------------------------------------------------


def run_qlearning_parameter_variation(num_episodes=500, num_runs=3):
    """
    Varies Q-learning hyperparameters (alpha, gamma, epsilon) and runs a set of experiments.
    Reports the final win rate (and standard deviation) for each parameter combination.
    """
    param_grid = {
        "alpha": [0.1, 0.5],
        "gamma": [0.7, 0.9],
        "epsilon": [0.1, 0.3]
    }
    results = []
    for alpha, gamma, epsilon in itertools.product(param_grid["alpha"], param_grid["gamma"], param_grid["epsilon"]):
        win_rates = []
        for run in range(num_runs):
            random.seed(run)
            game = HeadlessTicTacToe()
            q_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon,
                                     save_file=f'data/tictactoe/qtable_{alpha}_{gamma}_{epsilon}_{run}.pkl')
            wins = 0
            for episode in range(num_episodes):
                winner, _ = game.run_single_game(
                    "qlearning", "random", q_agent, verbose=False)
                if winner == 1:
                    wins += 1
            win_rate = wins / num_episodes * 100
            win_rates.append(win_rate)
        mean_win_rate = statistics.mean(win_rates)
        std_win_rate = statistics.stdev(win_rates) if len(win_rates) > 1 else 0
        results.append((alpha, gamma, epsilon, mean_win_rate, std_win_rate))
        print(
            f"Parameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon} => Win Rate: {mean_win_rate:.1f}% ± {std_win_rate:.1f}%")

    # Plot the parameter variation results using a bar chart with error bars.
    labels = [f"a{alpha}_g{gamma}_e{epsilon}" for alpha,
              gamma, epsilon, _, _ in results]
    win_means = [mean for _, _, _, mean, _ in results]
    win_stds = [std for _, _, _, _, std in results]
    plt.figure(figsize=(12, 6))
    x_pos = range(len(labels))
    plt.bar(x_pos, win_means, yerr=win_stds, capsize=5)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.xlabel("Parameter Combination (alpha, gamma, epsilon)")
    plt.ylabel("Win Rate (%)")
    plt.title("Q-learning Parameter Variation")
    plt.tight_layout()
    plt.savefig("qlearning_parameter_variation.png")
    plt.close()
    print("Q-learning hyperparameter variation plot saved as 'qlearning_parameter_variation.png'.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    num_games = 100  # Adjust this number as needed for each simulation run
    print(
        f"Starting TicTacToe AI simulation with {num_games} games per combination...")
    _ = run_simulations(num_games, save_plots=True, verbose=False)

    # Run multiple simulations to report mean and standard deviation for win rates and runtime
    run_multiple_simulations(num_runs=5, num_games=num_games)

    # Run Q-learning convergence experiment and plot convergence graph
    run_qlearning_convergence(num_episodes=1000, window=100)

    # Run Q-learning hyperparameter variation experiment
    run_qlearning_parameter_variation(num_episodes=500, num_runs=3)
