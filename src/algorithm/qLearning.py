"""
Created on 30/03/2025

@author: Aryan

Filename: qLearning.py

Relative Path: src/algorithm/qLearning.py
"""

import random
import collections
import pickle
import os


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, save_file='data/tictactoe/qtable.pkl'):
        """
        Q-Learning Agent for 2-player games like TicTacToe.
        alpha  = learning rate
        gamma  = discount factor
        epsilon= exploration rate (epsilon-greedy)
        save_file = path to Q-table pickle file
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_file = save_file

        # Q-Table: dict[state_key][action] = Q-value
        self.q_table = collections.defaultdict(
            lambda: collections.defaultdict(float))



        self.load_qtable_if_exists()

    def get_state_key(self, board):
        """
        Convert board to a hashable state key.
        Assumes player separation: only use this for one player's turns.
        """
        return tuple(tuple(row) for row in board)

    def choose_action(self, board, valid_actions):
        """
        Epsilon-greedy action selection.
        With probability epsilon, choose random.
        Otherwise, choose action with highest Q-value.
        """
        if not valid_actions:
            return None

        state_key = self.get_state_key(board)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        best_action = None
        best_q = float('-inf')
        for action in valid_actions:
            q = self.q_table[state_key][action]
            if q > best_q:
                best_q = q
                best_action = action

        return best_action

    def update(self, board, action, reward, next_board, next_valid_actions):
        """
        Single-step Q-learning update:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        if action is None:
            return

        state_key = self.get_state_key(board)
        next_state_key = self.get_state_key(next_board)

        old_q = self.q_table[state_key][action]
        max_q_next = 0
        if next_valid_actions:
            max_q_next = max(self.q_table[next_state_key][a]
                             for a in next_valid_actions)

        updated_q = old_q + self.alpha * \
            (reward + self.gamma * max_q_next - old_q)
        self.q_table[state_key][action] = updated_q

    def batch_update_from_game(self, game_trajectory, final_reward):
        """
        Apply updates for an entire game in reverse, with final reward.
        This propagates credit backward through the trajectory.
        """
        for i in reversed(range(len(game_trajectory))):
            board, action = game_trajectory[i]
            if i == len(game_trajectory) - 1:
                self.update(board, action, final_reward, board, [])
            else:
                next_board, _ = game_trajectory[i+1]
                self.update(board, action, 0, next_board, [])

        self.save_qtable()

    def save_qtable(self):
        try:
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
            with open(self.save_file, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
        except Exception as e:
            print("Could not save Q-table:", e)

    def load_qtable_if_exists(self):
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'rb') as f:
                    data = pickle.load(f)
                dd = collections.defaultdict(
                    lambda: collections.defaultdict(float))
                for s, d in data.items():
                    dd[s] = collections.defaultdict(float, d)
                self.q_table = dd
                print(f"Q-table loaded from {self.save_file}")
            except Exception as e:
                print("Could not load Q-table:", e)
