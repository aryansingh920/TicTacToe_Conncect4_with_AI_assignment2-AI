"""
Filename: qlearning.py
Path: algorithm/qlearning.py

A simple Tabular Q-learning approach for either TicTacToe or Connect4.

NOTE: For actual Connect4, the state space can be huge, so typically we might
use function approximations or more advanced techniques. Below is a basic
tabular approach for demonstration only.
"""

import random
import collections


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        alpha  = learning rate
        gamma  = discount factor
        epsilon= exploration rate (for epsilon-greedy)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-Table: dict with key=state, val=dict of (action -> q_value)
        self.q_table = collections.defaultdict(
            lambda: collections.defaultdict(float))

    def get_state_key(self, board):
        """
        Convert the board (2D list) to a tuple so it can be used as a dict key.
        Or any representation you prefer.
        """
        # For TicTacToe or Connect4: flatten board into a tuple.
        # e.g. ((1,0,2),(0,2,1),(0,0,0)) for TicTacToe
        return tuple(tuple(row) for row in board)

    def choose_action(self, board, valid_actions):
        """
        Epsilon-greedy policy:
        - With probability epsilon, pick random action
        - Otherwise pick argmax Q(s,a)
        """
        state_key = self.get_state_key(board)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Argmax over Q-values
            best_action = None
            best_q = float('-inf')
            for a in valid_actions:
                q = self.q_table[state_key][a]
                if q > best_q:
                    best_q = q
                    best_action = a
            return best_action

    def update(self, board, action, reward, next_board, next_valid_actions):
        """
        Q-learning update:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        state_key = self.get_state_key(board)
        next_state_key = self.get_state_key(next_board)

        old_q = self.q_table[state_key][action]

        # If next state is terminal, max_q_next = 0
        if not next_valid_actions:
            max_q_next = 0
        else:
            max_q_next = max([self.q_table[next_state_key][a]
                             for a in next_valid_actions])

        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q_table[state_key][action] = new_q
