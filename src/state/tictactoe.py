"""
Created on 30/03/2025

@author: Aryan

Filename: tictactoe.py

Relative Path: src/state/tictactoe.py
"""


import pickle
import os


class TicTacToeStateManager:
    def __init__(self, cache_file='data/tictactoe/tictactoe_states.pkl'):
        self.states = {}
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            self.load_states()
        else:
            self._generate_all_states()
            self.save_states()

    def _generate_all_states(self):
        def generate_states(board, current_player):
            board_tuple = tuple(tuple(row) for row in board)
            if board_tuple in self.states:
                return

            self.states[board_tuple] = {
                'player': current_player,
                'is_terminal': self._is_terminal(board),
                'winner': self._get_winner(board)
            }

            # If terminal, don't expand further
            if self._is_terminal(board):
                return

            # Next moves
            for r in range(3):
                for c in range(3):
                    if board[r][c] == 0:
                        new_board = [row[:] for row in board]
                        new_board[r][c] = current_player
                        next_player = 1 if current_player == 2 else 2
                        generate_states(new_board, next_player)

        # Start from empty board
        initial_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        generate_states(initial_board, 1)

    def save_states(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.states, f)
            print(f"States saved to {self.cache_file}")
        except Exception as e:
            print("Error saving states:", e)

    def load_states(self):
        try:
            with open(self.cache_file, 'rb') as f:
                self.states = pickle.load(f)
            print(f"States loaded from {self.cache_file}")
        except Exception as e:
            print("Error loading states:", e)
            self._generate_all_states()

    def get_state_count(self):
        return len(self.states)

    def _is_terminal(self, board):
        # row win
        for i in range(3):
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return True
        # col win
        for i in range(3):
            if board[0][i] != 0 and board[0][i] == board[1][i] == board[2][i]:
                return True
        # diagonals
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return True
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return True

        # Full board = draw
        if all(cell != 0 for row in board for cell in row):
            return True

        return False

    def _get_winner(self, board):
        # row
        for i in range(3):
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]
        # col
        for i in range(3):
            if board[0][i] != 0 and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]
        # diag
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]
        return 0  # no winner

    def get_next_possible_moves(self, board):
        """
        Return a list of moves: each item is:
            {'move': (r,c), 'new_board': [[...]], 'state_details': {...}}
        """
        x_count = sum(row.count(1) for row in board)
        o_count = sum(row.count(2) for row in board)
        current_player = 1 if x_count == o_count else 2

        possible_moves = []
        for r in range(3):
            for c in range(3):
                if board[r][c] == 0:
                    new_board = [row[:] for row in board]
                    new_board[r][c] = current_player
                    nb_tuple = tuple(tuple(row) for row in new_board)
                    details = self.states.get(nb_tuple, {})
                    possible_moves.append({
                        'move': (r, c),
                        'new_board': new_board,
                        'state_details': details
                    })
        return possible_moves
