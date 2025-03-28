import pickle
import os
import numpy as np


class Connect4StateManager:
    def __init__(self, cache_file='connect4_states.pkl', rows=5, cols=5):
        """
        Initialize the Connect 4 state manager
        
        :param cache_file: Path to save/load serialized states
        :param rows: Number of rows in the board
        :param cols: Number of columns in the board
        """
        self.rows = rows
        self.cols = cols
        self.states = {}
        self.cache_file = cache_file

        # Try to load existing states
        if os.path.exists(cache_file):
            self.load_states()
        else:
            # Generate states if no cache exists
            self._generate_all_states()
            self.save_states()

    def _generate_all_states(self):
        """
        Recursively generate all possible board states
        
        State representation:
        0 = Empty
        1 = Player 1 (Red)
        2 = Player 2 (Yellow)
        """
        def generate_states(board, current_player, last_move=None):
            # Convert board to tuple for hashability
            board_tuple = tuple(tuple(row) for row in board)

            # If this state already exists, return
            if board_tuple in self.states:
                return

            # Add the current state
            self.states[board_tuple] = {
                'player': current_player,
                'is_terminal': self._is_terminal(board),
                'winner': self._get_winner(board),
                'last_move': last_move
            }
            
            if len(self.states) % 100000 == 0:
                print(f"{len(self.states)} states generated...")

            # If game is terminal, don't generate further states
            if self._is_terminal(board):
                return

            # Try to place current player's piece in valid columns
            for col in range(self.cols):
                # Find the lowest empty row in this column
                row = self._get_lowest_empty_row(board, col)

                # If column is not full
                if row is not None:
                    # Create a copy of the board
                    new_board = [row[:] for row in board]
                    new_board[row][col] = current_player

                    # Switch players
                    next_player = 3 - current_player

                    # Recursively generate states
                    generate_states(new_board, next_player, (row, col))

        # Start with an empty board, Player 1 as first player
        initial_board = [[0 for _ in range(self.cols)]
                         for _ in range(self.rows)]


        generate_states(initial_board, 1)

    def _get_lowest_empty_row(self, board, col):
        """
        Find the lowest empty row in a given column
        
        :param board: Current board state
        :param col: Column to check
        :return: Lowest empty row index or None if column is full
        """
        for row in range(self.rows - 1, -1, -1):
            if board[row][col] == 0:
                return row
        return None

    def save_states(self):
        """
        Save states to a pickle file
        """
        try:
            with open(self.cache_file, 'wb') as f:
                # Pickle the entire states dictionary
                pickle.dump({
                    'states': self.states,
                    'rows': self.rows,
                    'cols': self.cols
                }, f)
            print(f"States saved to {self.cache_file}")
        except Exception as e:
            print(f"Error saving states: {e}")

    def load_states(self):
        """
        Load states from a pickle file
        """
        try:
            with open(self.cache_file, 'rb') as f:
                # Load the entire states dictionary
                data = pickle.load(f)
                self.states = data['states']
                self.rows = data['rows']
                self.cols = data['cols']
            print(f"States loaded from {self.cache_file}")
        except Exception as e:
            print(f"Error loading states: {e}")
            # Fallback to generating states
            self._generate_all_states()

    def _is_terminal(self, board):
        """
        Check if the game has ended
        
        :param board: Current board state
        :return: Boolean indicating if game is over
        """
        # Check for a winner
        if self._get_winner(board):
            return True

        # Check for a draw (full board)
        return all(cell != 0 for row in board for cell in row)

    def _get_winner(self, board):
        """
        Determine the winner of the board state
        
        :param board: Current board state
        :return: Winning player (1 or 2) or 0 if no winner
        """
        # Check horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if (board[r][c] != 0 and
                    board[r][c] == board[r][c+1] ==
                        board[r][c+2] == board[r][c+3]):
                    return board[r][c]

        # Check vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if (board[r][c] != 0 and
                    board[r][c] == board[r+1][c] ==
                        board[r+2][c] == board[r+3][c]):
                    return board[r][c]

        # Check positive diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if (board[r][c] != 0 and
                    board[r][c] == board[r+1][c+1] ==
                        board[r+2][c+2] == board[r+3][c+3]):
                    return board[r][c]

        # Check negative diagonal
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if (board[r][c] != 0 and
                    board[r][c] == board[r-1][c+1] ==
                        board[r-2][c+2] == board[r-3][c+3]):
                    return board[r][c]

        # No winner
        return 0

    def get_state_count(self):
        """Return total number of unique states"""
        return len(self.states)

    def print_terminal_states(self):
        """Print all terminal states with their winners"""
        terminal_states = {
            state: details for state, details in self.states.items()
            if details['is_terminal']
        }

        print(f"Total Terminal States: {len(terminal_states)}")
        for state, details in terminal_states.items():
            print("\nBoard State:")
            for row in state:
                print(row)
            print(f"Winner: {details['winner']}")

    def get_next_possible_moves(self, board):
        """
        Get all possible next moves for a given board state
        
        :param board: Current board state (list of lists)
        :return: List of possible next moves with resulting states
        """
        # Determine current player
        player_1_count = sum(row.count(1) for row in board)
        player_2_count = sum(row.count(2) for row in board)
        current_player = 1 if player_1_count == player_2_count else 2

        possible_moves = []
        for col in range(self.cols):
            # Find the lowest empty row in this column
            row = self._get_lowest_empty_row(board, col)

            # If column is not full
            if row is not None:
                # Create a copy of the board
                new_board = [row[:] for row in board]
                new_board[row][col] = current_player

                # Convert to tuple for lookup
                board_tuple = tuple(tuple(row) for row in new_board)

                possible_moves.append({
                    'move': (row, col),
                    'new_board': new_board,
                    'state_details': self.states.get(board_tuple, {})
                })

        return possible_moves

    def visualize_board(self, board):
        """
        Create a human-readable visualization of the board
        
        :param board: Board state to visualize
        """
        print("Connect 4 Board:")
        print(" " + " ".join(str(i) for i in range(self.cols)))
        for row in reversed(board):
            print("|" + "|".join(
                "R" if cell == 1 else
                "Y" if cell == 2 else
                " " for cell in row
            ) + "|")
        print("-" * (2 * self.cols + 1))


# Example usage and demonstration
def main():
    # Create or load state manager
    state_manager = Connect4StateManager()

    # Print total number of states
    print(f"Total Unique Board States: {state_manager.get_state_count()}")

    # Example board for demonstrating next possible moves
    example_board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0],
        [0, 1, 2, 2, 2, 0, 0]
    ]

    # Visualize the example board
    state_manager.visualize_board(example_board)

    # Get next possible moves
    next_moves = state_manager.get_next_possible_moves(example_board)
    print("\nPossible Next Moves:")
    for move in next_moves:
        print(f"Move: {move['move']}")
        state_manager.visualize_board(move['new_board'])
        print("Is Terminal:", move['state_details'].get('is_terminal', False))
        print("---")


if __name__ == "__main__":
    main()
