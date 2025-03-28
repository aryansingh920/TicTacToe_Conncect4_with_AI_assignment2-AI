import pickle
import os


class TicTacToeStateManager:
    def __init__(self, cache_file='tictactoe_states.pkl'):
        """
        Initialize the state manager with optional caching
        
        :param cache_file: Path to save/load serialized states
        """
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
        1 = X
        2 = O
        """
        def generate_states(board, current_player):
            # Convert board to tuple for hashability
            board_tuple = tuple(tuple(row) for row in board)

            # If this state already exists, return
            if board_tuple in self.states:
                return

            # Add the current state
            self.states[board_tuple] = {
                'player': current_player,
                'is_terminal': self._is_terminal(board),
                'winner': self._get_winner(board)
            }

            # If game is terminal, don't generate further states
            if self._is_terminal(board):
                return

            # Try to place current player's mark in empty spots
            for r in range(3):
                for c in range(3):
                    if board[r][c] == 0:
                        # Create a copy of the board
                        new_board = [row[:] for row in board]
                        new_board[r][c] = current_player

                        # Switch players
                        next_player = 3 - current_player

                        # Recursively generate states
                        generate_states(new_board, next_player)

        # Start with an empty board, X as first player
        initial_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        generate_states(initial_board, 1)

    def save_states(self):
        """
        Save states to a pickle file
        """
        try:
            with open(self.cache_file, 'wb') as f:
                # Pickle the entire states dictionary
                pickle.dump(self.states, f)
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
                self.states = pickle.load(f)
            print(f"States loaded from {self.cache_file}")
        except Exception as e:
            print(f"Error loading states: {e}")
            # Fallback to generating states
            self._generate_all_states()

    def _is_terminal(self, board):
        """Check if the game has ended"""
        # Check rows, columns, diagonals for a win
        for i in range(3):
            # Rows
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return True
            # Columns
            if board[0][i] != 0 and board[0][i] == board[1][i] == board[2][i]:
                return True

        # Diagonals
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return True
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return True

        # Check for draw (full board)
        if all(cell != 0 for row in board for cell in row):
            return True

        return False

    def _get_winner(self, board):
        """Determine the winner of the board state"""
        # Check rows
        for i in range(3):
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]

        # Check columns
        for i in range(3):
            if board[0][i] != 0 and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]

        # Diagonals
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]

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
        x_count = sum(row.count(1) for row in board)
        o_count = sum(row.count(2) for row in board)
        current_player = 1 if x_count == o_count else 2

        possible_moves = []
        for r in range(3):
            for c in range(3):
                if board[r][c] == 0:
                    # Create a copy of the board
                    new_board = [row[:] for row in board]
                    new_board[r][c] = current_player

                    # Convert to tuple for lookup
                    board_tuple = tuple(tuple(row) for row in new_board)

                    possible_moves.append({
                        'move': (r, c),
                        'new_board': new_board,
                        'state_details': self.states.get(board_tuple, {})
                    })

        return possible_moves


# Example usage and demonstration
def main():
    # Create or load state manager
    state_manager = TicTacToeStateManager()

    # Print total number of states
    print(f"Total Unique Board States: {state_manager.get_state_count()}")

    # Optional: Uncomment to print terminal states
    # state_manager.print_terminal_states()

    # Example of getting next possible moves
    example_board = [
        [1, 2, 0],
        [0, 1, 0],
        [2, 0, 0]
    ]

    next_moves = state_manager.get_next_possible_moves(example_board)
    print("\nPossible Next Moves:")
    for move in next_moves:
        print(f"Move: {move['move']}")
        print("New Board State:")
        for row in move['new_board']:
            print(row)
        print("Is Terminal:", move['state_details'].get('is_terminal', False))
        print("---")


if __name__ == "__main__":
    main()
