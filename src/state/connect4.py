import os
import pickle
import time
import signal
import multiprocessing

# --------------------------
# Helper functions
# --------------------------


def board_to_key(board):
    """Convert a board (list of lists) into a unique string key."""
    return ''.join(str(cell) for row in board for cell in row)


def check_terminal_and_winner(board, n):
    """
    Check if board is terminal and determine winner.
    Returns a tuple: (is_terminal: bool, winner: int)
      - winner = 1 or 2 if one has connect-4, or 0 for draw.
    Only one pass is made.
    """
    # Check each cell that is non-empty.
    for r in range(n):
        for c in range(n):
            player = board[r][c]
            if player == 0:
                continue

            # Horizontal check
            if c + 3 < n:
                if (board[r][c+1] == player and
                    board[r][c+2] == player and
                        board[r][c+3] == player):
                    return True, player

            # Vertical check
            if r + 3 < n:
                if (board[r+1][c] == player and
                    board[r+2][c] == player and
                        board[r+3][c] == player):
                    return True, player

            # Diagonal down-right
            if r + 3 < n and c + 3 < n:
                if (board[r+1][c+1] == player and
                    board[r+2][c+2] == player and
                        board[r+3][c+3] == player):
                    return True, player

            # Diagonal up-right
            if r - 3 >= 0 and c + 3 < n:
                if (board[r-1][c+1] == player and
                    board[r-2][c+2] == player and
                        board[r-3][c+3] == player):
                    return True, player

    # Draw check: if no empty cells
    if all(board[r][c] != 0 for r in range(n) for c in range(n)):
        return True, 0

    return False, 0


def get_next_open_row(board, col, n):
    """
    Return the row index (from bottom) where a piece will land if dropped in col.
    Return None if column is full.
    """
    for row in reversed(range(n)):
        if board[row][col] == 0:
            return row
    return None

# --------------------------
# Worker DFS function (iterative)
# --------------------------


def dfs_generate(board, current_player, n, log_frequency):
    """
    Iteratively DFS from the given board, returning a dictionary of states.
    Uses board string keys and a stack.
    """
    states = {}
    generation_count = 0
    start_time = time.time()
    stack = [(board, current_player)]
    while stack:
        curr_board, player = stack.pop()
        key = board_to_key(curr_board)
        if key in states:
            continue

        is_term, winner = check_terminal_and_winner(curr_board, n)
        states[key] = {'player': player,
                       'is_terminal': is_term, 'winner': winner}
        generation_count += 1

        if generation_count % log_frequency == 0:
            elapsed = time.time() - start_time
            print(
                f"[PID {os.getpid()}] Generated {generation_count} states; Subtree size so far: {len(states)}; Elapsed {elapsed:.2f}s")

        if is_term:
            continue

        next_player = 1 if player == 2 else 2
        # For each column, try to drop a piece if possible.
        for col in range(n):
            row = get_next_open_row(curr_board, col, n)
            if row is not None:
                # Create a new board copy with the piece dropped.
                new_board = [r[:] for r in curr_board]
                new_board[row][col] = player
                stack.append((new_board, next_player))
    return states


def worker(args):
    """
    Worker function for multiprocessing.
    args: (board, current_player, n, log_frequency)
    """
    board, current_player, n, log_frequency = args
    return dfs_generate(board, current_player, n, log_frequency)

# --------------------------
# Connect4StateManager Class
# --------------------------


class Connect4StateManager:
    def __init__(
        self,
        n=6,
        cache_file='data/connect4/connect4_states.pkl',
        log_file='data/connect4/generation_log.txt',
        log_frequency=100_000,
        use_parallel=False
    ):
        """
        Manages all possible states of an N×N Connect 4 board.
        Optimizations:
          - Uses string keys.
          - Combines terminal check and winner computation.
          - Optionally runs DFS in parallel for each first move.
          - Partially saves progress on interrupt.
        """
        self.n = n
        self.cache_file = cache_file
        self.log_file = log_file
        self.log_frequency = log_frequency
        self.use_parallel = use_parallel

        self.states = {}
        self._setup_signal_handler()

        if os.path.exists(self.cache_file):
            self.load_states()
        else:
            self._generate_all_states()
            self.save_states()

    def _setup_signal_handler(self):
        """
        Installs a signal handler so that on Ctrl-C (SIGINT) the current states are saved.
        """
        def handler(sig, frame):
            print("Interrupt received. Saving current state dictionary...")
            self.save_states()
            exit(0)
        signal.signal(signal.SIGINT, handler)

    def _log(self, message):
        """
        Log message to console and append to log file.
        """
        print(message)
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')

    def _generate_all_states(self):
        """
        Generates all states starting from the empty board.
        For parallel generation, splits by first valid moves.
        """
        self._log(f"Starting state generation for {self.n}x{self.n} board...")
        start_time = time.time()

        # Start from empty board.
        empty_board = [[0]*self.n for _ in range(self.n)]
        is_term, winner = check_terminal_and_winner(empty_board, self.n)
        key = board_to_key(empty_board)
        self.states[key] = {'player': 1,
                            'is_terminal': is_term, 'winner': winner}

        # Build jobs for each valid first move.
        jobs = []
        for col in range(self.n):
            row = get_next_open_row(empty_board, col, self.n)
            if row is not None:
                new_board = [r[:] for r in empty_board]
                new_board[row][col] = 1  # Player 1 makes the move.
                # The next move will be by player 2.
                jobs.append((new_board, 2, self.n, self.log_frequency))
        self._log(f"Launching parallel DFS on {len(jobs)} subtrees...")

        if self.use_parallel and jobs:
            # Use multiprocessing Pool to process subtrees.
            with multiprocessing.Pool() as pool:
                results = pool.map(worker, jobs)
            # Merge all dictionaries.
            for subdict in results:
                self.states.update(subdict)
        else:
            # Serial generation: process each job one by one.
            for job in jobs:
                subdict = worker(job)
                self.states.update(subdict)

        elapsed = time.time() - start_time
        self._log(
            f"Finished generating states. Total states={len(self.states)}, Time={elapsed:.2f} s")

    def save_states(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.states, f)
            self._log(f"Connect4 states saved to {self.cache_file}")
        except Exception as e:
            self._log(f"Error saving states: {e}")

    def load_states(self):
        try:
            with open(self.cache_file, 'rb') as f:
                self.states = pickle.load(f)
            self._log(f"Connect4 states loaded from {self.cache_file}")
        except Exception as e:
            self._log(f"Error loading states: {e}")
            self._generate_all_states()

    def get_state_count(self):
        return len(self.states)

    def get_next_possible_moves(self, board):
        """
        For a given board, returns a list of valid moves.
        Each move is a dictionary with:
          - 'move': column index
          - 'new_board': the board after the move
          - 'state_details': the stored state info for that new board
        """
        # Determine current player by piece counts.
        flat = [cell for row in board for cell in row]
        count1 = flat.count(1)
        count2 = flat.count(2)
        current_player = 1 if count1 == count2 else 2

        moves = []
        for col in range(self.n):
            row = get_next_open_row(board, col, self.n)
            if row is not None:
                new_board = [r[:] for r in board]
                new_board[row][col] = current_player
                key = board_to_key(new_board)
                details = self.states.get(key, {})
                moves.append({
                    'move': col,
                    'new_board': new_board,
                    'state_details': details
                })
        return moves


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # For demonstration, we use a 5x5 board.
    c4_manager = Connect4StateManager(
        n=5,
        cache_file='data/connect4/connect4_states_5x5.pkl',
        log_file='data/connect4/generation_log_5x5.txt',
        log_frequency=100_000,
        use_parallel=True
    )

    print("Total Connect4 states (5x5):", c4_manager.get_state_count())

    # Test: get next possible moves from an empty 5x5 board.
    empty_board_5x5 = [[0]*5 for _ in range(5)]
    next_moves = c4_manager.get_next_possible_moves(empty_board_5x5)
    for move_info in next_moves:
        print(
            f"Drop in column {move_info['move']} => is_terminal={move_info['state_details'].get('is_terminal')}, winner={move_info['state_details'].get('winner')}")
