"""
Generate a random board with still lifes.
"""

import numpy as np

from .game_physics import CellTypes


_ring = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int8)
_dot = 1 - _ring


def check_violations(board, neighbors):
    """
    Number of violations on each cell.
    """
    neg_violations = neighbors == 3
    pos_violations = np.abs(2*neighbors - 5) // 2
    return board * pos_violations + (1-board) * neg_violations


def try_flip(board, neighbors):
    """
    Determine how many cells in the moore neighborhood would go into
    a bad state if the center cell flipped.

    Both `board` and `neighbors` should be 3x3 arrays.
    The board should be boolean, while the neighbors array should contain
    the number of neighbors in the neighborhood of each cell (up to 8).

    Returns the number of change in number of violations (negative is better).
    """
    z = board[1,1]
    dz = 1 - 2*z
    v1 = np.sum(check_violations(board, neighbors))
    v2 = np.sum(check_violations(board + dz*_dot, neighbors + dz*_ring))
    return v2 - v1


def gen_still_life(
        board_size=(15, 15), max_iter=10000, randp=0.2, min_total=8,
        num_seeds=1):
    """
    Generate a random still life.

    This isn't terribly efficient. It seems like there should be a better
    algorithm for doing this, but there's *certainly* a more efficient way
    to run this algorithm (likely requires writing in C to get a significant
    speed up). This still probably won't be a bottleneck in any agent training.

    Note that the board is only populated in its interior. Its edges are bare.
    """
    board = np.zeros(board_size, dtype=np.int8)
    neighbors = np.zeros(board_size, dtype=np.int8)
    x0 = np.random.randint(1, board_size[1]-1, size=num_seeds)
    y0 = np.random.randint(1, board_size[0]-1, size=num_seeds)
    board[y0, x0] = 1
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not (dx == dy == 0):
                neighbors[y0+dy, x0+dx] += 1
    total = num_seeds
    for _ in range(max_iter):
        # Follow the WalkSAT algorithm, more or less.
        # Pick a cell that has bad conditions (wrong number of neighbors)
        # and flip either it or one of its neighbors, with more weight
        # given to those flips that minimize violations.
        bad_y, bad_x = np.nonzero(check_violations(board, neighbors))
        if len(bad_y) == 0:
            if total >= min_total:
                break
            bad_y, bad_x = np.nonzero(board)
            if len(bad_x) == 0:
                bad_x = [np.random.randint(board_size[1])]
                bad_y = [np.random.randint(board_size[0])]
        i = np.random.randint(len(bad_x))
        y_min = np.clip(bad_y[i]-1, 1, board_size[0]-2)
        y_max = np.clip(bad_y[i]+2, 1, board_size[0]-1)
        x_min = np.clip(bad_x[i]-1, 1, board_size[1]-2)
        x_max = np.clip(bad_x[i]+2, 1, board_size[1]-1)
        coords = np.array([
            (y, x) for y in range(y_min, y_max) for x in range(x_min, x_max)
        ])
        violations = np.array([
            try_flip(board[y-1:y+2, x-1:x+2], neighbors[y-1:y+2, x-1:x+2])
            for (y, x) in coords
        ])
        sub_board = board[coords[:,0], coords[:,1]]
        violations += 50 * (total <= min_total) * sub_board
        if np.random.random() < randp:
            k = np.random.randint(len(violations))
        else:
            k = np.argmin(violations)
        y, x = coords[k]
        z = board[y, x]
        dz = 1 - 2*z
        total += dz
        board[y, x] += dz
        neighbors[y-1:y+2, x-1:x+2] += _ring * dz
    return board


def gen_board(board_size=(15, 15)):
    area = board_size[0] * board_size[1]
    board = gen_still_life(board_size, min_total=area // 15, num_seeds=area // 100)
    board = (board * CellTypes.life).astype(np.int16)
    board[0,0] = CellTypes.player | CellTypes.color_r
    walls = (np.random.random(board.shape) < 0.05)
    board += (board == 0) * walls * CellTypes.wall
    crates = (np.random.random(board.shape) < 0.05)
    board += (board == 0) * crates * CellTypes.crate

    goals = np.zeros_like(board)
    goals += np.random.random(board.shape) < 0.1
    goals -= np.random.random(board.shape) < 0.05

    x0, y0 = np.nonzero(board == CellTypes.empty)
    k = np.random.randint(len(x0))
    board[x0[k], y0[k]] = CellTypes.level_exit | CellTypes.color_r
    goals[x0[k], y0[k]] = 0

    return {
        'board': board,
        'goals': goals,
        'agent_loc': (0,0),
    }


def print_board(board):
    board = np.pad(board, [(0,0), (0,1)], 'constant', constant_values=2)
    s = np.array([' .', ' x', ' \n'])
    print(''.join(s[board.ravel()]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=1, type=int)
    parser.add_argument('--min', default=10, type=int)
    parser.add_argument('--iter', default=1000, type=int)
    parser.add_argument('--p', default=0.2, type=float)
    parser.add_argument('--seeds', default=3, type=int)
    parser.add_argument('--size', default=15, type=int)
    args = parser.parse_args()
    for _ in range(args.n):
        board = gen_still_life(
            max_iter=args.iter, min_total=args.min,
            randp=args.p, num_seeds=args.seeds,
            board_size=(args.size, args.size))
        print_board(board)
