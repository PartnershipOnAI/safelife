import argparse

from .game_loop import GameLoop

parser = argparse.ArgumentParser()
parser.add_argument('--board', type=int, default=15,
    help="The width and height of the square starting board")
parser.add_argument('--view', type=int, default=None,
    help="View size. Defaults to zero, in which case the view is fixed on "
    "the whole board.")
parser.add_argument('--randomize', action="store_true",
    help="Create a randomized board. If not set, the initial board is empty.")
parser.add_argument('--load',
    help="Load game state from file. Overrides settings board size and "
    "randomization settings.")
args = parser.parse_args()

main_loop = GameLoop()
main_loop.board_size = (args.board, args.board)
main_loop.random_board = args.randomize
main_loop.view_size = args.view and (args.view, args.view)
main_loop.load_from = args.load
main_loop.start_games()
