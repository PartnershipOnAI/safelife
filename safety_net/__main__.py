import argparse

from .game_loop import GameLoop

parser = argparse.ArgumentParser()
parser.add_argument('--board', type=int, default=15,
    help="The width and height of the square starting board")
parser.add_argument('--centered_view', action='store_true',
    help="If true, the board is always centered on the agent.")
parser.add_argument('--view_size', type=int, default=None,
    help="View size. Implies a centered view.")
parser.add_argument('--fixed_orientation', action="store_true",
    help="Rotate the board such that the agent is always pointing 'up'. "
    "Implies a centered view. (not recommended for humans)")
parser.add_argument('--randomize', action="store_true",
    help="Create a randomized board. If not set, the initial board is empty.")
parser.add_argument('--load',
    help="Load game state from file. Overrides settings board size and "
    "randomization settings.")
args = parser.parse_args()

main_loop = GameLoop()
main_loop.board_size = (args.board, args.board)
main_loop.random_board = args.randomize
main_loop.centered_view = args.centered_view
main_loop.view_size = args.view_size and (args.view_size, args.view_size)
main_loop.fixed_orientation = args.fixed_orientation
main_loop.load_from = args.load
main_loop.start_games()
