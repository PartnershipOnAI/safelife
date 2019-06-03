import argparse

from .game_loop import GameLoop

parser = argparse.ArgumentParser()
parser.add_argument('--clear', action="store_true",
    help="Starts with an empty board.")
parser.add_argument('--board', type=int, default=25,
    help="The width and height of the square starting board")
parser.add_argument('--centered_view', action='store_true',
    help="If true, the board is always centered on the agent.")
parser.add_argument('--view_size', type=int, default=None,
    help="View size. Implies a centered view.")
parser.add_argument('--fixed_orientation', action="store_true",
    help="Rotate the board such that the agent is always pointing 'up'. "
    "Implies a centered view. (not recommended for humans)")
parser.add_argument('--difficulty', type=float, default=1.0,
    help="Difficulty of the random board. On a scale of 0-10.")
parser.add_argument('--load',
    help="Load game state from file. Overrides settings board size and "
    "randomization settings.")
parser.add_argument('--print_only', action="store_true",
    help="Don't run the game, just print the board.")
args = parser.parse_args()

main_loop = GameLoop()
main_loop.board_size = (args.board, args.board)
main_loop.random_board = not args.clear
main_loop.centered_view = args.centered_view
main_loop.view_size = args.view_size and (args.view_size, args.view_size)
main_loop.fixed_orientation = args.fixed_orientation
main_loop.load_from = args.load
main_loop.difficulty = args.difficulty
if args.print_only:
    main_loop.print_games()
else:
    main_loop.start_games()
