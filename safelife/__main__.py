import argparse

from . import rgb_renderer
from . import interactive_game

parser = argparse.ArgumentParser(description="""
The SafeLife command-line tool can be used to interactively play
a game of SafeLife, print procedurally generated SafeLife boards,
or convert saved boards to images for easy viewing.

Please select one of the available commands to run the program.
You can run `safelife <command> --help` to get more help on a
particular command.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
subparsers = parser.add_subparsers(dest="cmd", help="Top-level command.")
interactive_game._make_cmd_args(subparsers)
rgb_renderer._make_cmd_args(subparsers)
args = parser.parse_args()
if args.cmd is None:
    parser.print_help()
else:
    args.run_cmd(args)
