"""
SafeLife command-line tool. To run, use `python3 -m safelife <command>`.
"""

import argparse

from . import render_graphics
from . import interactive_game


def run():
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
    render_graphics._make_cmd_args(subparsers)
    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
    else:
        args.run_cmd(args)


if __name__ == "__main__":
    run()
