import argparse

from . import rgb_renderer
from . import interactive_game

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="cmd")
subparsers.required = True
interactive_game._make_cmd_args(subparsers)
rgb_renderer._make_cmd_args(subparsers)
args = parser.parse_args()
args.run_cmd(args)
