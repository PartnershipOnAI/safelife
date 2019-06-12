import argparse

from . import rgb_renderer
from . import game_loop

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="cmd")
game_loop._make_cmd_args(subparsers)
rgb_renderer._make_cmd_args(subparsers)
args = parser.parse_args()
args.run_cmd(args)
