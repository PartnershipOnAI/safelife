"""
Code in this module is devoted to playing the game interactively. It defines
the key bindings and the basic input->update game loop.
"""


import os
import sys
import glob
import numpy as np

from .game_physics import SafeLife
from .syntax_tree import StatefulProgram
from . import asci_renderer as renderer
from .gen_board import gen_game
from .keyboard_input import KEYS, getch
from .side_effects import player_side_effect_score
from .file_finder import find_files, LEVEL_DIRECTORY


MAGIC_WORDS = {
    'a': 'abra',
    'b': 'bin',
    'c': 'caloo',
    'd': 'distim',
    'e': 'err',
    'f': 'frabjous',
    'g': 'glom',
    'h': 'hazel',
    'i': 'illery',
    'j': 'jib',
    'k': 'kadabra',
    'l': 'listle',
    'm': 'marin',
    'n': 'nox',
    'o': 'oort',
    'p': 'ponday',
    'q': 'quell',
    'r': 'ribi',
    's': 'swarm',
    't': 'toop',
    'u': 'umbral',
    'v': 'vivify',
    'w': 'wasley',
    'x': 'xam',
    'y': 'yonder',
    'z': 'zephyr',
    'R': 'seppuku',
}

COMMAND_KEYS = {
    KEYS.LEFT_ARROW: "TURN LEFT",
    KEYS.RIGHT_ARROW: "TURN RIGHT",
    KEYS.UP_ARROW: "MOVE FORWARD",
    KEYS.DOWN_ARROW: "MOVE BACKWARD",
    'a': "TURN LEFT",
    'd': "TURN RIGHT",
    'w': "MOVE FORWARD",
    's': "MOVE BACKWARD",
    'i': "MOVE UP",
    'k': "MOVE DOWN",
    'j': "MOVE LEFT",
    'l': "MOVE RIGHT",
    'I': "TOGGLE UP",
    'K': "TOGGLE DOWN",
    'J': "TOGGLE LEFT",
    'L': "TOGGLE RIGHT",
    '\r': "NULL",
    'z': "NULL",
    'c': "TOGGLE",
    'f': "IFEMPTY",
    'r': "REPEAT",
    'p': "DEFINE",
    'o': "CALL",
    '/': "LOOP",
    "'": "CONTINUE",
    ';': "BREAK",
    '[': "BLOCK",
    'R': "RESTART",
}

COMMAND_WORDS = {
    cmd: MAGIC_WORDS[k] for k, cmd in COMMAND_KEYS.items()
    if k in MAGIC_WORDS
}

EDIT_KEYS = {
    KEYS.LEFT_ARROW: "MOVE LEFT",
    KEYS.RIGHT_ARROW: "MOVE RIGHT",
    KEYS.UP_ARROW: "MOVE UP",
    KEYS.DOWN_ARROW: "MOVE DOWN",
    'x': "PUT EMPTY",
    'a': "PUT AGENT",
    'z': "PUT LIFE",
    'Z': "PUT HARD LIFE",
    'w': "PUT WALL",
    'r': "PUT CRATE",
    'e': "PUT EXIT",
    'i': "PUT ICECUBE",
    't': "PUT PLANT",
    'T': "PUT TREE",
    'd': "PUT WEED",
    'p': "PUT PREDATOR",
    'f': "PUT FOUNTAIN",
    'n': "PUT SPAWNER",
    '1': "TOGGLE ALIVE",
    '2': "TOGGLE PRESERVING",
    '3': "TOGGLE INHIBITING",
    '4': "TOGGLE SPAWNING",
    '5': "CHANGE COLOR",
    'g': "CHANGE GOAL",
    '%': "CHANGE COLOR FULL CYCLE",
    'G': "CHANGE GOAL FULL CYCLE",
    's': "SAVE",
    'S': "SAVE AS",
    'R': "REVERT",
    'Q': "ABORT LEVEL",
}
TOGGLE_EDIT = '`'
TOGGLE_RECORD = '*'
START_SHELL = '\\'


class GameLoop(object):
    """
    Play the game interactively. For humans.
    """
    game_cls = SafeLife
    board_size = (25, 25)
    random_board = True
    difficulty = 1  # for random boards
    load_from = None
    view_size = None
    centered_view = False
    fixed_orientation = False
    gen_params = None

    total_points = 0
    total_steps = 0
    total_safety_score = 0
    editing = False
    recording = False
    recording_directory = "./plays/"

    def load_levels(self):
        if self.load_from:
            # Load file names directly
            for fname in self.load_from:
                yield self.game_cls.load(fname)
        elif self.random_board:
            gen_params = self.gen_params or {}
            gen_params.setdefault('difficulty', self.difficulty)
            gen_params.setdefault('board_shape', self.board_size)
            while True:
                yield gen_game(**gen_params)
        else:
            yield self.game_cls(board_size=self.board_size)

    def next_recording_name(self):
        pattern = os.path.join(self.recording_directory, 'rec-*.npz')
        old_recordings = glob.glob(pattern)
        if not old_recordings:
            n = 1
        else:
            n = max(
                int(os.path.split(fname)[1][4:-4])
                for fname in old_recordings
            ) + 1
        fname = 'rec-{:03d}.npz'.format(n)
        return os.path.join(self.recording_directory, fname)

    def play(self, game):
        os.system('clear')
        program = StatefulProgram(game)
        game.is_editing = self.editing
        states = []
        goals = []
        orientations = []
        state_changed = True

        while not game.game_over:
            output = "\x1b[H\x1b[J"
            if game.title:
                output += "\x1b[1m%s\x1b[0m\n" % game.title
            output += "Score: \x1b[1m%i\x1b[0m\n" % self.total_points
            output += "Steps: \x1b[1m%i\x1b[0m\n" % self.total_steps
            output += "Completed: %s / %s\n" % game.completion_ratio()
            output += "Powers: \x1b[3m%s\x1b[0m\n" % renderer.agent_powers(game)
            game.update_exit_colors()
            if self.editing:
                output += "\x1b[1m*** EDIT MODE ***\x1b[0m\n"
            if self.recording and state_changed:
                states.append(game.board.copy())
                goals.append(game.goals.copy())
                orientations.append(game.orientation)
            if self.recording:
                output += "\x1b[1m*** RECORDING ***\x1b[0m\n"
            output += renderer.render_board(game,
                self.centered_view, self.view_size, self.fixed_orientation)
            output += ' '.join(program.action_log) + '\n'
            output += "%s\n" % (program.root,)
            output += program.message + "\n"
            words = [COMMAND_WORDS.get(c, '_') for c in program.command_queue]
            output += "Command: " + ' '.join(words)
            sys.stdout.write(output)
            sys.stdout.flush()

            key = getch()
            state_changed = False
            if key == KEYS.INTERRUPT:
                raise KeyboardInterrupt
            elif key == KEYS.DELETE:
                program.pop_command()
            elif key == TOGGLE_RECORD:
                self.recording = not self.recording
            elif key == TOGGLE_EDIT:
                # Toggle the edit status. This will allow the user to
                # add/destroy blocks without advancing the game's physics.
                self.editing = not self.editing
                game.is_editing = self.editing
            elif key == START_SHELL:
                from IPython import embed; embed()  # noqa
            elif self.editing and key in EDIT_KEYS:
                # Execute action immediately.
                program.message = game.execute_edit(EDIT_KEYS[key]) or ""
            elif not self.editing and key in COMMAND_KEYS:
                points, steps = program.add_command(COMMAND_KEYS[key])
                self.total_points += points
                self.total_steps += steps
                state_changed = steps > 0
            if game.game_over == "RESTART":
                game.revert()

        if states:
            os.makedirs(self.recording_directory, exist_ok=True)
            np.savez(
                self.next_recording_name(),
                board=states, orientation=orientations, goals=goals)

        if game.game_over != "ABORT LEVEL":
            print("Side effect scores (lower is better):\n")
            side_effect_scores = player_side_effect_score(game)
            subtotal = sum(side_effect_scores.values())
            self.total_safety_score += subtotal
            for ctype, score in side_effect_scores.items():
                sprite = renderer.render_cell(ctype)
                print("       %s: %6.2f" % (sprite, score))
            print("    -------------")
            print("    Total: %6.2f" % subtotal)
            print("\n\n(hit any key to continue)")
            getch()

    def start_games(self):
        self.total_points = 0
        self.total_steps = 0
        self.total_safety_score = 0
        try:
            for game in self.load_levels():
                self.play(game)
            print("\n\nGame over!")
            print("\nFinal score:", self.total_points)
            print("Final safety score: %0.2f" % self.total_safety_score)
            print("Total steps:", self.total_steps, "\n\n")
        except KeyboardInterrupt:
            print("\nGame aborted")

    def print_games(self):
        for i, game in enumerate(self.load_levels()):
            if game.title:
                print("\nBoard #%i - %s" % (i+1, game.title))
            else:
                print("\nBoard #%i" % (i+1))
            print(renderer.render_board(game))
            if getch() == KEYS.INTERRUPT:
                break


def _make_cmd_args(subparsers):
    # used by __main__.py to define command line tools
    play_parser = subparsers.add_parser(
        "play", help="Play a game of SafeLife in the terminal.")
    print_parser = subparsers.add_parser(
        "print", help="Generate new game boards and print to terminal.")
    for parser in (play_parser, print_parser):
        # they use some of the same commands
        parser.add_argument('load_from',
            nargs='*', help="Load game state from file. "
            "Effectively overrides board size and difficulty. "
            "Note that files will be searched for in the 'levels' folder "
            "if not found relative to the current working directory.")
        parser.add_argument('--board', type=int, default=25,
            help="The width and height of the square starting board")
        parser.add_argument('--difficulty', type=float, default=1.0,
            help="Difficulty of the random board. On a scale of 0-10.")
        parser.add_argument('--gen_params',
            help="Parameters for random board generation. "
            "Can either be a json file or a (quoted) json string.")
        parser.set_defaults(run_cmd=_run_cmd_args)
    play_parser.add_argument('--clear', action="store_true",
        help="Starts with an empty board.")
    play_parser.add_argument('--centered_view', action='store_true',
        help="If true, the board is always centered on the agent.")
    play_parser.add_argument('--view_size', type=int, default=None,
        help="View size. Implies a centered view.")
    play_parser.add_argument('--fixed_orientation', action="store_true",
        help="Rotate the board such that the agent is always pointing 'up'. "
        "Implies a centered view. (not recommended for humans)")


def _run_cmd_args(args):
    main_loop = GameLoop()
    main_loop.board_size = (args.board, args.board)
    if args.gen_params:
        import json
        fname = args.gen_params
        if fname[:-5] != '.json':
            fname += '.json'
        if not os.path.exists(fname):
            fname = os.path.join(LEVEL_DIRECTORY, 'random', fname)
        if os.path.exists(fname):
            with open(fname) as f:
                main_loop.gen_params = json.load(f)
        else:
            try:
                main_loop.gen_params = json.loads(args.gen_params)
            except json.JSONDecodeError as err:
                raise ValueError('"%s" is neither a file nor valid json')
    else:
        main_loop.load_from = list(find_files(*args.load_from))
    main_loop.difficulty = args.difficulty
    if args.cmd == "print":
        main_loop.print_games()
    else:
        main_loop.random_board = not args.clear
        main_loop.centered_view = args.centered_view
        main_loop.view_size = args.view_size and (args.view_size, args.view_size)
        main_loop.fixed_orientation = args.fixed_orientation
        main_loop.start_games()
