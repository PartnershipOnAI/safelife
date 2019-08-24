import os
import sys
import glob
from types import SimpleNamespace
import numpy as np

from .game_physics import SafeLife
# from .syntax_tree import StatefulProgram
from . import asci_renderer as renderer
from .gen_board import gen_game
from .keyboard_input import KEYS, getch
from .side_effects import player_side_effect_score
from .file_finder import find_files, LEVEL_DIRECTORY


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
    # 'f': "IFEMPTY",
    # 'r': "REPEAT",
    # 'p': "DEFINE",
    # 'o': "CALL",
    # '/': "LOOP",
    # "'": "CONTINUE",
    # ';': "BREAK",
    # '[': "BLOCK",
    'R': "RESTART",
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
HELP_KEY = '?'


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
    print_only = False
    recording_directory = "./plays/"

    def __init__(self):
        self.state = SimpleNamespace(
            screen="INTRO",
            game=None,
            total_points=0,
            total_steps=0,
            total_safety_score=0,
            editing=False,
            recording=False,
            recording_data=None,
            side_effects=None,
            total_side_effects=0,
            message="",
            last_command="",
            level_num=0,
        )
        if self.print_only:
            try:
                self.state.game = self.next_level()
                self.state.screen = "GAME"
            except StopIteration:
                self.state.screen = None
                print("No game boards to print")

    def level_generator(self):
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

    def next_level(self):
        if not hasattr(self, '_level_generator'):
            self._level_generator = self.level_generator()
        self.state.level_num += 1
        return next(self._level_generator)

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

    def handle_input(self, key):
        state = self.state
        state.message = ""
        state.last_command = ""
        if key == KEYS.INTERRUPT:
            state.screen = None
        elif self.print_only:
            # Hit any key to get to the next level
            try:
                state.game = self.next_level()
                state.screen = "GAME"
            except StopIteration:
                state.game = None
                state.screen = None
        elif key == HELP_KEY:
            # Switch to the help screen. Will later pop the state.
            if state.screen != "HELP":
                state.prior_screen = state.screen
                state.screen = "HELP"
        elif state.screen in ("INTRO", "LEVEL SUMMARY"):
            # Hit any key to get to the next level
            try:
                state.game = self.next_level()
                state.screen = "GAME"
            except StopIteration:
                state.game = None
                state.screen = "GAMEOVER"
        elif state.screen == "HELP":
            # Hit any key to get back to prior state
            state.screen = state.prior_screen
        elif key == TOGGLE_RECORD:
            state.recording = not state.recording
        elif key == TOGGLE_EDIT:
            state.editing = not state.editing
            if state.game is not None:
                state.game.is_editing = state.editing
        elif key == START_SHELL:
            from IPython import embed; embed()  # noqa
        elif state.screen == "GAME":
            game = state.game
            game.is_editing = state.editing

            if state.editing and key in EDIT_KEYS:
                # Execute action immediately.
                command = EDIT_KEYS[key]
                state.last_command = command
                state.message = game.execute_edit(command) or ""
            elif not state.editing and key in COMMAND_KEYS:
                command = COMMAND_KEYS[key]
                state.last_command = command
                if command.startswith("TURN "):
                    # Just execute the action. Don't do anything else.
                    game.execute_action(command)
                else:
                    # All other commands take one action
                    state.total_steps += 1
                    start_pts = game.current_points()
                    game.advance_board()
                    action_pts = game.execute_action(command)
                    end_pts = game.current_points()
                    state.total_points += (end_pts - start_pts) + action_pts
                    # record...
            if game.game_over == "RESTART":
                state.total_points -= game.current_points()
                game.revert()
                state.total_points += game.current_points()
            elif game.game_over == "ABORT LEVEL":
                state.game = self.next_level()
            elif game.game_over:
                state.screen = "LEVEL SUMMARY"
                state.side_effects = player_side_effect_score(game)
                subtotal = sum(state.side_effects.values())
                state.total_side_effects += subtotal
        elif state.screen == "GAMEOVER":
            state.screen = None

    def render_asci(self):
        intro_text = """
        ############################################################
        ##                        SafeLife                        ##
        ############################################################

        Use the arrow keys to move, 'c' to create or destroy life,
        and 'enter' to stand still. Try not to make too big of a
        mess!

        (Hit '?' to access help, or any other key to continue.)
        """

        help_text = """
        SafeLife
        ========

        Play mode
        ---------
        arrows:  movement            c:  create / destroy
        return:  wait                R:  restart level

        `:  toggle edit mode
        *:  start / stop recording
        \:  enter shell

        Edit mode
        ---------
        x:  empty                    1:  toggle alive
        a:  agent                    2:  toggle preserving
        z:  life                     3:  toggle inhibiting
        Z:  hard life                4:  toggle spawning
        w:  wall                     5:  change agent color
        r:  crate                    %:  change agent color (full range)
        e:  exit                     g:  change goal color
        i:  icecube                  G:  change goal color (full range)
        t:  plant                    s:  save
        T:  tree                     S:  save as
        p:  predator                 R:  revert level
        f:  fountain                 Q:  abort level
        n:  spawner
        """
        if not self.print_only:
            output = "\x1b[H\x1b[J"
        else:
            output = "\n"
        state = self.state
        if state.screen == "INTRO":
            output += intro_text
        elif state.screen == "HELP":
            output += help_text
        elif state.screen == "GAME" and state.game is not None:
            game = state.game
            game.update_exit_colors()
            if game.title:
                output += "\x1b[1m%s\x1b[0m\n" % game.title
            else:
                output += "\x1b[1mBoard #%i\x1b[0m\n" % state.level_num
            if self.print_only:
                output += "\n"
            else:
                output += "Score: \x1b[1m%i\x1b[0m\n" % state.total_points
                output += "Steps: \x1b[1m%i\x1b[0m\n" % state.total_steps
                output += "Completed: %s / %s\n" % game.completion_ratio()
                output += "Powers: \x1b[3m%s\x1b[0m\n" % renderer.agent_powers(game)
                if state.editing:
                    output += "\x1b[1m*** EDIT MODE ***\x1b[0m\n"
                if state.recording:
                    output += "\x1b[1m*** RECORDING ***\x1b[0m\n"
            output += renderer.render_board(game,
                self.centered_view, self.view_size, self.fixed_orientation)
            if self.print_only:
                output += "\n"
            else:
                output += '\nAction: %s\n%s\n' % (state.last_command, state.message)
        elif state.screen == "LEVEL SUMMARY" and state.side_effects is not None:
            output += "Side effect scores (lower is better):\n"
            subtotal = sum(state.side_effects.values())
            for ctype, score in state.side_effects.items():
                sprite = renderer.render_cell(ctype)
                output += "       %s: %6.2f" % (sprite, score)
            output += "    -------------"
            output += "    Total: %6.2f" % subtotal
            output += "\n\n(hit any key to continue)"
        elif state.screen == "GAMEOVER":
            output += "\n\nGame over!"
            output += "\nFinal score:", state.total_points
            output += "Final safety score: %0.2f" % state.total_side_effects
            output += "Total steps:", state.total_steps, "\n\n"
        sys.stdout.write(output)
        sys.stdout.flush()

    def render(self):
        self.render_asci()

    def run(self):
        os.system('clear')
        while self.state.screen is not None:
            self.render()
            self.handle_input(getch())


GameLoop().run()
