"""
Code in this module is devoted to playing the game interactively. It defines
the key bindings and the basic input->update game loop.
"""


import os
import sys
import glob

from .game_physics import GameOfLife
from .syntax_tree import StatefulProgram
from . import asci_renderer as renderer
from .gen_board import gen_board
from .keyboard_input import KEYS, getch


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
}

COMMAND_KEYS = {
    KEYS.LEFT_ARROW: "LEFT",
    KEYS.RIGHT_ARROW: "RIGHT",
    KEYS.UP_ARROW: "UP",
    KEYS.DOWN_ARROW: "DOWN",
    'a': "LEFT",
    'd': "RIGHT",
    'w': "UP",
    's': "DOWN",
    '\r': "NULL",
    'z': "NULL",
    'c': "TOGGLE",
    'i': "IFEMPTY",
    'r': "REPEAT",
    'p': "DEFINE",
    'o': "CALL",
    'l': "LOOP",
    'u': "CONTINUE",
    'b': "BREAK",
    'k': "BLOCK",
}

COMMAND_WORDS = {
    cmd: MAGIC_WORDS[k] for k, cmd in COMMAND_KEYS.items()
    if k in MAGIC_WORDS
}

EDIT_KEYS = {
    KEYS.LEFT_ARROW: "LEFT",
    KEYS.RIGHT_ARROW: "RIGHT",
    KEYS.UP_ARROW: "UP",
    KEYS.DOWN_ARROW: "DOWN",
    'x': "PUT EMPTY",
    'c': "PUT LIFE",
    'w': "PUT WALL",
    'r': "PUT CRATE",
    'e': "PUT EXIT",
    'i': "PUT ICECUBE",
    't': "PUT PLANT",
    'd': "PUT WEED",
    'p': "PUT PREDATOR",
    'f': "PUT FOUNTAIN",
    'n': "PUT SPAWNER",
    '1': "TOGGLE ALIVE",
    '2': "TOGGLE PRESERVING",
    '3': "TOGGLE INHIBITING",
    '4': "TOGGLE SPAWNING",
    '5': "CHANGE COLOR",
    's': "SAVE",
    'S': "SAVE AS",
    'R': "REVERT",
    'Q': "END LEVEL",
}
TOGGLE_EDIT = '`'


class GameLoop(object):
    """
    Play the game interactively. For humans.
    """
    game_cls = GameOfLife
    board_size = (10, 10)
    random_board = False  # Later this should be a difficulty slider
    load_from = None
    view_size = None
    centered_view = False
    fixed_orientation = False

    total_points = 0
    total_steps = 0
    total_safety_score = 0
    editing = False

    def load_levels(self):
        if self.load_from and os.path.isdir(self.load_from):
            for fname in sorted(glob.glob(os.path.join(self.load_from, '*.npz'))):
                yield self.game_cls.load(fname)
        elif self.load_from:
            yield self.game_cls.load(self.load_from)
        elif self.random_board:
            while True:
                game = self.game_cls(board_size=None)
                game.deserialize(gen_board(self.board_size))
                yield game
        else:
            yield self.game_cls(board_size=self.board_size)

    def play(self, game):
        os.system('clear')
        program = StatefulProgram(game)

        while not game.game_over:
            output = "\x1b[H\x1b[J"
            if game.title:
                output += "\x1b[1m%s\x1b[0m\n" % game.title
            output += "Score: \x1b[1m%i\x1b[0m\n" % self.total_points
            output += "Steps: \x1b[1m%i\x1b[0m\n" % self.total_steps
            output += "Powers: \x1b[3m%s\x1b[0m\n" % renderer.agent_powers(game)
            if self.editing:
                output += "\x1b[1m*** EDIT MODE ***\x1b[0m\n"
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
            if key == KEYS.INTERRUPT:
                raise KeyboardInterrupt
            elif key == KEYS.DELETE:
                program.pop_command()
            elif key == TOGGLE_EDIT:
                # Toggle the edit status. This will allow the user to
                # add/destroy blocks without advancing the game's physics.
                self.editing = not self.editing
            elif self.editing and key in EDIT_KEYS:
                # Execute action immediately.
                program.message = game.execute_edit(EDIT_KEYS[key]) or ""
            elif not self.editing and key in COMMAND_KEYS:
                points, steps = program.add_command(COMMAND_KEYS[key])
                self.total_points += points
                self.total_steps += steps

        if game.game_over != -1:
            print("Side effect scores (lower is better):\n")
            side_effect_scores = game.side_effect_score()
            subtotal = sum(side_effect_scores.values())
            self.total_safety_score += subtotal
            for ctype, score in side_effect_scores.items():
                sprite = renderer.render_cell(ctype)
                print("        %s: %6.2f" % (sprite, score))
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


if __name__ == "__main__":
    GameLoop().start_games()
