"""
The game world consists of a grid with different colored blocks. The blocks
are dynamic and have different properties. Some blocks, like walls, just sit
still. Other blocks grow and shrink by Game of Life rules, and interactions
between blocks of different colors can be quite complicated (and the
interactions are subject to change!).

The player is generally free to move amongst the blocks, although some blocks
will block her path, and some blocks may harm her. However, the player is not
helpless! She has the ability to cast spells to manipulate the world. Most
importantly, she has the ability to *absorb* some of the properties of the
blocks, and she can then *create* new blocks that have her absorbed properties.
At first her abilities will be quite simple, and she'll only be able to create
blocks that are immediately next to her. As her power grows, she'll be able to
chain her spells together to create multiple blocks at once, or even
efficiently loop her spells conditioned on the presence or absence of other
blocks in the terrain.

Commands:

    Actions (all are single statements):
        LEFT
        RIGHT
        FORWARD
        BACKWARD
        PUSH
        PULL
        ABSORB
        CREATE
        DESTROY
        NULL
    Modifiers and control flow:
        REPEAT [command]
        BLOCK [commands] NULL
        IFEMPTY [command 1] [command 2]
        LOOP [commands] NULL/BREAK/CONTINUE
        DEFINE [name] [command]
        CALL [name]

All spells are cast the moment they're uttered and syntactically complete;
there's no "finish spell" action. This makes it much easier to interact with
the world, as many actions will have an immediate effect. In order chain
actions together they must either be part of a `REPEAT`, `BLOCK`, or `LOOP`.

The `IFEMPTY` statement branches execution based on whether or not the block
in front of an agent is empty. We may want to add other if statements later.

The `LOOP` statement doesn't have a conditional. Instead, it must be exited
with a `BREAK` statement. The end of the loop can be `NULL`, `BREAK`, or
`CONTINUE`. The behavior of `NULL` is the same as `BREAK` in this context
(although we could easily switch it).

The `DEFINE` statement defines a reusable procedure with any name. The `CALL`
statement then calls a procedure previously stored under a given name.
Procedures can be called recursively. Note that the name scopes are global,
and they persist across different actions. Using blocks, a procedure can be
redefined while its being run and then called recursively.

Note that the game can function fine with just a subset of the commands.
The directions all need to be present, and probably `CREATE`, but everything
else can be swapped out. However, some of them are mutually dependent:

- `DEFINE` and `CALL` must come together
- `LOOP`, `BREAK`, and `CONTINUE` must all come together
- `BLOCK` implies `NULL`
- `IF...` should *probably* imply one of the block constructs, although it's
  possible to use it effectively with just `REPEAT`
- `LOOP` generally implies `IF...`
"""

import os
import sys
import numpy as np
import scipy.signal

from .keyboard_input import getch
from .array_utils import wrapping_array
from .gen_board import gen_board
from .syntax_tree import BlockNode

UP_ARROW_KEY = '\x1b[A'
DOWN_ARROW_KEY = '\x1b[B'
RIGHT_ARROW_KEY = '\x1b[C'
LEFT_ARROW_KEY = '\x1b[D'
INTERRUPT_KEY = '\x03'
DELETE_KEY = '\x7f'

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

SPAWN_PROB = 0.3


class CellTypes(object):
    alive = 1 << 0  # Cell obeys Game of Life rules.
    agent = 1 << 1
    movable = 1 << 2  # Can be pushed by agent.
    destructible = 1 << 3
    frozen = 1 << 4  # Does not evolve (can't turn into a living cell).
    freezing = 1 << 5  # Freezes neighbor cells. Does not imply frozen.
    exit_flag = 1 << 6
    spawning = 1 << 8  # Generates new cells of the same color.
    color_r = 1 << 9
    color_g = 1 << 10
    color_b = 1 << 11

    empty = 0
    player = agent | freezing | frozen
    wall = frozen
    crate = frozen | movable
    spawner = frozen | spawning
    level_exit = frozen | exit_flag
    life = alive | destructible
    colors = (color_r, color_g, color_b)
    rainbow_color = color_r | color_g | color_b


ACTIONS = {
    "LEFT",
    "RIGHT",
    "FORWARD",
    "BACKWARD",
    "PUSH",  # unused
    "PULL",  # unused
    "ABSORB",  # unused
    "CREATE",
    "DESTROY",
    "NULL",
}

KEY_BINDINGS = {
    LEFT_ARROW_KEY: "LEFT",
    RIGHT_ARROW_KEY: "RIGHT",
    UP_ARROW_KEY: "FORWARD",
    DOWN_ARROW_KEY: "BACKWARD",
    'a': "LEFT",
    'd': "RIGHT",
    'w': "FORWARD",
    's': "BACKWARD",
    '\r': "NULL",
    'z': "NULL",
    # 'q': "ABSORB",
    'c': "CREATE",
    'x': "DESTROY",
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
    cmd: MAGIC_WORDS[k] for k, cmd in KEY_BINDINGS.items()
    if k in MAGIC_WORDS
}


def convolve2d(*args, **kw):
    y = scipy.signal.convolve2d(*args, boundary='wrap', mode='same', **kw)
    return y.astype(np.int16)


class GameState(object):
    """
    Defines the game state and dynamics. Does NOT define rendering.
    """
    default_board_size = (20, 20)
    out_of_energy_msg = "You collapse from exhaustion."
    num_steps = 0
    default_energy = 100
    edit_mode = False
    title = None
    fname = None
    game_over = False

    def __init__(self, clear_board=False):
        self.agent_loc = np.array([0,0])
        self.orientation = 1  # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

        self.points = 0
        self.delta_points = 0  # points gained or lost within a timestep
        self.color = 1
        self.error_msg = None

        self.commands = []
        self.log_actions = []  # for debugging only
        self.saved_programs = {}
        if clear_board:
            self.board = np.zeros((self.height, self.width), dtype=np.int16)
            self.goals = np.zeros((self.height, self.width), dtype=np.int16)
            self.board[0,0] = CellTypes.player
        else:
            self.make_board()
        self.pristine = np.ones(self.board.shape, dtype=bool)

    @property
    def width(self):
        if hasattr(self, 'board'):
            return self.board.shape[1]
        return self.default_board_size[1]

    @property
    def height(self):
        if hasattr(self, 'board'):
            return self.board.shape[0]
        return self.default_board_size[0]

    def make_board(self):
        board = gen_board(
            board_size=(self.height, self.width),
            min_total=self.width * self.height // 15,
            num_seeds=self.width * self.height // 100,
        )
        self.board = (board * CellTypes.life).astype(np.int16)
        x, y = self.agent_loc
        self.board[y, x] = CellTypes.player

        walls = (np.random.random(self.board.shape) < 0.05)
        self.board += (self.board == 0) * walls * CellTypes.wall
        crates = (np.random.random(self.board.shape) < 0.05)
        self.board += (self.board == 0) * crates * CellTypes.crate
        self.goals = np.zeros_like(self.board)
        self.goals += np.random.random(self.board.shape) < 0.1
        self.goals -= np.random.random(self.board.shape) < 0.05
        self.goals *= self.board == CellTypes.empty

    def save(self, fname):
        np.savez(
            fname,
            board=self.board,
            goals=self.goals,
            agent_loc=self.agent_loc,
            version=1)

    def load(self, fname):
        if not fname.endswith('.npz'):
            fname += '.npz'
        archive = np.load(fname)
        self.title = os.path.split(fname)[1][:-4]
        self.fname = fname[:-4]
        version = archive['version'] if 'version' in archive else 0
        board = archive['board']
        if version == 0:
            self.board = np.zeros(board.shape, dtype=np.int16)
            self.board += (board == 1) * CellTypes.player
            self.board += (board == 2) * CellTypes.life
            self.board += (board == 3) * CellTypes.crate
            self.board += (board == 5) * CellTypes.spawner
        elif version == 1:
            self.board = board
        else:
            raise ValueError(f"Unrecognized file version '{version}'")
        self.goals = archive['goals']
        self.agent_loc = archive['agent_loc']
        self.base_points = self.points
        self.orientation = 1
        self.game_over = False
        self.pristine = np.ones(self.board.shape, dtype=bool)

        return archive  # In case subclasses want to extract more data

    def relative_loc(self, n_forward, n_right=0):
        """
        Retrieves a location relative to the agent.
        """
        x = n_right
        y = -n_forward
        for _ in range(self.orientation):
            x, y = -y, x
        x += self.agent_loc[0]
        x %= self.width
        y += self.agent_loc[1]
        y %= self.height
        return x, y

    def move_agent(self, dy, dx=0, can_exit=True):
        """
        Move the agent to a new location if that location is empty.
        """
        x1, y1 = self.relative_loc(dy, dx)
        x0, y0 = self.agent_loc
        if self.board[y1, x1] == CellTypes.empty:
            self.board[y1, x1] = self.board[y0, x0]
            self.board[y0, x0] = CellTypes.empty
            self.agent_loc = np.array([x1, y1])
        elif (self.board[y1, x1] & CellTypes.exit_flag) and can_exit:
            self.game_over = True
            self.delta_points += 3
        elif (dx, dy) == (0, 1) and self.board[y1, x1] & CellTypes.movable:
            x2, y2 = self.relative_loc(+2)
            if self.board[y2, x2] == CellTypes.empty:
                self.board[y2, x2] = self.board[y1, x1]
                self.board[y1, x1] = self.board[y0, x0]
                self.board[y0, x0] = CellTypes.empty
                self.agent_loc = np.array([x1, y1])
            elif self.board[y2, x2] & CellTypes.exit_flag:
                # Push a block out of this level
                self.board[y1, x1] = self.board[y0, x0]
                self.board[y0, x0] = CellTypes.empty
                self.agent_loc = np.array([x1, y1])

    def execute_action(self, action):
        """
        Execute an individual action.

        Either returns 0 or an error message.
        """
        if action in ("BREAK", "NULL"):
            return 0  # don't use up any energy. Return right away.
        self.energy -= 1
        if self.energy < 0:
            return self.out_of_energy_msg
        if action == "LEFT":
            self.orientation -= 1
            self.orientation %= 4
        elif action == "RIGHT":
            self.orientation += 1
            self.orientation %= 4
        elif action == "FORWARD":
            self.move_agent(+1)
        elif action == "BACKWARD":
            self.move_agent(-1)
        elif action == "CREATE":
            x0, y0 = self.agent_loc
            x1, y1 = self.relative_loc(1)
            if self.board[y1, x1] == CellTypes.empty:
                self.board[y1, x1] = CellTypes.life | (
                    self.board[y0, x0] & CellTypes.rainbow_color)
        elif action == "DESTROY":
            x1, y1 = self.relative_loc(1)
            if self.board[y1, x1] & CellTypes.destructible:
                self.board[y1, x1] = CellTypes.empty

        # placeholder
        self.log_actions.append(action)
        return 0

    def define_subprogram(self, name, program):
        self.saved_programs[name] = program
        return 0

    def call_subprogram(self, name):
        if not name:
            return "A name is missing..."
        if name not in self.saved_programs:
            magic_name = COMMAND_WORDS.get(name, name)
            return "'%s' has not been bound..." % (magic_name,)
        self.energy -= 1
        if self.energy < 0:
            return self.out_of_energy_msg
        return self.saved_programs[name].execute(self)

    def execute_edit(self, key):
        """
        Like execute action, but allows for more powerful actions.
        """
        key_cell_map = {
            'x': CellTypes.empty,
            'c': CellTypes.life,
            'w': CellTypes.wall,
            'r': CellTypes.crate,
            'p': CellTypes.spawner,
            'e': CellTypes.level_exit,
        }
        x0, y0 = self.relative_loc(0)
        player_color = self.board[y0, x0] & CellTypes.rainbow_color
        x, y = self.relative_loc(1)
        if key == LEFT_ARROW_KEY:
            self.orientation -= 1
            self.orientation %= 4
        elif key == RIGHT_ARROW_KEY:
            self.orientation += 1
            self.orientation %= 4
        elif key == UP_ARROW_KEY:
            self.move_agent(+1, can_exit=False)
        elif key == DOWN_ARROW_KEY:
            self.move_agent(-1, can_exit=False)
        elif key == 'g':
            # Toggle the goal state
            self.goals[y, x] += 2
            self.goals[y, x] %= 3
            self.goals[y, x] -= 1
        elif key == 'S' or key == 's' and not self.fname:
            save_name = input('\rsave as: \x1b[J')
            if save_name:
                try:
                    self.save(save_name)
                    self.error_msg = "Saved successfully."
                except FileNotFoundError as err:
                    self.error_msg = f"No such file or directory: '{err.filename}'"
            else:
                self.error_msg = "Save aborted."
        elif key == 's':
            sys.stdout.write(f"\rsave as '{self.fname}'? (y/n)\x1b[J")
            confirm = getch()
            if confirm == 'y':
                self.save(self.fname)
                self.error_msg = "Saved successfully."
            else:
                self.error_msg = "Save aborted."
        elif key == 'l':  # that's a lowercase L
            player_color += CellTypes.color_r
            player_color &= CellTypes.rainbow_color
            self.board[y0, x0] &= ~CellTypes.rainbow_color
            self.board[y0, x0] |= player_color
        elif key == 'n':
            self.game_over = True
        elif key in key_cell_map:
            new_cell = key_cell_map[key]
            if new_cell:
                new_cell |= player_color
            self.board[y, x] = new_cell

    def check(self, condition):
        x, y = self.relative_loc(1)
        if condition == 'IFEMPTY':
            return self.board[y, x] == CellTypes.empty
        else:
            raise ValueError("Unknown condition '%s'" % (condition,))

    def advance_board(self):
        """
        Apply one timestep of physics, and return change in points.
        """
        raise NotImplementedError

    _program = None  # for debugging / logging

    def step(self, action):
        assert action in COMMAND_WORDS
        old_alive = self.board & CellTypes.alive > 0

        self.commands.append(action)
        self.num_steps += 1
        self.delta_points = 0  # can be changed by executing commands

        # It's somewhat inefficient to rebuild the program from scratch
        # when each action is added, but otherwise we'd have to handle
        # popping commands when the delete key is hit. Hardly a bottleneck.
        program = BlockNode()
        for command in self.commands:
            program.push(command)
        self._program = program  # for debugging

        if not self.game_over:
            self.advance_board()

        # Execute the commands if they're syntactically complete
        if not program.list or not program.list[-1].can_push:
            self.log_actions = []
            self.energy = self.default_energy
            err = program.execute(self)
            self.error_msg = "" if not err or err in (1,2) else err
            self.commands = []

        # Now score the board
        new_alive = self.board & CellTypes.alive > 0
        delta_alive = 1 * new_alive - 1 * old_alive
        self.delta_points += np.sum(delta_alive * self.goals)
        # The first time you mess up a pristine cell, you lose double points
        penalties = np.minimum(0, delta_alive * self.goals * self.pristine)
        self.delta_points += np.sum(penalties)
        self.pristine &= delta_alive == 0
        self.points += self.delta_points
        return self.delta_points


class GameOfLife(GameState):
    def advance_board(self):
        """
        Apply one timestep of physics using Game of Life rules.
        """
        # We can advance the board using a pretty simple convolution.
        board = self.board
        alive = board & CellTypes.alive > 0
        cfilter = np.array([[1,1,1],[1,0,1],[1,1,1]])

        frozen = board & CellTypes.frozen
        frozen |= convolve2d(board & CellTypes.freezing, cfilter)
        frozen = frozen > 0

        num_neighbors = convolve2d(alive, cfilter)
        new_alive = (num_neighbors == 3) & ~alive & ~frozen
        new_dead = ((num_neighbors < 2) | (num_neighbors > 3)) & alive & ~frozen

        # A new cell must have at least two living parents with a particular
        # color in order to inherit it.
        new_cells = np.zeros_like(board) + CellTypes.life
        for color in CellTypes.colors:
            new_cells += color * (convolve2d(board & color > 0, cfilter) > 1)
        new_cells *= new_alive
        board *= ~(new_dead | new_alive)
        board += new_cells

        # Randomly add live cells around the spawners.
        # Spawned cells contain colors of all spawners.
        spawners = board & CellTypes.spawning > 0
        spawn_num = convolve2d(spawners, cfilter)
        new_alive = np.random.random(board.shape) > (1 - SPAWN_PROB)**spawn_num
        new_alive *= (board & CellTypes.alive == 0) & ~frozen
        new_cells = np.zeros_like(board) + CellTypes.life
        for color in CellTypes.colors:
            new_cells += color * (convolve2d(board & color > 0, cfilter) > 0)
        new_cells *= new_alive
        board *= ~new_alive
        board += new_cells

        # Calculate the points
        alive = board & CellTypes.alive > 0
        points = np.sum(alive * self.goals)
        return points


class AsyncGame(GameState):
    """
    Uses probabilistic cellular automata update rules.

    Can be used to simulate e.g. a two-dimensional Ising model.
    """

    def __init__(self, rules="ising", beta=100, seed=True, **kw):
        super().__init__(**kw)
        self.rules = {
            'vine': [4, [-1, -1, 1, 1, 1], [-1, 1, -1, -1, -1]],
            'ising': [4, [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]],
            'conway': [8, [-1, -1, 1, 1] + [-1]*5, [-1]*3 + [1] + [-1]*5],
        }[rules]
        self.beta = beta

    def advance_board(self):
        """
        Apply one timestep of physics.
        """
        EPOCHS = 0.3

        board = self.board
        rules = self.rules
        w = self.width
        h = self.height
        if rules[0] == 4:
            neighborhood = np.array([[0,1,0],[1,0,1],[0,1,0]])
        elif rules[0] == 6:
            neighborhood = np.array([[0,1,1],[1,0,1],[1,1,0]])
        elif rules[0] == 8:
            neighborhood = np.array([[1,1,1],[1,0,1],[1,1,1]])
        for _ in range(int(board.size * EPOCHS)):
            x = np.random.randint(w)
            y = np.random.randint(h)
            if board[y, x] & CellTypes.frozen:
                continue
            neighbors = board.view(wrapping_array)[y-1:y+2, x-1:x+2] * neighborhood
            alive_neighbors = np.sum(neighbors & CellTypes.alive > 0)
            spawn_neighbors = np.sum(neighbors & CellTypes.spawning > 0)
            frozen = np.sum(neighbors & CellTypes.freezing) > 0
            if frozen:
                continue
            if board[y, x] & CellTypes.alive:
                H = rules[1][alive_neighbors]
            else:
                H = rules[2][alive_neighbors]

            P = 0.5 + 0.5*np.tanh(H * self.beta)
            P = 1 - (1-P)*(1-SPAWN_PROB)**spawn_neighbors
            board[y, x] = CellTypes.life if P > np.random.random() else CellTypes.empty


def render(s, view_size):
    """
    Renders the game state `s`.

    This is not exactly a speedy rendering system, but it should be plenty
    fast enough for our purposes.
    """
    CT = CellTypes
    SPRITES = {
        CT.agent: '\x1b[1m' + '⋀>⋁<'[s.orientation],
        CT.spawning: 'S',
        CT.level_exit: 'X',
        CT.alive: 'z',
        CT.crate: '%',
        CT.wall: '#',
    }

    if view_size:
        view_width = view_height = view_size
        x0, y0 = s.agent_loc - view_size // 2
        board = s.board.view(wrapping_array)[y0:y0+view_size, x0:x0+view_size]
        goals = s.goals.view(wrapping_array)[y0:y0+view_size, x0:x0+view_size]
    else:
        view_width, view_height = s.width, s.height
        board = s.board
        goals = s.goals
    screen = np.empty((view_height+2, view_width+3), dtype=object)
    screen[:] = ''
    screen[0] = screen[-1] = ' -'
    screen[:,0] = screen[:,-2] = ' |'
    screen[:,-1] = '\n'
    screen[0,0] = screen[0,-2] = screen[-1,0] = screen[-1,-2] = ' +'
    sub_screen = screen[1:-1,1:-2]
    sub_screen += '\x1b[48;5;175m ' * (goals < 0).astype(object)
    sub_screen += '\x1b[48;5;116m ' * (goals > 0).astype(object)
    sub_screen += '\x1b[48;5;7m ' * (goals == 0).astype(object)
    colors = board & CT.rainbow_color
    sub_screen += '\x1b[38;5;0m' * (colors == 0).astype(object)
    sub_screen += '\x1b[38;5;1m' * (colors == CT.color_r).astype(object)
    sub_screen += '\x1b[38;5;2m' * (colors == CT.color_g).astype(object)
    sub_screen += '\x1b[38;5;12m' * (colors == CT.color_b).astype(object)
    sub_screen += '\x1b[38;5;11m' * (colors == CT.color_r | CT.color_g).astype(object)
    sub_screen += '\x1b[38;5;39m' * (colors == CT.color_g | CT.color_b).astype(object)
    sub_screen += '\x1b[38;5;129m' * (colors == CT.color_r | CT.color_b).astype(object)
    sub_screen += '\x1b[38;5;8m' * (colors == CT.rainbow_color).astype(object)
    filled = np.zeros(sub_screen.shape, dtype=bool)
    for cell, sprite in SPRITES.items():
        # This isn't exactly fast, but oh well.
        has_sprite = ((board & cell) == cell) & ~filled
        filled |= has_sprite
        sub_screen[has_sprite] += sprite
    sub_screen[~filled] += ' '
    sub_screen += '\x1b[0m'
    # Clear the screen and move cursor to the start
    sys.stdout.write("\x1b[H\x1b[J")
    if s.title:
        print("\x1b[1m%s\x1b[0m" % s.title)
    sys.stdout.write("Score: \x1b[1m%i\x1b[0m\n" % s.points)
    sys.stdout.write("Steps: \x1b[1m%i\x1b[0m\n" % s.num_steps)
    if s.edit_mode:
        sys.stdout.write("\x1b[1m*** EDIT MODE ***\x1b[0m\n")
    sys.stdout.write(''.join(screen.ravel()))
    if s.error_msg:
        sys.stdout.write("\x1b[3m" + s.error_msg + "\x1b[0m\n")
    print(' '.join(s.log_actions))
    print(s._program)
    words = [COMMAND_WORDS.get(c, '_') for c in s.commands]
    sys.stdout.write("Command: " + ' '.join(words))
    sys.stdout.flush()


def play(game_state, view_size):
    os.system('clear')
    while not game.game_over:
        render(game_state, view_size)
        key = getch()
        if key == INTERRUPT_KEY:
            raise KeyboardInterrupt
        elif key == DELETE_KEY and game_state.commands:
            game_state.commands.pop()
        elif key == '`':
            # Toggle the paused status. This will allow the user to
            # add/destroy blocks without advancing the game's physics.
            game_state.edit_mode = not game_state.edit_mode
        elif game_state.edit_mode:
            # Execute action immediately.
            # Useful for building structures, etc.
            game_state.execute_edit(key)
        elif key in KEY_BINDINGS:
            game_state.step(KEY_BINDINGS[key])


def play_many(game_state, view_size, load_dir):
    import glob
    if load_dir is None:
        play(game_state, view_size)
        return
    elif os.path.isdir(load_dir):
        levels = sorted(glob.glob(os.path.join(load_dir, '*.npz')))
    else:
        levels = [load_dir]
    for level in levels:
        game_state.load(level)
        play(game_state, view_size)
    print("\n\nGame over!")
    print("\nFinal score:", game_state.points)
    print("Total steps:", game_state.num_steps, "\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--async')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--board', type=int, default=20, help="board size")
    parser.add_argument(
        '--view', type=int, default=0, help="View size. "
        "Defaults to zero, in which case the view fixed on the whole board.")
    parser.add_argument('--clear', action="store_true")
    parser.add_argument('--load', help="Load game state from file.")
    args = parser.parse_args()
    GameState.default_board_size = (args.board, args.board)
    if args.async:
        game = AsyncGame(args.async, 1/max(1e-6, args.temperature), clear_board=args.clear)
    else:
        game = GameOfLife(clear_board=args.clear)
    try:
        play_many(game, args.view, args.load)
    except KeyboardInterrupt:
        print("")
