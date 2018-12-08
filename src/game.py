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
        CHECK [action]
        REPEAT [command]
        BLOCK [commands] NULL
        IFSUCCESS [command 1] [command 2]
        LOOP [commands] NULL/BREAK/CONTINUE
        DEFINE [name] [command]
        CALL [name]

All spells are cast the moment they're uttered and syntactically complete;
there's no "finish spell" action. This makes it much easier to interact with
the world, as many actions will have an immediate effect. In order chain
actions together they must either be part of a `REPEAT`, `BLOCK`, or `LOOP`.

The `IFSUCCESS` statement acts as the standard ternary operator, using the boolean
success of the previous action as input. The `CHECK` operator does a dry run of
an action, returning its would-be success value without causing other effects.

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
- `CHECK` implies `IF...` (it's useless without it)
- `IF...` should *probably* imply one of the block constructs, although it's
  possible to use it effectively with just `REPEAT`
- `LOOP` generally implies `IF...`
"""

import os
import sys
import numpy as np
import scipy.signal

from getch import getch
from array_utils import wrapping_array
from gen_board import gen_board

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

OBJECT_TYPES = {
    'empty': 0,
    'agent': 1,
    'block': 2,
    'wall': 3,
    'spawner': 5,
}
OBJECT_TYPES = {k: np.array(v, dtype=np.int8) for k,v in OBJECT_TYPES.items()}
SPAWN_PROB = 0.3


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
    "WALL",  # create a wall object instead of a block
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
    'q': "ABSORB",
    'c': "CREATE",
    'x': "DESTROY",
    'e': "WALL",
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
    return scipy.signal.convolve2d(*args, boundary='wrap', mode='same', **kw)


def make_new_node(val):
    if val in ACTIONS:
        node_class = ActionNode
    else:
        node_class = {
            "CHECK": CheckNode,
            "REPEAT": RepeatNode,
            "BLOCK": BlockNode,
            "IFSUCCESS": IfSuccessNode,
            "IFEMPTY": IfEmptyNode,
            "LOOP": LoopNode,
            "CONTINUE": ContinueNode,
            "BREAK": BreakNode,
            "DEFINE": DefineNode,
            "CALL": CallNode,
        }[val]
    return node_class(val)


class SyntaxNode(object):
    """
    Base class for building the parse tree.
    """
    can_push = True
    default_name = None

    def __init__(self, name=default_name):
        self.name = name

    def push(self, command):
        """
        Add a child command to this node of the tree.
        """
        raise NotImplementedError

    def execute(self, state):
        """
        Execute the program defined by this node and any sub-nodes.

        The `state` object should contain an instance of `GameState`, defined
        below.
        """
        raise NotImplementedError

    def __str__(self):
        return "<%s>" % (self.name,)

    def __repr__(self):
        return str(self)


class ActionNode(SyntaxNode):
    """
    Simple node that executes a single parameter-less action in the game state.
    Has no sub-nodes.
    """
    can_push = False

    def execute(self, state):
        return state.execute_action(self.name)

    def __str__(self):
        return "<%s>" % (self.name,)


class CheckNode(SyntaxNode):
    """
    Node that wraps the subsequent action in a dry run.

    This can be used to set the state's `last_action_bool` flag without
    actually executing the action. That flag can then be queried by an if node.
    """
    action = None

    def push(self, val):
        self.action = val
        self.can_push = False

    def execute(self, state):
        return state.execute_action(self.action, dry_run=True)

    def __str__(self):
        return "<CHECK %s>" % (self.action,)


class BlockNode(SyntaxNode):
    """
    Node to wrap multiple commands into a single block.

    Necessary for control flow with more than single command.
    Note that the `NULL` (or `BREAK` or `CONTINUE`) command must pushed onto
    the node in order to exit the block.
    """
    def __init__(self, name="BLOCK"):
        self.list = []

    def push(self, val):
        if self.list and self.list[-1].can_push:
            self.list[-1].push(val)
        elif val == 'NULL':
            self.can_push = False
        else:
            self.list.append(make_new_node(val))
            if val in ('BREAK', 'CONTINUE'):
                self.can_push = False

    def execute(self, state):
        for node in self.list:
            err = node.execute(state)
            if err:
                return err
        return 0

    def __str__(self):
        return str(self.list)


class DefineNode(SyntaxNode):
    """
    Binds a routine to a given name which can later be executed with `CallNode`.

    Note that the definition starts off in block mode.
    """
    def __init__(self, name, var_name=None):
        self.name = name
        self.var_name = var_name
        self.command = BlockNode()

    def push(self, val):
        if self.var_name is None:
            self.var_name = val
        elif self.command is None:
            self.command = make_new_node(val)
            self.can_push = self.command.can_push
        else:
            self.command.push(val)
            self.can_push = self.command.can_push

    def execute(self, state):
        if self.var_name is None:
            return "A name is missing..."
        state.saved_commands[self.var_name] = self.command
        return 0

    def __str__(self):
        return "<DEFINE %s %s>" % (self.var_name, self.command)


class CallNode(SyntaxNode):
    """
    Executes a routine previously defined with `DefineNode`.
    """
    var_name = None

    def push(self, val):
        self.var_name = val
        self.can_push = False

    def execute(self, state):
        state.energy -= 1
        if self.var_name is None:
            return "A name is missing..."
        elif state.energy < 0:
            return state.out_of_energy_msg
        elif self.var_name in state.saved_commands:
            return state.saved_commands[self.var_name].execute(state)
        else:
            return "'%s' has not been bound..." % \
                COMMAND_WORDS.get(self.var_name, self.var_name)

    def __str__(self):
        return "<CALL %s>" % (self.var_name,)


class RepeatNode(SyntaxNode):
    """
    Repeats the subsequent command upon execution.
    """
    default_name = "REPEAT"
    command = None

    def push(self, val):
        if self.command is None:
            self.command = make_new_node(val)
        else:
            self.command.push(val)
        self.can_push = self.command.can_push

    def execute(self, state):
        if self.command is None:
            return "A command is missing..."
        return self.command.execute(state) or self.command.execute(state)

    def __str__(self):
        return "<REPEAT %s>" % (self.command,)


class ContinueNode(SyntaxNode):
    default_name = "CONTINUE"
    flag = 1
    can_push = False

    def execute(self, state):
        state.energy -= 1
        if state.energy < 0:
            return state.out_of_energy_msg
        return self.flag


class BreakNode(ContinueNode):
    DEFAULT_NAME = "BREAK"
    flag = 2


class LoopNode(BlockNode):
    """
    Loops the following commands.

    The loop is broken with either `NULL` or `BREAK`, and repeated with
    `CONTINUE`.
    """
    def execute(self, state):
        while True:
            for node in self.list:
                err = node.execute(state)
                if err == ContinueNode.flag:
                    break
                if err == BreakNode.flag:
                    return 0
                if err:
                    return err
            else:
                # Add an implicit BREAK at the end of the loop
                return 0

    def __str__(self):
        return "<LOOP %s>" % (self.list)


class IfNode(SyntaxNode):
    """
    Base class for branching nodes.
    """
    default_name = "IF"
    yes_node = None
    no_node = None

    def get_bool(self, state):
        raise NotImplementedError

    def push(self, val):
        if self.yes_node is None:
            self.yes_node = make_new_node(val)
        elif self.yes_node.can_push:
            self.yes_node.push(val)
        elif self.no_node is None:
            self.no_node = make_new_node(val)
            self.can_push = self.no_node.can_push
        else:
            self.no_node.push(val)
            self.can_push = self.no_node.can_push

    def execute(self, state):
        if self.yes_node is None or self.no_node is None:
            return "A command is missing..."
        elif self.get_bool(state):
            return self.yes_node.execute(state)
        else:
            return self.no_node.execute(state)

    def __str__(self):
        return "<%s %s %s>" % (self.default_name, self.yes_node, self.no_node)


class IfSuccessNode(IfNode):
    """
    Branches conditioned on the state's `last_action_bool` flag.
    """
    default_name = "IFSUCCESS"

    def get_bool(self, state):
        return state.last_action_bool


class IfEmptyNode(IfNode):
    """
    Branches conditioned on the status of the position in front of the agent.
    """
    default_name = "IFEMPTY"

    def get_bool(self, state):
        x, y = state.relative_loc(1)
        return state.board[y, x] == OBJECT_TYPES['empty']


class GameState(object):
    """
    Defines the game state and dynamics. Does NOT define rendering.
    """
    width = 20
    height = 20
    out_of_energy_msg = "You collapse from exhaustion."
    num_steps = 0
    default_energy = 100
    edit_mode = False

    def __init__(self, clear_board=False):
        self.agent_loc = np.array([0,0])
        self.orientation = 0  # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

        self.points = 0
        self.smooth_points = 0
        self.color = 1
        self.error_msg = None

        self.commands = []
        self.log_actions = []  # for debugging only
        self.saved_commands = {}
        self.last_action_bool = False
        if clear_board:
            self.board = np.zeros((self.height, self.width), dtype=np.int8)
            self.goals = self.board.copy()
            self.board[0,0] = OBJECT_TYPES['agent']
        else:
            self.make_board()

    def make_board(self):
        self.board = gen_board(
            board_size=(self.height, self.width),
            min_total=self.width * self.height // 15,
            num_seeds=self.width * self.height // 100,
        ) * OBJECT_TYPES['block']
        x, y = self.agent_loc
        self.board[y, x] = OBJECT_TYPES['agent']

        walls = (np.random.random(self.board.shape) < 0.1)
        self.board += (self.board == 0) * walls * OBJECT_TYPES['wall']
        self.goals = np.zeros_like(self.board)
        self.goals += np.random.random(self.board.shape) < 0.1
        self.goals -= np.random.random(self.board.shape) < 0.05
        self.goals *= self.board == OBJECT_TYPES['empty']

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

    def move_agent(self, x, y):
        """
        Move the agent to a new location if that location is empty.
        """
        if self.board[y, x] == OBJECT_TYPES['empty']:
            self.board[self.agent_loc[1], self.agent_loc[0]] = 0
            self.board[y, x] = OBJECT_TYPES['agent']
            self.agent_loc = np.array([x, y])
            self.last_action_bool = True
        else:
            self.last_action_bool = False

    def execute_action(self, action, dry_run=False):
        """
        Execute an individual action.

        Either returns 0 or an error message.
        """
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
            self.move_agent(*self.relative_loc(1))
        elif action == "BACKWARD":
            self.move_agent(*self.relative_loc(-1))
        elif action == "CREATE":
            x, y = self.relative_loc(1)
            if self.board[y, x] == OBJECT_TYPES['empty']:
                self.last_action_bool = True
                self.board[y, x] = OBJECT_TYPES['block']
            else:
                self.last_action_bool = False
        elif action == "DESTROY":
            x, y = self.relative_loc(1)
            if self.board[y, x] == OBJECT_TYPES['block']:
                self.last_action_bool = True
                self.board[y, x] = OBJECT_TYPES['empty']
            else:
                self.last_action_bool = False

        # placeholder
        self.log_actions.append(action)
        return 0

    def execute_edit(self, key):
        """
        Like execute action, but allows for more powerful actions.
        """
        cell_types = {
            'x': OBJECT_TYPES['empty'],
            'c': OBJECT_TYPES['block'],
            'w': OBJECT_TYPES['wall'],
            's': OBJECT_TYPES['spawner'],
        }
        x, y = self.relative_loc(1)
        if key == LEFT_ARROW_KEY:
            self.orientation -= 1
            self.orientation %= 4
        elif key == RIGHT_ARROW_KEY:
            self.orientation += 1
            self.orientation %= 4
        elif key == UP_ARROW_KEY:
            self.move_agent(*self.relative_loc(1))
        elif key == DOWN_ARROW_KEY:
            self.move_agent(*self.relative_loc(-1))
        elif key == 'g':
            # Toggle the goal state
            self.goals[y, x] += 2
            self.goals[y, x] %= 3
            self.goals[y, x] -= 1
        elif key in cell_types:
            self.board[y, x] = cell_types[key]

    def advance_board(self):
        """
        Apply one timestep of physics, and return change in points.
        """
        raise NotImplementedError

    _program = None  # for debugging / logging

    def step(self, action):
        assert action in COMMAND_WORDS
        self.commands.append(action)
        # It's somewhat inefficient to rebuild the program from scratch
        # when each action is added, but otherwise we'd have to handle
        # popping commands when the delete key is hit. Hardly a bottleneck.
        program = BlockNode()
        for command in self.commands:
            program.push(command)
        self._program = program  # for debugging
        points = self.advance_board()
        self.points = points
        self.smooth_points = 0.95 * self.smooth_points + 0.05 * points
        self.num_steps += 1
        if not program.list or not program.list[-1].can_push:
            # Reached the end of the list.
            self.log_actions = []
            self.energy = self.default_energy
            err = program.execute(self)
            self.error_msg = "" if not err or err in (1,2) else err
            self.commands = []


class GameOfLife(GameState):
    def advance_board(self):
        """
        Apply one timestep of physics using Game of Life rules.
        """
        # We can advance the board using a pretty simple convolution.
        blocks = self.board == OBJECT_TYPES['block']
        cfilter = np.array([[2,2,2],[2,1,2],[2,2,2]])
        new_blocks = (np.abs(convolve2d(blocks, cfilter) - 6) <= 1)
        # Randomly add blocks around the spawners.
        spawners = 0.5 * convolve2d(self.board == OBJECT_TYPES['spawner'], cfilter)
        new_blocks |= np.random.random(self.board.shape) > (1 - SPAWN_PROB)**spawners
        # But don't change the board next to the agent or where there's a wall
        freezers = self.board == OBJECT_TYPES['agent']
        frozen = (self.board % 2 == 1) | (convolve2d(freezers, cfilter) > 0)
        self.board *= frozen
        self.board += (new_blocks & ~frozen) * OBJECT_TYPES['block']

        blocks = self.board == OBJECT_TYPES['block']
        reward = np.sum(blocks * self.goals)
        return reward


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
        EPOCHS = 0.1

        board = self.board
        rules = self.rules
        w = self.width
        h = self.height
        AGENT = OBJECT_TYPES['agent']
        EMPTY = OBJECT_TYPES['empty']
        BLOCK = OBJECT_TYPES['block']
        WALL = OBJECT_TYPES['wall']
        SPAWNER = OBJECT_TYPES['spawner']
        spawn = convolve2d(self.board == SPAWNER, np.ones((3,3)))
        for _ in range(int(board.size * EPOCHS)):
            x = np.random.randint(w)
            y = np.random.randint(h)
            if board[y, x] in (AGENT, WALL, SPAWNER):
                continue
            xp = (x+1) % w
            xn = (x-1) % w
            yp = (y+1) % h
            yn = (y-1) % h
            neighbors = 0
            neighbors += board[y, xp] == BLOCK
            neighbors += board[y, xn] == BLOCK
            neighbors += board[yn, x] == BLOCK
            neighbors += board[yp, x] == BLOCK
            if rules[0] > 4:
                neighbors += board[yn, xp] == BLOCK
                neighbors += board[yp, xn] == BLOCK
            if rules[0] > 6:
                neighbors += board[yn, xn] == BLOCK
                neighbors += board[yp, xp] == BLOCK
            if board[y, x] == BLOCK:
                H = rules[1][neighbors]
            else:
                H = rules[2][neighbors]
            P = 0.5 + 0.5*np.tanh(H * self.beta)
            P = 1 - (1-P)*(1-SPAWN_PROB)**spawn[y, x]
            board[y, x] = BLOCK if P > np.random.random() else EMPTY

        blocks = self.board == OBJECT_TYPES['block']
        reward = np.sum(blocks * self.goals)
        return reward


def render(s, view_size):
    """
    Renders the game state `s`.

    This is not exactly a speedy rendering system, but it should be plenty
    fast enough for our purposes.
    """
    SPRITES = {
        'agent': ['\x1b[1m%s\x1b[0m' % {
            0: "⋀",
            1: ">",
            2: "⋁",
            3: "<",
        }[s.orientation]] * 3,
        'empty': ('\x1b[31m*\x1b[0m', ' ', '\x1b[34m*\x1b[0m'),
        'wall': ('#', '#', '#'),
        'block': ('\x1b[31mz\x1b[0m', 'z', '\x1b[34mz\x1b[0m'),
        'spawner': ('S', 'S', 'S'),
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
    screen[:] = ' '
    screen[0] = screen[-1] = '-'
    screen[:,0] = screen[:,-2] = '|'
    screen[:,-1] = '\n'
    screen[0,0] = screen[0,-2] = screen[-1,0] = screen[-1,-2] = '+'
    sub_screen = screen[1:-1,1:-2]
    for key, sprite in SPRITES.items():
        sub_screen[(board == OBJECT_TYPES[key]) & (goals < 0)] = sprite[0]
        sub_screen[(board == OBJECT_TYPES[key]) & (goals == 0)] = sprite[1]
        sub_screen[(board == OBJECT_TYPES[key]) & (goals > 0)] = sprite[2]
    # Clear the screen and move cursor to the start
    sys.stdout.write("\x1b[H\x1b[J")
    sys.stdout.write("Score: \x1b[1m%i\x1b[0m\n" % s.points)
    sys.stdout.write("Smoothed: \x1b[1m%0.3f\x1b[0m\n " % s.smooth_points)
    if s.edit_mode:
        sys.stdout.write("\x1b[1m*** EDIT MODE ***\x1b[0m\n")
    sys.stdout.write(' '.join(screen.ravel()))
    if s.error_msg:
        sys.stdout.write("\x1b[3m" + s.error_msg + "\x1b[0m\n")
    print(' '.join(s.log_actions))
    print(s._program)
    words = [COMMAND_WORDS.get(c, '_') for c in s.commands]
    sys.stdout.write("Command: " + ' '.join(words))
    sys.stdout.flush()


def play(game_state, view_size):
    os.system('clear')
    while True:
        render(game_state, view_size)
        key = getch()
        if key == INTERRUPT_KEY:
            print("")
            return
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--async')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--board', type=int, default=20, help="board size")
    parser.add_argument('--view', type=int, default=0, help=
        "view size. Defaults to zero, in which case the view isn't centered.")
    parser.add_argument('--clear', action="store_true")
    args = parser.parse_args()
    if args.async:
        AsyncGame.width = AsyncGame.height = args.board
        game = AsyncGame(args.async, 1/max(1e-6, args.temperature), clear_board=args.clear)
    else:
        GameOfLife.width = GameOfLife.height = args.board
        game = GameOfLife(clear_board=args.clear)
    play(game, args.view)
