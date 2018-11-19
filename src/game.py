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
        IFTHEN [command 1] [command 2]
        LOOP [commands] NULL/BREAK/CONTINUE
        DEFINE [name] [command]
        CALL [name]

All spells are cast the moment they're uttered and syntactically complete;
there's no "finish spell" action. This makes it much easier to interact with
the world, as many actions will have an immediate effect. In order chain
actions together they must either be part of a `REPEAT`, `BLOCK`, or `LOOP`.

The `IFTHEN` statement acts as the standard ternary operator, using the boolean
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
- `CHECK` implies `IFTHEN` (it's useless without it)
- `IFTHEN` should *probably* imply one of the block constructs
- `LOOP` generally implies `IFTHEN`
"""

import os
import sys
import numpy as np

from getch import getch

UP_ARROW_KEY = '\x1b[A'
DOWN_ARROW_KEY = '\x1b[B'
RIGHT_ARROW_KEY = '\x1b[C'
LEFT_ARROW_KEY = '\x1b[D'
INTERRUPT_KEY = '\x03'

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
    'agent': 1,
    'wall': 2,
}

ACTIONS = {
    "LEFT",
    "RIGHT",
    "FORWARD",
    "BACKWARD",
    "PUSH",
    "PULL",
    "ABSORB",
    "CREATE",
    "DESTROY",
    "NULL",
}


def make_new_node(val):
    if val in ACTIONS:
        node_class = ActionNode
    else:
        node_class = {
            "CHECK": CheckNode,
            "REPEAT": RepeatNode,
            "BLOCK": BlockNode,
            "IFTHEN": IfThenNode,
            "LOOP": LoopNode,
            "CONTINUE": ContinueNode,
            "BREAK": BreakNode,
            "DEFINE": DefineNode,
            "CALL": CallNode,
        }[val]
    return node_class(val)


class SyntaxNode(object):
    can_push = True

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return str(self)


class ActionNode(SyntaxNode):
    can_push = False

    def execute(self, state):
        return state.execute_action(self.name)

    def __str__(self):
        return "<%s>" % (self.name,)


class CheckNode(SyntaxNode):
    action = None

    def push(self, val):
        self.action = val
        self.can_push = False

    def execute(self, state):
        return state.execute_action(self.action, dry_run=True)

    def __str__(self):
        return "<CHECK %s>" % (self.action,)


class BlockNode(SyntaxNode):
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
    def __init__(self, name, var_name=None):
        self.name = name
        self.var_name = var_name
        self.command = None

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
            return "'%s' has not been bound..." % state.command_to_word.get(self.var_name)

    def __str__(self):
        return "<CALL %s>" % (self.var_name,)


class RepeatNode(SyntaxNode):
    def __init__(self, name="REPEAT"):
        self.name = name
        self.command = None

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
    flag = 1
    can_push = False

    def execute(self, state):
        state.energy -= 1
        if state.energy < 0:
            return state.out_of_energy_msg
        return self.flag

    def __str__(self):
        return "<CONTINUE>"


class BreakNode(ContinueNode):
    flag = 2

    def __str__(self):
        return "<BREAK>"


class LoopNode(BlockNode):
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


class IfThenNode(SyntaxNode):
    def __init__(self, name="IFTHEN"):
        self.name = name
        self.yes_node = None
        self.no_node = None

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
        elif state.last_action_bool:
            return self.yes_node.execute(state)
        else:
            return self.no_node.execute(state)

    def __str__(self):
        return "<IFTHEN %s %s>" % (self.yes_node, self.no_node)


class GameState(object):
    width = 20
    height = 20
    out_of_energy_msg = "You collapse from exhaustion."

    def __init__(self):
        self.agent_loc = np.array([0,0])
        self.orientation = 0  # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)

        self.score = 0
        self.color = 1
        self.board[self.agent_loc[0], self.agent_loc[1]] = OBJECT_TYPES['agent']
        self.error_msg = None
        self.command_key = {
            # later we'll want to randomize this
            'a': "LEFT",
            'd': "RIGHT",
            's': "BACKWARD",
            'w': "FORWARD",
            'z': "NULL",
            'q': "ABSORB",
            'c': "CREATE",
            'i': "IFTHEN",
            'r': "REPEAT",
            'p': "DEFINE",
            'o': "CALL",
            'l': "LOOP",
            'u': "CONTINUE",
            'b': "BREAK",
            'k': "BLOCK",
        }
        self.command_to_word = {
            v: MAGIC_WORDS[k] for k, v in self.command_key.items()
        }

        self.commands = []
        self.log_actions = []  # for debugging only
        self.saved_commands = {}
        self.last_action_bool = False

    def move_agent(self, dx, dy):
        new_loc = self.agent_loc + [dx, dy]
        new_loc %= [self.width, self.height]
        self.board[self.agent_loc[1], self.agent_loc[0]] = 0
        self.board[new_loc[1], new_loc[0]] = OBJECT_TYPES['agent']
        self.agent_loc = new_loc

    def execute_action(self, action, dry_run=False):
        self.energy -= 1
        if self.energy < 0:
            return self.out_of_energy_msg
        if action == "LEFT":
            self.orientation -= 1
            self.orientation %= 4
        elif action == "RIGHT":
            self.orientation += 1
            self.orientation %= 4

        # placeholder
        self.log_actions.append(action)
        self.last_action_bool = True
        return 0

    _program = None

    def step(self, action):
        if action == '\x7f':
            # Delete key
            if self.commands:
                self.commands.pop()
        elif action in self.command_key:
            # It's somewhat inefficient to rebuild the program from scratch
            # when each action is added, but otherwise we'd have to handle
            # popping commands when the delete key is hit. Hardly a bottleneck.
            self.commands.append(action)
            program = BlockNode()
            for command in self.commands:
                program.push(self.command_key[command])
            self._program = program
            if not program.list[-1].can_push:
                # Reached the end of the list.
                self.log_actions = []
                self.energy = 25
                err = program.execute(self)
                self.error_msg = "" if not err or err in (1,2) else err
                self.commands = []
            # Then need to evolve the board one step
        return 0


def render(s):
    # This is not exactly a speedy rendering system, but oh well!

    SPRITES = {
        'agent': '\x1b[1m%s\x1b[0m' % {
            0: "⋀",
            1: ">",
            2: "⋁",
            3: "<",
        }[s.orientation],
        'wall': '#',
    }
    screen = np.empty((s.height+2, s.width+3), dtype=object)
    screen[:] = ' '
    screen[0] = screen[-1] = '-'
    screen[:,0] = screen[:,-2] = '|'
    screen[:,-1] = '\n'
    screen[0,0] = screen[0,-2] = screen[-1,0] = screen[-1,-2] = '+'
    sub_screen = screen[1:-1,1:-2]
    for key, sprite in SPRITES.items():
        sub_screen[s.board == OBJECT_TYPES[key]] = sprite
    # Clear the screen and move cursor to the start
    sys.stdout.write("\x1b[H\x1b[J")
    sys.stdout.write("Score: \x1b[1m%i\x1b[0m\n " % s.score)
    sys.stdout.write(' '.join(screen.ravel()))
    if s.error_msg:
        sys.stdout.write("\x1b[3m" + s.error_msg + "\x1b[0m\n")
    print(' '.join(s.log_actions))
    print(s._program)
    words = [MAGIC_WORDS.get(c, '_') for c in s.commands]
    sys.stdout.write("Command: " + ' '.join(words))
    sys.stdout.flush()


def play():
    os.system('clear')
    game_state = GameState()
    while True:
        render(game_state)
        action = getch()
        if action == INTERRUPT_KEY:
            print("")
            return
        game_state.step(action)


if __name__ == "__main__":
    play()
