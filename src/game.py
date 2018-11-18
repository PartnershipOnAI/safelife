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

    LEFT
    RIGHT
    FORWARD
    BACK
    END
    ABSORB [direction]
    CREATE [direction]
    PUSH [direction]
    REPEAT [command] END
    IF [direction] [command 1] END [command 2] END
    LOOP [command] END
    CONTINUE
    BREAK
    DEFINE [name] [command] END
    CALL [name]

The `END` statements aren't strictly necessary. If they're not present,
they'll implicitly be appended to the end of the command string. So, for
example, `REPEAT LEFT REPEAT UP` would be the same as
`REPEAT LEFT REPEAT UP END END` and go `LEFT UP UP LEFT UP UP`.

The `IF` statement checks to see if the absorbed color matches the color of
the tile.

The `LOOP` statement doesn't have a conditional. Instead, it must be exited
with a `BREAK` statement. The end of the loop can be `END`, `BREAK`, or
`CONTINUE`. The behavior of `END` is the same as `BREAK` in this context
(although we could easily switch it). `END` can also be used as a direction,
in which case it's the "stand still" direction.

The `DEFINE` statement defines a reusable procedure with any name. The `CALL`
statement then calls a procedure previously stored under a given name.
Procedures can be called recursively, to some limit. Note that the name scopes
are global.
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


def make_new_node(val):
    node_classes = {
        "LEFT": MoveNode,
        "RIGHT": MoveNode,
        "FORWARD": MoveNode,
        "BACKWARD": MoveNode,
        "ABSORB": ActionNode,
        "CREATE": ActionNode,
        "PUSH": ActionNode,
        "REPEAT": RepeatNode,
        "IF": IfNode,
        "LOOP": LoopNode,
        "CONTINUE": ContinueNode,
        "BREAK": BreakNode,
        "DEFINE": DefineNode,
        "CALL": CallNode,
    }
    assert val in node_classes, val
    return node_classes[val](val)


class SyntaxNode(object):
    can_push = True

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return str(self)


class ActionNode(SyntaxNode):
    direction = None

    def push(self, val):
        self.direction = val
        self.can_push = False

    def execute(self, state):
        if self.direction is None:
            return "A direction is missing..."
        return state.execute_action(self.name, self.direction)

    def __str__(self):
        return "<%s %s>" % (self.name, self.direction)


class MoveNode(ActionNode):
    def __init__(self, direction):
        self.name = "MOVE"
        self.direction = direction
        self.can_push = False


class BlockNode(SyntaxNode):
    def __init__(self, name="BLOCK"):
        self.list = []

    def push(self, val):
        if self.list and self.list[-1].can_push:
            self.list[-1].push(val)
        elif val == 'END':
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
        self.command = BlockNode()

    def push(self, val):
        if self.var_name is None:
            self.var_name = val
        else:
            self.command.push(val)
            self.can_push = self.command.can_push

    def execute(self, state):
        if self.var_name is None:
            return "A name is missing..."
        state.defined_commands[self.var_name] = self.command
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
        elif self.var_name in state.defined_commands:
            return state.defined_commands[self.var_name].execute(state)
        else:
            return "'%s' has not been bound..." % state.command_to_word.get(self.var_name)

    def __str__(self):
        return "<CALL %s>" % (self.var_name,)


class RepeatNode(SyntaxNode):
    def __init__(self, name="REPEAT"):
        self.name = name
        self.command = BlockNode()

    def push(self, val):
        self.command.push(val)
        self.can_push = self.command.can_push

    def execute(self, state):
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


class IfNode(SyntaxNode):
    def __init__(self, name="IF"):
        self.name = name
        self.cond = None
        self.yes_node = BlockNode()
        self.no_node = BlockNode()

    def push(self, val):
        if self.cond is None:
            self.cond = val
        elif self.yes_node.can_push:
            self.yes_node.push(val)
        else:
            self.no_node.push(val)
            self.can_push = self.no_node.can_push

    def execute(self, state):
        if state.check_direction(self.cond):
            return self.yes_node.execute(state)
        else:
            return self.no_node.execute(state)

    def __str__(self):
        return "<IF %s %s %s>" % (self.cond, self.yes_node, self.no_node)


class GameState(object):
    width = 20
    height = 20
    out_of_energy_msg = "You collapse from exhaustion."

    def __init__(self):
        self.agent_loc = np.array([0,0])
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)
        self.commands = []
        self.score = 0
        self.color = 1
        self.board[self.agent_loc[0], self.agent_loc[1]] = OBJECT_TYPES['agent']
        self.defined_commands = {}
        self.error_msg = None
        self.command_key = {
            # later we'll want to randomize this
            'a': "LEFT",
            'd': "RIGHT",
            's': "BACKWARD",
            'w': "FORWARD",
            'z': "END",
            'q': "ABSORB",
            'c': "CREATE",
            'f': "IF",
            'r': "REPEAT",
            'p': "DEFINE",
            'o': "CALL",
            'l': "LOOP",
            'u': "CONTINUE",
            'b': "BREAK",
        }
        self.command_to_word = {
            v: MAGIC_WORDS[k] for k, v in self.command_key.items()
        }
        self.log_msg = ""

    def move_agent(self, dx, dy):
        new_loc = self.agent_loc + [dx, dy]
        new_loc %= [self.width, self.height]
        self.board[self.agent_loc[1], self.agent_loc[0]] = 0
        self.board[new_loc[1], new_loc[0]] = OBJECT_TYPES['agent']
        self.agent_loc = new_loc

    def execute_commands(self):
        self.log_msg = ""
        command_tree = BlockNode("root")
        self.error_msg = ""
        for command in self.commands:
            if command_tree.can_push:
                command_tree.push(self.command_key[command])
        self.energy = 25
        err = command_tree.execute(self)
        self.error_msg = "" if not err or err in (1,2) else err
        self.commands = []

        # placeholder
        delta_score = 5
        self.score += delta_score
        return delta_score

    def execute_action(self, name, direction):
        if direction not in ('LEFT', 'RIGHT', 'FORWARD', 'BACKWARD', 'END'):
            return "'%s' is not a direction..." % self.command_to_word.get(direction)
        self.energy -= 1
        if self.energy < 0:
            return self.out_of_energy_msg
        self.log_msg += "%s %s " % (name, direction)
        return 0  # placeholder

    def check_direction(self, direction):
        if direction not in ('LEFT', 'RIGHT', 'FORWARD', 'BACKWARD', 'END'):
            return "'%s' is not a direction..." % self.command_to_word.get(direction)
        return True  # placeholder

    def step(self, action):
        # Returns change in score
        if action in '\r\n':
            # enter key
            return self.execute_commands()
        elif action.startswith('\x1b'):
            # arrow key
            dx = (action == RIGHT_ARROW_KEY) - (action == LEFT_ARROW_KEY)
            dy = (action == DOWN_ARROW_KEY) - (action == UP_ARROW_KEY)
            self.move_agent(dx, dy)
        elif action == '\x7f':
            # delete key
            self.commands = self.commands[:-1]
        elif action in self.command_key:
            self.commands.append(action)
        return 0


def render(s):
    # This is not exactly a speedy rendering system, but oh well!
    SPRITES = {
        'agent': '\x1b[1m@\x1b[0m',
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
    sys.stdout.write(s.log_msg + '\n')
    words = [MAGIC_WORDS.get(c, '_') for c in s.commands]
    sys.stdout.write("Command: " + ' '.join(words))
    sys.stdout.flush()


def main():
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
    main()
