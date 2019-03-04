"""
The syntax tree is used to build complicated 'spells' during game play.
Each new command gets pushed onto the tree, either adding to or filling out
its terminal node. Once a tree is complete, it can be executed on a given
game 'state'. The state only needs to define a few distinct functions:

    - execute_action(action) -> 0 or error
    - check(condition) -> boolean
    - define_subprogram(name, program) -> 0 or error
    - call_subprogram(name) -> 0 or error

The syntax tree will step through all of the commands, executing each one
on the state in turn. The state is then responsible for handling the actions
and all of their effects.
"""

COMMAND_COST = {
    "LEFT": 0,
    "RIGHT": 0,
    "UP": 1,
    "DOWN": 1,
    "NULL": 1,
    "TOGGLE": 1,
    "IFEMPTY": 0,
    "REPEAT": 0,
    "DEFINE": 0,
    "CALL": 1,
    "LOOP": 0,
    "CONTINUE": 1,
    "BREAK": 1,
    "BLOCK": 0,
}


class ProgramError(Exception):
    pass


class StatefulProgram(object):
    """
    Executes programs on a given game state.

    Programs are built up of individual commands which are added using the
    `add_command` method. Most commands will cause the game state to advance
    as soon as they're added to the command queue, but the queue only gets
    executed when the commands are syntactically complete.

    Attributes
    ----------
    max_executions : int
        Total number of actions or function calls that are allowed to be
        executed at once.
    excess_execution_penalty : float
        Reward penalty for exceeding `max_executions`.
    command_queue : list of strings
        The current commands that constitute the program.
    root : BlockNode
        The root node of the program which can be executed.
    subprograms : dictionary of commands to SyntaxNodes
        Subprograms can be bound to a command and saved for repeated use.
    action_log : list of command strings
        Executed actions from the last run of the program.
    message : str
        Log or error message from the last run of the program.
    game_state : GameState object
    """
    max_executions = 100
    excess_execution_penalty = -3

    def __init__(self, game_state):
        self.command_queue = []
        self.subprograms = {}
        self.action_log = []
        self.root = BlockNode()
        self.game_state = game_state
        self.message = ""

    def add_command(self, command):  # -> reward, steps
        # First advance the game board
        start_pts = self.game_state.current_points()
        if not self.game_state.game_over:
            num_steps = COMMAND_COST.get(command, 1)
            for _ in range(num_steps):
                self.game_state.advance_board()
        else:
            num_steps = 0

        self.command_queue.append(command)
        self.root.push(command)
        if not self.root.can_run():
            return 0, num_steps

        # Set a few temporary attributes which get called in execute_action,
        # but shouldn't be thought of as persisting.
        self._num_executions = 0
        self._reward = 0
        # Then set a couple of peristant variables as logs
        self.message = ""
        self.action_log.clear()
        try:
            self.root.execute(self)
        except ProgramError as err:
            self.message = err.args[0]

        # Reset the command queue
        self.command_queue.clear()
        self.root = BlockNode()

        end_pts = self.game_state.current_points()
        return (end_pts - start_pts) + self._reward, num_steps

    def pop_command(self):
        if self.command_queue:
            self.command_queue.pop()
        self.root = BlockNode()
        for command in self.command_queue:
            self.root.push(command)

    def increment_executions(self):
        self._num_executions += 1
        if self._num_executions > self.max_executions:
            self._reward += self.excess_execution_penalty
            raise ProgramError("You collapse from exhaustion.")

    def execute_action(self, action):
        self.increment_executions()
        self._reward += self.game_state.execute_action(action)
        self.action_log.append(action)

    def define_subprogram(self, name, program):
        self.subprograms[name] = program

    def call_subprogram(self, name):
        if not name:
            raise ProgramError("A name is missing...")
        if name not in self.subprograms:
            raise ProgramError("No such directive exists...")
        self.increment_executions()
        return self.subprograms[name].execute(self)


def make_new_node(val):
    node_classes = {
        "REPEAT": RepeatNode,
        "BLOCK": BlockNode,
        "IFEMPTY": IfNode,
        "LOOP": LoopNode,
        "DEFINE": DefineNode,
        "CALL": CallNode,
    }
    node_class = node_classes.get(val, ActionNode)
    return node_class(val)


class SyntaxNode(object):
    """
    Base class for building the parse tree.
    """
    can_push = True
    default_name = None

    def __init__(self, name=None):
        self.name = name or self.default_name

    def push(self, command):
        """
        Add a child command to this node of the tree.
        """
        raise NotImplementedError

    def execute(self, state):
        """
        Execute the program defined by this node and any sub-nodes.
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
        state.execute_action(self.name)

    def __str__(self):
        return "<%s>" % (self.name,)


class BlockNode(SyntaxNode):
    """
    Node to wrap multiple commands into a single block.

    Necessary for control flow with more than single command.
    Note that the `NULL` (or `BREAK` or `CONTINUE`) command must pushed onto
    the node in order to exit the block.
    """
    default_name = "BLOCK"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
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
            node.execute(state)

    def can_run(self):
        return self.list and not self.list[-1].can_push or not self.can_push

    def __str__(self):
        return str(self.list)


class DefineNode(SyntaxNode):
    """
    Binds a routine to a given name which can later be executed with `CallNode`.

    Note that the definition starts off in block mode.
    """
    default_name = "DEFINE"

    def __init__(self, name=None, var_name=None):
        self.name = name or self.default_name
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
        state.define_subprogram(self.var_name, self.command)

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
        state.call_subprogram(self.var_name)

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
            raise ProgramError("A command is missing...")
        self.command.execute(state)
        self.command.execute(state)

    def __str__(self):
        return "<REPEAT %s>" % (self.command,)


class LoopNode(BlockNode):
    """
    Loops the following commands.

    The loop is broken with either `NULL` or `BREAK`, and repeated with
    `CONTINUE`.
    """
    def execute(self, state):
        while True:
            for node in self.list:
                node.execute(state)
                if node.name == "CONTINUE":
                    break
                if node.name == "BREAK":
                    return
            else:
                # Add an implicit BREAK at the end of the loop
                state.increment_executions()
                return

    def __str__(self):
        return "<LOOP %s>" % (self.list)


class IfNode(SyntaxNode):
    """
    Class for branching nodes.

    During execution calls `state.check(self.name)` to see which branch it
    should take.
    """
    default_name = "IF"
    yes_node = None
    no_node = None

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
            raise ProgramError("A command is missing...")
        elif state.check(self.name):
            self.yes_node.execute(state)
        else:
            self.no_node.execute(state)

    def __str__(self):
        return "<%s %s %s>" % (self.name, self.yes_node, self.no_node)
