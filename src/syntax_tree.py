"""
The syntax tree is used to build complicated 'spells' during gameplay.
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


def make_new_node(val):
    node_classes = {
        "REPEAT": RepeatNode,
        "BLOCK": BlockNode,
        "IFEMPTY": IfNode,
        "LOOP": LoopNode,
        "CONTINUE": ContinueNode,
        "BREAK": BreakNode,
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

        Returns 0 on success. Other return values can either be flags
        to execute loops (e.g., break or continue), or errors that errors
        that propagate upwards.
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
        return state.define_subprogram(self.var_name, self.command)

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
        return state.call_subprogram(self.var_name)

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
        error = state.execute_action("CONTINUE")
        return error or self.flag


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
            return "A command is missing..."
        elif state.check(self.name):
            return self.yes_node.execute(state)
        else:
            return self.no_node.execute(state)

    def __str__(self):
        return "<%s %s %s>" % (self.name, self.yes_node, self.no_node)
