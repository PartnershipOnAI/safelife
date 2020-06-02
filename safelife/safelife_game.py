"""
Rules for the game environment(s) and the basic player actions.

All environments are grid-based cellular automata with discrete evolution.
Agents can perform actions within/upon the environments outside of an
evolutionary step. Therefore, an environment is primarily responsible for
three things:

1. maintaining environment state (and saving and loading it);
2. advancing the environment one step;
3. executing agent actions and edits.
"""

import os
from importlib import import_module

import numpy as np

from .helper_utils import (
    wrapping_array,
    wrapped_convolution as convolve2d,
)
from .random import coinflip, get_rng
from .speedups import advance_board


ORIENTATION = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "FORWARD": 4,
    "BACKWARD": 6,
}


class CellTypes(object):
    """
    Collection of flags for cellular automata.

    Attributes
    ----------
    alive
        Every cell in the cellular automata system can either be 'alive' or
        'dead'. At each evolutionary step live cells can be converted to dead
        cells and vice versa depending on the presence of neighboring live cells.
    movable
        Marks cells that can be pushed by the agent.
    destructible
        Marks cells that can be destroyed by the agent.
    frozen
        Frozen cells do not change during each evolutionary step.
        They won't become alive if they're dead, and they won't die if they're
        alive.
    preserving
        Preserving cells prevent all of their neighbors from dying.
    inhibiting
        Inhibiting cells prevent all neighbors from being born.
    spawning
        Spawning cells randomly create new living cells as their neighbors.
    color_(rgb)
        Every cell can have one of three color flags, for a total of 8 possible
        colors. New cells typically take on the color attributes of the cells
        that created them.
    agent
        Special flag to mark the cell as being occupied by an agent.
        Mostly used for rendering (both to humans and machines), as the actual
        location of the agent is stored separately.
    exit
        Special flag to mark a level's exit. The environment typically stops
        once an agent reaches the exit.
    """

    alive_bit = 0  # Cell obeys Game of Life rules.
    agent_bit = 1
    pushable_bit = 2  # Can be pushed by agent.
    pullable_bit = 15  # (out of order for historical reasons)
    destructible_bit = 3
    frozen_bit = 4  # Does not evolve.
    preserving_bit = 5  # Neighboring cells do not die.
    inhibiting_bit = 6  # Neighboring cells cannot be born.
    spawning_bit = 7  # Randomly generates neighboring cells.
    exit_bit = 8
    color_bit = 9

    alive = np.uint16(1 << alive_bit)
    agent = np.uint16(1 << agent_bit)
    pushable = np.uint16(1 << pushable_bit)
    pullable = np.uint16(1 << pullable_bit)
    destructible = np.uint16(1 << destructible_bit)
    frozen = np.uint16(1 << frozen_bit)
    preserving = np.uint16(1 << preserving_bit)
    inhibiting = np.uint16(1 << inhibiting_bit)
    spawning = np.uint16(1 << spawning_bit)
    exit = np.uint16(1 << exit_bit)
    color_r = np.uint16(1 << color_bit)
    color_g = np.uint16(1 << color_bit + 1)
    color_b = np.uint16(1 << color_bit + 2)

    empty = np.uint16(0)
    freezing = inhibiting | preserving
    # Note that the player is marked as "destructible" so that they never
    # contribute to producing indestructible cells.
    player = agent | freezing | frozen | destructible
    wall = frozen
    movable = pushable | pullable
    crate = frozen | movable
    spawner = frozen | spawning | destructible
    hard_spawner = frozen | spawning
    level_exit = frozen | exit
    life = alive | destructible
    colors = (color_r, color_g, color_b)
    rainbow_color = color_r | color_g | color_b
    ice_cube = frozen | freezing | movable
    plant = frozen | alive | movable
    tree = frozen | alive
    fountain = preserving | frozen
    parasite = inhibiting | alive | pushable | frozen
    weed = preserving | alive | pushable | frozen
    powers = alive | freezing | spawning


class GameState(object):
    """
    Attributes
    ----------
    board : ndarray of ints with shape (h, w)
        Array of cells that make up the board. Note that the board is laid out
        according to the usual array convention where [0,0] corresponds to the
        top left.
    agent_loc : tuple
        x and y coordinates of the agent
    orientation : int
        Which way the agent is pointing. In range [0,3] with 0 indicating up.
    spawn_prob : float in [0,1]
        Probability for spawning new live cells next to spawners.
    file_name : str
        Path to .npz file where the state is to be saved.
    points_on_level_exit : float
    game_over : bool
        Flag to indicate that the current game has ended.
    num_steps : int
        Number of steps taken since last reset.
    can_toggle_powers : bool
        If true, players can absorb special powers of indestructible blocks
        by executing the "create" action on them. If they already have the
        power, they instead lose it. Note that the "freezing" power effectively
        cancels out the "alive" and "spawning" powers.
    can_toggle_colors : bool
        If true, players can also absorb the colors of indestructible blocks.
    min_performance : float
        Don't allow the agent to exit the level until the level is at least
        this fraction completed. If negative, the agent can always exit.
    """
    spawn_prob = 0.3
    orientation = 1
    agent_loc = (0, 0)
    edit_loc = (0, 0)
    edit_color = 0
    board = None
    file_name = None
    game_over = False
    points_on_level_exit = +1
    num_steps = 0
    min_performance = -1

    can_toggle_powers = False
    can_toggle_colors = False

    def __init__(self, board_size=(10,10)):
        self.exit_locs = (np.array([], dtype=int), np.array([], dtype=int))
        if board_size is None:
            # assume we'll load a new board from file
            pass
        else:
            self.make_default_board(board_size)
            self._init_data = self.serialize()
        self.initial_points = 0
        self.initial_available_points = 0

    def make_default_board(self, board_size):
        self.board = np.zeros(board_size, dtype=np.uint16)
        self.agent_loc = (board_size[1]//2, board_size[0]//2)
        self.board[self.agent_loc[1],self.agent_loc[0]] = CellTypes.player

    def serialize(self):
        """Return a dict of data to be serialized."""
        cls = self.__class__
        return {
            "spawn_prob": self.spawn_prob,
            "orientation": self.orientation,
            "agent_loc": self.agent_loc,
            "board": self.board.copy(),
            "class": "%s.%s" % (cls.__module__, cls.__name__),
            "min_performance": self.min_performance,
        }

    def deserialize(self, data, as_initial_state=True):
        """Load game state from a dictionary or npz archive."""
        keys = data.dtype.fields if hasattr(data, 'dtype') else data
        if as_initial_state:
            self._init_data = data
        self.board = data['board'].copy()
        if 'spawn_prob' in keys:
            self.spawn_prob = float(data['spawn_prob'])
        if 'orientation' in keys:
            self.orientation = int(data['orientation'])
        if 'agent_loc' in keys:
            self.agent_loc = tuple(data['agent_loc'])
        if 'min_performance' in keys:
            self.min_performance = float(data['min_performance'])
        self.update_exit_locs()
        self.game_over = False
        self.num_steps = 0

    def save(self, file_name=None):
        """Saves the game state to disk."""
        file_name = os.path.expanduser(file_name)
        file_name = os.path.abspath(file_name)
        if file_name is None:
            file_name = self.file_name
        if file_name is None:
            raise ValueError("Must specify a file name")
        if not file_name.endswith('.npz'):
            file_name += '.npz'
        self.file_name = file_name
        self._init_data = self.serialize()
        self.num_steps = 0
        np.savez_compressed(file_name, **self._init_data)

    def revert(self):
        """Revert to the last saved state."""
        if hasattr(self, '_init_data'):
            self.deserialize(self._init_data)
            return True
        return False

    @classmethod
    def loaddata(cls, data, auto_cls=True):
        """Load game state from a dictionary or npz archive (class agnostic)"""
        keys = data.dtype.fields if hasattr(data, 'dtype') else data
        if auto_cls and 'class' in keys:
            cls_components = str(data['class']).split('.')
            mod_name = '.'.join(cls_components[:-1])
            cls_name = cls_components[-1]
            try:
                mod = import_module(mod_name)
            except ImportError:
                mod = import_module(__name__)
            cls = getattr(mod, cls_name)
        obj = cls(board_size=None)
        obj.deserialize(data)
        return obj

    @classmethod
    def load(cls, file_name, auto_cls=True):
        """Load game state from disk."""
        file_name = os.path.expanduser(file_name)
        file_name = os.path.abspath(file_name)
        obj = cls.loaddata(np.load(file_name), auto_cls)
        obj.file_name = file_name
        return obj

    @property
    def width(self):
        """Width of the game board."""
        return self.board.shape[1]

    @property
    def height(self):
        """Height of the game board."""
        return self.board.shape[0]

    @property
    def title(self):
        """The bare file name without path or extension."""
        if self.file_name is None:
            return None
        else:
            fname = os.path.split(self.file_name)[-1]
            return '.'.join(fname.split('.')[:-1])

    @property
    def edit_color_name(self):
        return [
            'black',
            'red',
            'green',
            'yellow',
            'blue',
            'magenta',
            'cyan',
            'white',
        ][(self.edit_color & CellTypes.rainbow_color) >> CellTypes.color_bit]

    def relative_loc(self, n_forward, n_right=0):
        """
        Retrieves a location relative to the agent.

        Note the board wraps (topologically a torus).
        """
        dx = n_right
        dy = -n_forward
        for _ in range(self.orientation):
            # Rotate clockwise 90 degrees
            dx, dy = -dy, dx
        x0, y0 = self.agent_loc
        return (x0 + dx) % self.width, (y0 + dy) % self.height

    def move_agent(self, dy, dx=0):
        """
        Move the agent to a new location if that location is empty.

        Returns any associated reward.
        """
        x0, y0 = self.agent_loc
        x1, y1 = self.relative_loc(dy, dx)
        x2, y2 = self.relative_loc(-dy, -dx)
        can_push = (abs(dy), dx) == (1, 0)
        board = self.board
        reward = 0
        if board[y1, x1] == CellTypes.empty:
            board[y1, x1] = board[y0, x0]
            board[y0, x0] = CellTypes.empty
            self.agent_loc = (x1, y1)
        elif (board[y1, x1] & CellTypes.exit) and self.can_exit():
            # Don't actually move the agent, just mark as exited.
            self.game_over = True
            reward += self.points_on_level_exit
        elif can_push and board[y1, x1] & CellTypes.pushable:
            x3, y3 = self.relative_loc(dy*2)
            if board[y3, x3] == CellTypes.empty:
                # Push the cell forward one.
                board[y3, x3] = board[y1, x1]
                board[y1, x1] = board[y0, x0]
                board[y0, x0] = CellTypes.empty
                self.agent_loc = (x1, y1)
            elif board[y3, x3] & CellTypes.exit:
                # Push a block out of this level
                board[y1, x1] = board[y0, x0]
                board[y0, x0] = CellTypes.empty
                self.agent_loc = (x1, y1)
        agent_did_move = self.agent_loc == (x1, y1) and (x0, y0) != (x1, y1)
        if can_push and board[y2, x2] & CellTypes.pullable and agent_did_move:
            board[y0, x0] = board[y2, x2]
            board[y2, x2] = CellTypes.empty
        return reward

    def execute_action(self, action):
        """
        Execute an individual action and return the associated reward.

        Parameters
        ----------
        action : str
            Name of action to execute
        """
        board = self.board
        reward = 0
        if self.game_over:
            pass
        elif action.startswith("MOVE "):
            direction = ORIENTATION[action[5:]]
            if direction < 4:
                self.orientation = direction
                reward = self.move_agent(1)
            else:
                # Relative direction. Either forward (4) or backward (6)
                reward = self.move_agent(5 - direction)
        elif action.startswith("TURN "):
            direction = ORIENTATION[action[5:]]
            self.orientation += 2 - direction
            self.orientation %= 4
        elif action.startswith("FACE "):
            self.orientation = ORIENTATION[action[5:]]
        elif action.startswith("TOGGLE"):
            if len(action) > 6:
                # Toggle in a particular direction
                self.orientation = ORIENTATION[action[7:]]
            x0, y0 = self.agent_loc
            x1, y1 = self.relative_loc(1)
            player_color = board[y0, x0] & CellTypes.rainbow_color
            target_cell = board[y1, x1]
            if target_cell == CellTypes.empty:
                board[y1, x1] = CellTypes.life | player_color
            elif target_cell & CellTypes.destructible:
                board[y1, x1] = CellTypes.empty
            else:
                toggle_bits = CellTypes.powers * self.can_toggle_powers
                toggle_bits |= CellTypes.rainbow_color * self.can_toggle_colors
                board[y0, x0] ^= board[y1, x1] & toggle_bits
        elif action in ("RESTART", "ABORT LEVEL", "PREV LEVEL", "NEXT LEVEL"):
            self.game_over = action
        return reward

    def execute_edit(self, command):
        """
        Edit the board. Returns an error or success message, or None.

        Parameters
        ----------
        command : str
        """
        named_objects = {
            'EMPTY': CellTypes.empty,
            'LIFE': CellTypes.life,
            'HARD LIFE': CellTypes.alive,
            'WALL': CellTypes.wall,
            'CRATE': CellTypes.crate,
            'SPAWNER': CellTypes.spawner,
            'HARD SPAWNER': CellTypes.hard_spawner,
            'EXIT': CellTypes.level_exit,
            'ICECUBE': CellTypes.ice_cube,
            'PLANT': CellTypes.plant,
            'TREE': CellTypes.tree,
            'FOUNTAIN': CellTypes.fountain,
            'PARASITE': CellTypes.parasite,
            'WEED': CellTypes.weed,
        }
        toggles = {
            'ALIVE': CellTypes.alive,
            'INHIBITING': CellTypes.inhibiting,
            'PRESERVING': CellTypes.preserving,
            'SPAWNING': CellTypes.spawning,
        }
        board = self.board
        x0, y0 = self.agent_loc
        x1, y1 = self.edit_loc
        if command.startswith("MOVE "):
            direction = ORIENTATION[command[5:]]
            if direction % 2 == 0:
                dx, dy = 0, direction - 1
            else:
                dx, dy = 2 - direction, 0
            self.edit_loc = ((x1 + dx) % self.width, (y1 + dy) % self.height)
        elif command == "PUT AGENT":
            agent = board[y0, x0] & ~CellTypes.rainbow_color
            board[y0, x0] = 0
            board[y1, x1] = agent | self.edit_color
            self.agent_loc = self.edit_loc
        elif (command.startswith("PUT ") and command[4:] in named_objects and
                self.agent_loc != self.edit_loc):
            x1, y1 = self.edit_loc
            board[y1, x1] = named_objects[command[4:]]
            if board[y1, x1]:
                board[y1, x1] |= self.edit_color
        elif command.startswith("CHANGE COLOR"):
            if command.endswith("FULL CYCLE"):
                self.edit_color += CellTypes.color_r
            elif self.edit_color:
                self.edit_color <<= 1
            else:
                self.edit_color = CellTypes.color_r
            self.edit_color &= CellTypes.rainbow_color
            return "EDIT COLOR: " + self.edit_color_name
        elif command.startswith("TOGGLE ") and command[7:] in toggles:
            board[y0, x0] ^= toggles[command[7:]]
        elif command == "REVERT":
            if not self.revert():
                return "No saved state; cannot revert."
        elif command in ("ABORT LEVEL", "PREV LEVEL", "NEXT LEVEL"):
            self.game_over = command
        self.update_exit_locs()

    def shift_board(self, dx, dy):
        """Utility function. Translate the entire board (edges wrap)."""
        self.board = np.roll(self.board, dy, axis=0)
        self.board = np.roll(self.board, dx, axis=1)
        self.agent_loc = tuple(
            (np.array(self.agent_loc) + [dx, dy]) % [self.width, self.height])
        self.update_exit_locs()

    def resize_board(self, dx, dy):
        """Utility function. Expand or shrink the board."""
        height, width = self.board.shape
        if width <= 0 or height <= 0:
            raise ValueError("Cannot resize to zero.")
        new_board = np.zeros((height+dy, width+dx), dtype=self.board.dtype)
        height += min(0, dy)
        width += min(0, dx)
        new_board[:height, :width] = self.board[:height, :width]
        self.board = new_board
        self.agent_loc = tuple(
            np.array(self.agent_loc) % [self.width, self.height])
        self.update_exit_locs()

    def clip_board(self, left=0, right=0, top=0, bottom=0):
        """Utility function. Clip edges off of the board."""
        height, width = self.board.shape
        if left + right >= width or top + bottom >= height:
            raise ValueError("Board clipped to zero")
        self.shift_board(-left, -top)
        self.resize_board(-(left+right), -(bottom+top))

    def advance_board(self):
        """
        Apply one timestep of physics.
        """
        raise NotImplementedError

    @property
    def is_stochastic(self):
        raise NotImplementedError

    def check(self, condition):
        """
        Checks for a particular condition in front of the agent.

        Used for programmatic action sequences.
        """
        x, y = self.relative_loc(1)
        if condition == 'IFEMPTY':
            return self.board[y, x] == CellTypes.empty
        else:
            raise ValueError("Unknown condition '%s'" % (condition,))

    def current_points(self):
        """
        Current point value of the board.

        This depends on the current board state only.
        It does not depend on the initial board state or the board history.
        """
        return 0

    def available_points(self):
        """Number of points remaining to be earned."""
        return 0

    def points_earned(self):
        """Number of points that have been earned."""
        return self.current_points() - self.initial_points

    def required_points(self):
        """Total number of points needed to open the level exit."""
        req_points = self.min_performance * self.initial_available_points
        return max(0, int(np.ceil(req_points)))

    def can_exit(self):
        if self.min_performance < 0:
            return True
        return self.points_earned() >= self.required_points()

    def update_exit_locs(self):
        self.exit_locs = np.nonzero(self.board & CellTypes.exit)

    def update_exit_colors(self):
        if self.can_exit():
            exit_type = CellTypes.level_exit | CellTypes.color_r
        else:
            exit_type = CellTypes.level_exit
        i1, i2 = self.exit_locs
        self.board[i1, i2] = exit_type


class GameWithGoals(GameState):
    """
    Mixin for adding goal states to the game.

    Attributes
    ----------
    goals : ndarray
        Point value associated with each cell. Can be negative.
    point_table: ndarray
        Lookup table that maps goals (rows) and cell colors (columns) to
        point values for individual cells. Colors are KRGYBMCW.
    """
    goals = None
    _static_goals = None  # can be set to True for minor performance boost

    point_table = np.array([
        # k   r   g   y   b   m   c   w
        [+0, -1, +0, +0, +0, +0, +0, +0],  # black / no goal
        [-3, +3, -3, +0, -3, +0, -3, -3],  # red goal
        [+0, -3, +5, +0, +0, +0, +3, +0],  # green goal
        [-3, +0, +0, +3, +0, +0, +0, +0],  # yellow goal
        [+3, -3, +3, +0, +5, +3, +3, +3],  # blue goal
        [-3, +3, -3, +0, -3, +5, -3, -3],  # magenta goal
        [+3, -3, +3, +0, +3, +0, +5, +3],  # cyan goal
        [+0, -1, +0, +0, +0, +0, +0, +0],  # white / rainbow goal
    ])
    point_table.setflags(write=False)

    def make_default_board(self, board_size):
        super().make_default_board(board_size)
        self.goals = np.zeros_like(self.board)

    def serialize(self):
        data = super().serialize()
        data['goals'] = self.goals.copy()
        return data

    def deserialize(self, data, as_initial_state=True):
        super().deserialize(data, as_initial_state)
        self.goals = data['goals']
        if as_initial_state:
            self.initial_points = self.current_points()
            self.initial_available_points = self.available_points()
        self._static_goals = None

    def execute_edit(self, command):
        if command.startswith("GOALS "):
            # Swap goals and board so that the edit is done on the goals.
            self.board, self.goals = self.goals, self.board
            rval = super().execute_edit(command[6:])
            self.board, self.goals = self.goals, self.board
            self._static_goals = None
        else:
            rval = super().execute_edit(command)
        return rval

    def current_points(self, board=None, goals=None):
        if board is None:
            board = self.board
        if goals is None:
            goals = self.goals
        goals = (goals & CellTypes.rainbow_color) >> CellTypes.color_bit
        cell_colors = (board & CellTypes.rainbow_color) >> CellTypes.color_bit
        alive = board & CellTypes.alive > 0
        cell_points = self.point_table[goals, cell_colors] * alive
        return np.sum(cell_points)

    def available_points(self, board=None, goals=None):
        """
        Calculate the remaining points that are available on the board.

        This assumes that all goals can be filled in with any live color that
        exists on the board. It also assumes that the total number of goal
        cells of each type is constant. Both of these can easily be violated
        in practice.
        """
        if board is None:
            board = self.board
        if goals is None:
            goals = self.goals

        # Shift board and goals to only show their color. Values are [0, 8].
        goals = (goals & CellTypes.rainbow_color) >> CellTypes.color_bit
        cell_colors = (board & CellTypes.rainbow_color) >> CellTypes.color_bit

        # Mask out columns in the point table for which no colors are available
        alive_cells = board & CellTypes.alive > 0
        agent_cells = board & CellTypes.agent > 0
        available_colors = np.unique(cell_colors[alive_cells | agent_cells])
        mask = np.zeros(8, dtype=bool)
        mask[available_colors] = True
        pt_table = self.point_table * mask

        # Remove immovable cells from both goals and board
        immovable = board & (
            CellTypes.frozen | CellTypes.movable | CellTypes.destructible
        ) == CellTypes.frozen
        goals *= ~immovable
        cell_colors *= ~immovable

        # Calculate baseline rewards with current board, plus the total
        # available reward if all goals are filled.
        baseline_score = np.sum(pt_table[goals, cell_colors] * alive_cells)
        possible_score = np.sum(np.max(pt_table, axis=1)[goals])

        return possible_score - baseline_score

    def shift_board(self, dx, dy):
        """Utility function. Translate the entire board (edges wrap)."""
        super().shift_board(dx, dy)
        self.goals = np.roll(self.goals, dy, axis=0)
        self.goals = np.roll(self.goals, dx, axis=1)

    def resize_board(self, dx, dy):
        """Utility function. Expand or shrink the board."""
        super().resize_board(dx, dy)
        height, width = self.goals.shape
        new_goals = np.zeros((height+dy, width+dx), dtype=self.goals.dtype)
        height += min(0, dy)
        width += min(0, dx)
        new_goals[:height, :width] = self.goals[:height, :width]
        self.goals = new_goals


class SafeLifeGame(GameWithGoals):
    """
    Specifies all rules for the SafeLife game environment.

    Along with parent classes, this defines all of SafeLife's basic physics
    and the actions that the player can take.
    """

    def advance_board(self):
        self.num_steps += 1

        self.board = advance_board(self.board, self.spawn_prob)

        if not self._static_goals:
            new_goals = advance_board(self.goals, self.spawn_prob)
            if self._static_goals is None:
                # Check to see if they are, in fact, static
                self._static_goals = (
                    not (new_goals & CellTypes.spawning).any() and
                    (new_goals == self.goals).all()
                )
            self.goals = new_goals

    @property
    def is_stochastic(self):
        return (self.board & CellTypes.spawning).any()


class GameOfLife(GameWithGoals):
    """
    A more general version of SafeLifeGame which can use different
    cellular automata rules. Experimental!

    Conway's Game of Life uses cellular automata rules B3/S23.
    These can be changed though.

    Attributes
    ----------
    survive_rule : tuple of ints
        Number of neighbors that are required for a cell to survive to the
        next generation.
    born_rule : tuple of ints
        Number of neighbors that are required for a dead cell to come to life.
    """

    survive_rule = (2, 3)
    born_rule = (3,)

    def advance_board(self):
        """
        Apply one timestep of physics using Game of Life rules.
        """
        # We can advance the board using a pretty simple convolution,
        # so we don't have to execute a lot of loops in python.
        # Of course, this probably won't be sufficient for extremely
        # large boards.
        self.num_steps += 1
        board = self.board
        cfilter = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint16)

        alive = board & CellTypes.alive > 0
        spawning = board & CellTypes.spawning > 0
        frozen = board & CellTypes.frozen > 0

        can_die = ~frozen & (
            convolve2d(board & CellTypes.preserving, cfilter) == 0)
        can_grow = ~frozen & (
            convolve2d(board & CellTypes.inhibiting, cfilter) == 0)

        num_neighbors = convolve2d(alive, cfilter)
        num_spawn = convolve2d(spawning, cfilter)
        spawn_prob = 1 - (1 - self.spawn_prob)**num_spawn
        has_spawned = coinflip(spawn_prob, board.shape)

        born_rule = np.zeros(9, dtype=bool)
        born_rule[list(self.born_rule)] = True
        dead_rule = np.ones(9, dtype=bool)
        dead_rule[list(self.survive_rule)] = False

        new_alive = (born_rule[num_neighbors] | has_spawned) & ~alive & can_grow
        new_dead = dead_rule[num_neighbors] & alive & can_die

        new_flags = np.zeros_like(board)
        color_weights = 1 * alive + 2 * spawning
        for color in CellTypes.colors:
            # For each of the colors, see if there are two or more neighbors
            # that have it. If so, any new cells (whether born or spawned)
            # will also get that color.
            has_color = board & color > 0
            new_color = convolve2d(has_color * color_weights, cfilter) >= 2
            new_flags += color * new_color
        indestructible = alive & (board & CellTypes.destructible == 0)
        new_flags += CellTypes.destructible * (convolve2d(indestructible, cfilter) < 2)

        board *= ~(new_alive | new_dead)
        board += new_alive * (CellTypes.alive + new_flags)

    @property
    def is_stochastic(self):
        return (self.board & CellTypes.spawning).any()


class AsyncGame(GameWithGoals):
    """
    Game with asynchronous updates. Experimental!

    Asynchronous game physics work by updating the board one cell at a time,
    with many individual cell updates occurring for each board update.
    The order is random, so the system is naturally stochastic. In addition,
    a temperature parameter can be tuned to make the individual cell updates
    stochastic.

    Attributes
    ----------
    energy_rules : array of shape (2, num_neighbors + 1)
        The energy difference between a cell living and dying given its current
        state (live and dead, respectively) and the number of living neighbors
        that it has. The number of neighbors should either be 4, 6, or 8 for
        Von Neumann, hexagonal, and Moore neighborhoods respectively.
    temperature : float
        Higher temperatures lead to noisier systems, and are equivalent to
        lowering the values in the energy rules. Zero temperature yields a
        perfectly deterministic update per cell (although the cell update
        ordering is still random).
    cells_per_update : float
        Number of cell updates to perform at each board update, expressed as a
        fraction of the total board size. Can be more than 1.
    """
    energy_rule_sets = {
        'conway': (
            (-1, -1, +1, +1, -1, -1, -1, -1, -1),
            (-1, -1, -1, +1, -1, -1, -1, -1, -1),
        ),
        'ising': (
            (-2, -1, 0, +1, +2),
            (-2, -1, 0, +1, +2),
        ),
        'vine': (
            (-1, -1, +1, +1, +1),
            (-1, +1, -1, -1, -1),
        ),
    }
    energy_rules = energy_rule_sets['conway']
    temperature = 0
    cells_per_update = 0.3

    def serialize(self):
        data = super().serialize()
        data['energy_rules'] = self.energy_rules
        return data

    def deserialize(self, data, *args, **kw):
        super().deserialize(data, *args, **kw)
        self.energy_rules = data['energy_rules']

    def advance_board(self):
        """
        Apply one timestep of physics using an asynchronous update.

        Note that this is going to be quite slow. It should probably be
        replaced by an implementation in C, especially if used for training
        AI agents.
        """
        board = self.board
        rules = self.energy_rules
        h, w = board.shape
        beta = 1.0 / max(1e-20, self.temperature)
        if len(rules[0]) - 1 == 4:
            neighborhood = np.array([[0,1,0],[1,0,1],[0,1,0]])
        elif len(rules[0]) - 1 == 6:
            neighborhood = np.array([[0,1,1],[1,0,1],[1,1,0]])
        elif len(rules[0]) - 1 == 8:
            neighborhood = np.array([[1,1,1],[1,0,1],[1,1,1]])
        else:
            raise RuntimeError("async rules must have length 5, 7, or 9")
        rng = get_rng()
        for _ in range(int(board.size * self.cells_per_update)):
            x = rng.choice(w)
            y = rng.choice(h)
            if board[y, x] & CellTypes.frozen:
                continue
            neighbors = board.view(wrapping_array)[y-1:y+2, x-1:x+2] * neighborhood
            alive_neighbors = np.sum(neighbors & CellTypes.alive > 0)
            spawn_neighbors = np.sum(neighbors & CellTypes.spawning > 0)
            frozen = np.sum(neighbors & CellTypes.freezing) > 0
            if frozen:
                continue
            if board[y, x] & CellTypes.alive:
                H = rules[0][alive_neighbors]
            else:
                H = rules[1][alive_neighbors]

            P = 0.5 + 0.5*np.tanh(H * beta)
            P = 1 - (1-P)*(1-self.spawn_prob)**spawn_neighbors
            board[y, x] = CellTypes.life if coinflip(P) else CellTypes.empty
