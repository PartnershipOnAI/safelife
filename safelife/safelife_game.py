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
from functools import wraps

import numpy as np

from .helper_utils import (
    wrapping_array,
    wrapped_convolution as convolve2d,
)
from .random import coinflip, get_rng, set_rng
from .speedups import advance_board, alive_counts, execute_actions


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
    orientation_bit = 12

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
    orientation_mask = np.uint16(3 << orientation_bit)

    empty = np.uint16(0)
    freezing = inhibiting | preserving
    movable = pushable | pullable
    # Note that the player is marked as "destructible" so that they never
    # contribute to producing indestructible cells.
    player = agent | freezing | frozen | destructible
    wall = frozen
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
    agent_locs : ndarray
        Coordinates of each agent on the board (row, col)
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
    seed : np.random.SeedSequence
        Seed used for stochastic dynamics
    """
    spawn_prob = 0.3
    edit_loc = (0, 0)
    edit_color = 0
    board = None
    file_name = None
    game_over = False
    points_on_level_exit = +1
    num_steps = 0
    _seed = None
    _rng = None

    def __init__(self, board_size=(10,10)):
        self.exit_locs = (np.array([], dtype=int), np.array([], dtype=int))
        self.agent_locs = np.empty((0,2), dtype=int)
        if board_size is None:
            # assume we'll load a new board from file
            pass
        else:
            self.make_default_board(board_size)
            self._init_data = self.serialize()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def rng(self):
        return self._rng if self._rng is not None else get_rng()

    @staticmethod
    def use_rng(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            with set_rng(self.rng):
                return f(self, *args, **kwargs)
        return wrapper

    def make_default_board(self, board_size):
        self.board = np.zeros(board_size, dtype=np.uint16)
        self.agent_locs = np.array(board_size).reshape(1,2) // 2
        self.agent_names = np.array(['agent0'])
        self.board[self.agent_locs_idx] = CellTypes.player

    def serialize(self):
        """Return a dict of data to be serialized."""
        cls = self.__class__
        return {
            "spawn_prob": self.spawn_prob,
            "agent_locs": self.agent_locs.copy(),
            "agent_names": self.agent_names.copy(),
            "board": self.board.copy(),
            "class": "%s.%s" % (cls.__module__, cls.__name__),
        }

    def deserialize(self, data, as_initial_state=True):
        """Load game state from a dictionary or npz archive."""
        keys = data.dtype.fields if hasattr(data, 'dtype') else data
        if as_initial_state:
            self._init_data = data
        self.board = data['board'].copy()
        if 'spawn_prob' in keys:
            self.spawn_prob = float(data['spawn_prob'])
        if 'agent_loc' in keys:
            # Old single agent setting
            self.agent_locs = np.array(data['agent_loc'])[None,::-1]
        elif 'agent_locs' in keys:
            self.agent_locs = np.array(data['agent_locs'])
        if 'agent_names' in keys:
            self.agent_names = np.array(data['agent_names'])
        else:
            self.agent_names = np.array([
                'agent%i' % i for i in range(len(self.agent_locs))
            ])
        if 'orientation' in keys:
            self.orientation = int(data['orientation'])
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
            fname, *ext = fname.rsplit('.', 1)
            procgen = ext and ext[0] in ('json', 'yaml')
            if procgen and self._seed and self._seed.spawn_key:
                # Append the spawn key as the episode number
                fname += '-e' + str(self._seed.spawn_key[-1])
            return fname

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

    @property
    def orientation(self):
        """Orientation of the agents. For backwards compatibility."""
        agents = self.board[self.agent_locs_idx]
        out = (agents & CellTypes.orientation_mask) >> CellTypes.orientation_bit
        return out.astype(np.int64)

    @orientation.setter
    def orientation(self, value):
        value = (np.array(value, dtype=np.uint16) & 3) << CellTypes.orientation_bit
        self.board[self.agent_locs_idx] &= ~CellTypes.orientation_mask
        self.board[self.agent_locs_idx] |= value

    @property
    def agent_locs_idx(self):
        """
        Convenience for easier array indexing.

        E.g., ``agents = self.board[self.agent_locs_idx]``
        """
        return tuple(self.agent_locs.T)

    def execute_action(self, action):
        """
        Perform a named agent action.

        This is primarily for interactive use. Learning algorithms
        and environments should call `execute_actions()` instead.
        """
        if self.game_over or len(self.agent_locs) == 0:
            pass
        elif action.startswith("MOVE "):
            direction = ORIENTATION[action[5:]]
            flip = 2 if direction == 6 else 0
            if direction < 4:
                self.execute_actions(direction + 1)
            else:
                # Relative direction. Either forward (4) or backward (6)
                direction = self.orientation ^ flip
                self.execute_actions(direction + 1)
            self.orientation ^= flip
            self.game_over = self.has_exited().any()
        elif action.startswith("TURN "):
            direction = ORIENTATION[action[5:]]
            self.orientation += 2 - direction
            self.orientation %= 4
        elif action.startswith("FACE "):
            self.orientation = ORIENTATION[action[5:]]
        elif action.startswith("TOGGLE"):
            if len(action) > 6:
                # Toggle in a particular direction
                direction = ORIENTATION[action[7:]]
            else:
                direction = self.orientation
            self.execute_actions(direction + 5)
        elif action in ("RESTART", "ABORT LEVEL", "PREV LEVEL", "NEXT LEVEL"):
            self.game_over = action
        return 0

    def execute_actions(self, actions):
        """
        Perform (potentially different) actions for each agent.

        Parameters
        ----------
        actions : int or ndarray
            Actions for each agent. Should be in range [0-8].
        """
        execute_actions(self.board, self.agent_locs, actions)

    def execute_edit(self, command, board=None):
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
            'AGENT': CellTypes.player,
        }
        toggles = {
            "AGENT": CellTypes.agent,
            "ALIVE": CellTypes.alive,
            "PUSHABLE": CellTypes.pushable,
            "PULLABLE": CellTypes.pullable,
            "DESTRUCTIBLE": CellTypes.destructible,
            "FROZEN": CellTypes.frozen,
            "PRESERVING": CellTypes.preserving,
            "INHIBITING": CellTypes.inhibiting,
            "SPAWNING": CellTypes.spawning,
            "EXIT": CellTypes.exit,
        }
        if board is None:
            board = self.board
        edit_loc = self.edit_loc
        if command.startswith("MOVE "):
            direction = ORIENTATION[command[5:]]
            if direction % 2 == 0:
                dx = np.array([direction - 1, 0])
            else:
                dx = np.array([0, 2 - direction])
            self.edit_loc = tuple((edit_loc + dx) % board.shape)
        elif command.startswith("PUT ") and command[4:] in named_objects:
            board[edit_loc] = named_objects[command[4:]]
            if board[edit_loc]:
                board[edit_loc] |= self.edit_color
        elif command == "NEXT EDIT COLOR":
            self.edit_color += CellTypes.color_r
            self.edit_color &= CellTypes.rainbow_color
            return "EDIT COLOR: " + self.edit_color_name
        elif command == "PREVIOUS EDIT COLOR":
            self.edit_color -= CellTypes.color_r
            self.edit_color &= CellTypes.rainbow_color
            return "EDIT COLOR: " + self.edit_color_name
        elif command == "APPLY EDIT COLOR":
            board[edit_loc] &= ~CellTypes.rainbow_color
            board[edit_loc] |= self.edit_color
        elif command.startswith("TOGGLE ") and command[7:] in toggles:
            board[edit_loc] ^= toggles[command[7:]]
        elif command == "REVERT":
            if not self.revert():
                return "No saved state; cannot revert."
        elif command in ("ABORT LEVEL", "PREV LEVEL", "NEXT LEVEL"):
            self.game_over = command
        self.update_exit_locs()
        self.update_exit_colors()
        self.update_agent_locs()

    def shift_board(self, dx, dy):
        """Utility function. Translate the entire board (edges wrap)."""
        self.board = np.roll(self.board, dy, axis=0)
        self.board = np.roll(self.board, dx, axis=1)
        self.agent_locs += [dy, dx]
        self.agent_locs %= self.board.shape
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
        out_of_bounds = np.any(self.agent_locs >= new_board.shape, axis=1)
        self.agent_locs = self.agent_locs[~out_of_bounds]
        self.edit_loc = tuple(np.array(self.edit_loc) % new_board.shape)
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

    def has_exited(self):
        """
        Boolean value for each agent.
        """
        agents = self.board[self.agent_locs_idx]
        return agents & (CellTypes.agent | CellTypes.exit) == CellTypes.exit

    def agent_is_active(self):
        """
        Boolean value for each agent
        """
        agents = self.board[self.agent_locs_idx]
        return agents & CellTypes.agent > 0

    def current_points(self):
        """
        Current point value of the board.

        This depends on the current board state only.
        It does not depend on the initial board state or the board history.
        """
        return self.points_on_level_exit * self.has_exited()

    def can_exit(self):
        # As long as the agent is on the board, it can exit
        # (this is overridden below)
        return self.board[self.agent_locs_idx] & CellTypes.agent > 0

    def update_exit_locs(self):
        exits = self.board & (CellTypes.exit | CellTypes.agent) == CellTypes.exit
        self.exit_locs = np.nonzero(exits)

    def update_exit_colors(self):
        can_exit = self.can_exit()

        # Set the exit bit on top of each agent that is allowed to exit
        self.board[self.agent_locs_idx] &= ~CellTypes.exit
        self.board[self.agent_locs_idx] |= CellTypes.exit * can_exit

        # If any agent can exit, set the exit color to red.
        # This is mostly a visual aid for humans, but could conceivably be
        # used by non-human agents too (e.g., recognize when another agent is
        # able to open the exit).
        if can_exit.any():
            exit_type = CellTypes.level_exit | CellTypes.color_r
        else:
            exit_type = CellTypes.level_exit
        self.board[self.exit_locs] = exit_type

    def update_agent_locs(self):
        new_locs = np.stack(
            np.nonzero(self.board & CellTypes.agent),
            axis=1)

        # Do some gymnastics to respect the old order
        old_locs = self.agent_locs
        compare = np.all(new_locs[None] == old_locs[:,None], axis=-1)
        self.agent_locs = np.append(
            old_locs[np.any(compare, axis=1)],
            new_locs[~np.any(compare, axis=0)],
            axis=0)

        # but for now, don't do the same gymnastics with agent names
        # (this should be fixed later)
        if len(old_locs) != len(new_locs):
            self.agent_names = np.array([
                'agent%i' % i for i in range(len(self.agent_locs))
            ])


class GameWithGoals(GameState):
    """
    Mixin for adding goal states to the game.

    Attributes
    ----------
    goals : ndarray
        Point value associated with each cell. Can be negative.
    points_table: ndarray
        Lookup table that maps goals (rows) and cell colors (columns) to
        point values for individual cells. Colors are KRGYBMCW.
    min_performance : float
        Don't allow the agent to exit the level until the level is at least
        this fraction completed. If negative, the agent can always exit.
    """
    goals = None
    _static_goals = None  # can be set to True for minor performance boost
    min_performance = -1

    # TODO: make a different point table for each color agent
    default_points_table = np.array([
        # k   r   g   y   b   m   c   w  empty
        [+0, -1, +0, +0, +0, +0, +0, +0, 0],  # black / no goal
        [-3, +3, -3, +0, -3, +0, -3, -3, 0],  # red goal
        [+0, -3, +5, +0, +0, +0, +3, +0, 0],  # green goal
        [-3, +0, +0, +3, +0, +0, +0, +0, 0],  # yellow goal
        [+3, -3, +3, +0, +5, +3, +3, +3, 0],  # blue goal
        [-3, +3, -3, +0, -3, +5, -3, -3, 0],  # magenta goal
        [+3, -3, +3, +0, +3, +0, +5, +3, 0],  # cyan goal
        [+0, -1, +0, +0, +0, +0, +0, +0, 0],  # white / rainbow goal
    ])
    default_points_table.setflags(write=False)

    def make_default_board(self, board_size):
        super().make_default_board(board_size)
        self.goals = np.zeros_like(self.board)
        self._needs_new_counts = True
        self.setup_initial_counts()
        self.reset_points_table()

    def serialize(self):
        data = super().serialize()
        data['goals'] = self.goals.copy()
        data['points_table'] = self.points_table.copy()
        data['min_performance'] = self.min_performance
        return data

    def deserialize(self, data, as_initial_state=True):
        super().deserialize(data, as_initial_state)

        keys = data.dtype.fields if hasattr(data, 'dtype') else data
        self.goals = data['goals']
        if 'min_performance' in keys:
            self.min_performance = data['min_performance']
        if 'points_table' in keys:
            self.points_table = data['points_table']
        else:
            self.reset_points_table()
        self._needs_new_counts = True
        if as_initial_state:
            self.setup_initial_counts()
        self._static_goals = None
        self.update_exit_colors()

    def execute_edit(self, command):
        if command.startswith("GOALS "):
            rval = super().execute_edit(command[6:], self.goals)
            self._static_goals = None
        else:
            rval = super().execute_edit(command)
        self._needs_new_counts = True
        if len(self.points_table) != len(self.agent_locs):
            # Ideally we would handle this in a smarter way, but for now
            # if we add or subtract an agent, we just reset the scoring to
            # the default values.
            self.reset_points_table()
        return rval

    def execute_action(self, action):
        self._needs_new_counts = True
        return super().execute_action(action)

    @property
    def alive_counts(self):
        if getattr(self, '_needs_new_counts', True):
            self._needs_new_counts = False
            self._alive_counts = alive_counts(self.board, self.goals)
            self._alive_counts.setflags(write=False)
        return self._alive_counts

    def setup_initial_counts(self):
        """
        Record the counts of live cells and possible colors for new cells.
        """
        self.initial_counts = self.alive_counts
        self.initial_colors = np.zeros(9, dtype=bool)
        generators = CellTypes.agent | CellTypes.alive | CellTypes.spawning
        colors = self.board[self.board & generators > 0] & CellTypes.rainbow_color
        colors = np.unique(colors) >> CellTypes.color_bit
        self.initial_colors[colors] = True
        self.initial_colors[-1] = True  # 'empty' color

    def reset_points_table(self):
        """
        Reset the points table to default values.
        """
        num_agents = len(self.agent_locs)
        self.points_table = np.tile(self.default_points_table, [num_agents, 1, 1])

    def current_points(self):
        points = self.points_table * self.alive_counts
        points = points.reshape(-1,72)  # unravel the points for easier sum
        return np.sum(points, axis=1) + super().current_points()

    def points_earned(self):
        """Number of points that have been earned."""
        delta_counts = self.alive_counts - self.initial_counts
        points = self.points_table * delta_counts
        points = points.reshape(-1,72)  # unravel the points for easier sum
        return np.sum(points, axis=1) + super().current_points()

    def initial_available_points(self):
        """
        Total points available to the agents, assuming all goals can be filled.
        """
        goal_counts = np.sum(self.initial_counts, axis=1)  # int array, length 8
        # Zero out columns in the point table corresponding to unavailable colors
        points_table = self.points_table * self.initial_colors  # shape (n,8,9)
        max_points = np.max(points_table, axis=2)  # shape (n,8)
        total_available = np.sum(max_points * goal_counts, axis=1)  # shape (n,)

        initial_points = self.points_table * self.initial_counts
        initial_points = np.sum(initial_points.reshape(-1,72), axis=1)

        return total_available - initial_points

    def required_points(self):
        """Total number of points needed to open the level exit."""
        req_points = self.min_performance * self.initial_available_points()
        return np.maximum(0, np.int64(np.ceil(req_points)))

    def can_exit(self):
        points_earned = np.maximum(0, self.points_earned())
        is_agent = self.board[self.agent_locs_idx] & CellTypes.agent > 0
        return is_agent & (points_earned >= self.required_points())

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

    @GameState.use_rng
    def advance_board(self):
        self.num_steps += 1
        self._needs_new_counts = True

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

    @GameState.use_rng
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

    @GameState.use_rng
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
