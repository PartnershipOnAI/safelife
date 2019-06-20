import queue
from multiprocessing import Pool

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .game_physics import SafeLife, CellTypes
from .array_utils import wrapping_array
from .gen_board import gen_game


class SafeLifeEnv(gym.Env):
    """
    A gym-like environment wrapper for SafeLife.

    This adds a few minor adjustments on top of the core rules:

    - A time limit is added. The episode ends when the time limit is reached
      with no further points awarded.
    - An (optional) penalty is applied when the agent doesn't move between
      time steps.
    - The observation is always centered on the agent, and the observation size
      doesn't have to match the size of the game board (it can be larger or
      smaller).
    - The “white” goal cells are (optionally) removed from the observation.
      These don't have any effect on the scoring or gameplay; they exist only
      as a visual reference.

    Parameters
    ----------
    max_steps : int
    no_movement_penalty : float
    remove_white_goals : bool
    output_channels : None or tuple of ints
        Specifies which channels get output in the observation.
        If None, the output is just a single channel copy of the board.
        If a tuple, each corresponding bit is given its own binary channel.
    view_shape : (int, int)
        Shape of the agent observation.
    board_gen_params : dict
        Parameters to be passed to :func:`gen_board.gen_game()`.
    fixed_levels : list of level names
        If set, levels are loaded from disk rather than procedurally generated.
    randomize_fixed_levels : bool
        If true, fixed levels will be played in a random order (shuffled once
        per epoch).
    """

    metadata = {
        'render.modes': ['ansi', 'rgb_array'],
        'video.frames_per_second': 30
    }
    action_names = (
        "NULL",
        "MOVE UP",
        "MOVE RIGHT",
        "MOVE DOWN",
        "MOVE LEFT",
        "TOGGLE UP",
        "TOGGLE RIGHT",
        "TOGGLE DOWN",
        "TOGGLE LEFT",
    )
    state = None

    # The following are default parameters that can be overridden during
    # initialization, or really at any other time.
    # `view_shape` and `output_channels` should probably be kept constant
    # after initialization.
    max_steps = 1200
    no_movement_penalty = 0.02
    remove_white_goals = True
    view_shape = (15, 15)
    output_channels = tuple(range(15))  # default to all channels

    _fixed_levels = []
    randomize_fixed_levels = True

    _pool = Pool(processes=8)
    _min_queue_size = 1

    def __init__(self, **kwargs):
        self.board_gen_params = {}
        for key, val in kwargs.items():
            if (not key.startswith('_') and hasattr(self, key) and
                    not callable(getattr(self, key))):
                setattr(self, key, val)
            else:
                raise ValueError("Unrecognized parameter: '%s'" % (key,))

        self.action_space = spaces.Discrete(len(self.action_names))
        if self.output_channels is None:
            self.observation_space = spaces.Box(
                low=0, high=2**15,
                shape=self.view_shape,
                dtype=np.uint16,
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=self.view_shape + (len(self.output_channels),),
                dtype=np.uint16,
            )
        self.seed()
        self._board_queue = queue.deque()

    @property
    def fixed_levels(self):
        return tuple(level.file_name for level in self._fixed_levels)

    @fixed_levels.setter
    def fixed_levels(self, val):
        self._fixed_levels = [SafeLife.load(fname) for fname in val]
        self._level_idx = len(self._fixed_levels)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        board = self.state.board.copy()
        goals = self.state.goals.copy()

        # Get rid of the frozen flag for the agent and exit.
        # (maybe a minor optimization)
        #   agent_or_exit = (board & (CellTypes.agent | CellTypes.exit)) > 0
        #   board ^= CellTypes.frozen * agent_or_exit

        # Get rid of white cells in the goals.
        # They effectively act as just a background pattern, and they can
        # be confusing for an agent.
        if self.remove_white_goals:
            goals *= (goals != CellTypes.rainbow_color)

        # Combine board and goals into one array
        board += (goals << 3)

        # And center the array on the agent.
        h, w = self.view_shape
        x0, y0 = self.state.agent_loc
        x0 -= w // 2
        y0 -= h // 2
        board = board.view(wrapping_array)[y0:y0+h, x0:x0+w]
        board = board.view(np.ndarray)

        # If the environment specifies output channels, output a boolean array
        # with the channels as the third dimension. Otherwise output a bit
        # array.
        if self.output_channels:
            shift = np.array(list(self.output_channels), dtype=np.int16)
            board = (board[...,None] & (1 << shift)) >> shift
        return board

    def step(self, action):
        assert self.state is not None, "State not initializeddef."
        old_position = self.state.agent_loc
        self.state.advance_board()
        action_name = self.action_names[action]
        base_reward = self.state.execute_action(action_name)
        new_state_value = self.state.current_points()
        base_reward += new_state_value - self._old_state_value
        self._old_state_value = new_state_value
        self._num_steps += 1
        times_up = self._num_steps >= self.max_steps
        done = self.state.game_over or times_up
        standing_still = old_position == self.state.agent_loc
        reward = base_reward / 3.0 - standing_still * self.no_movement_penalty
        return self._get_obs(), reward, done, {
            'did_move': not standing_still,
            'times_up': times_up,
            'base_reward': base_reward,
        }

    def reset(self):
        if self._fixed_levels:
            if self._level_idx >= len(self._fixed_levels):
                self._level_idx = 0
                if self.randomize_fixed_levels:
                    np.random.shuffle(self._fixed_levels)
            self.state = self._fixed_levels[self._level_idx]
            self._level_idx += 1
            self.state.revert()
        else:
            while len(self._board_queue) <= self._min_queue_size:
                self._board_queue.append(self._pool.apply_async(
                    gen_game, (), self.board_gen_params))
            self.state = self._board_queue.popleft().get()
        self._old_state_value = self.state.current_points()
        self._num_steps = 0
        return self._get_obs()

    def render(self, mode='ansi'):
        if mode == 'ansi':
            from .asci_renderer import render_board
            return render_board(self.state, view_size=self.view_shape)
        else:
            from .rgb_renderer import render_game
            return render_game(self.state)

    def close(self):
        pass


def test_run(logdir=None):
    """
    A quick test to show that the environment is working properly
    """
    import shutil
    import os
    from . import wrappers

    if logdir is None:
        logdir = os.path.abspath(os.path.join(__file__, '../../data/gym-test/'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    env = wrappers.VideoMonitor(SafeLifeEnv(), logdir)
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == "__main__":
    test_run()
