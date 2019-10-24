import warnings

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from safelife import speedups
from .game_physics import CellTypes
from .helper_utils import recenter_view


class SafeLifeEnv(gym.Env):
    """
    A gym-like environment that wraps SafeLifeGame.

    This adds a few minor adjustments on top of the core rules:

    - A time limit is added. The episode ends when the time limit is reached
      with no further points awarded.
    - The observation is always centered on the agent, and the observation size
      doesn't have to match the size of the game board (it can be larger or
      smaller).
    - The “white” goal cells are (optionally) removed from the observation.
      These don't have any effect on the scoring or gameplay; they exist only
      as a visual reference.

    Parameters
    ----------
    level_iterator : iterator
        An iterator which produces :class:`game_physics.SafeLifeGame` instances.
        For example, :func:`file_finder.safelife_loader` will produce new games
        from saved game files or procedural generation parameters. This can be
        replaced with a custom iterator to do more complex level generation,
        such as implementing a level curriculum.
    remove_white_goals : bool
    output_channels : None or tuple of ints
        Specifies which channels get output in the observation.
        If None, the output is just a single channel copy of the board.
        If a tuple, each corresponding bit is given its own binary channel.
    view_shape : (int, int)
        Shape of the agent observation.
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
    game = None

    # The following are default parameters that can be overridden during
    # initialization.
    remove_white_goals = True
    view_shape = (15, 15)
    output_channels = tuple(range(15))  # default to all channels

    def __init__(self, level_iterator, **kwargs):
        self.level_iterator = level_iterator

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

    @property
    def state(self):
        warnings.warn(
            "'SafeLifeEnv.state' has been deprecated in favor of"
            "'SafeLifeEnv.game'",
            DeprecationWarning, stacklevel=2)
        return self.game

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        speedups.seed(seed)
        return [seed]

    def get_obs(self, board=None, goals=None, agent_loc=None):
        if board is None:
            board = self.game.board
        if goals is None:
            goals = self.game.goals
        if agent_loc is None:
            agent_loc = self.game.agent_loc

        board = board.copy()
        goals = goals & CellTypes.rainbow_color

        # Get rid of white cells in the goals.
        # They effectively act as just a background pattern, and they can
        # be confusing for an agent.
        if self.remove_white_goals:
            goals *= (goals != CellTypes.rainbow_color)

        # Combine board and goals into one array
        board += (goals << 3)

        # And center the array on the agent.
        board = recenter_view(
            board, self.view_shape, agent_loc[::-1], self.game.exit_locs)

        # If the environment specifies output channels, output a boolean array
        # with the channels as the third dimension. Otherwise output a bit
        # array.
        if self.output_channels:
            shift = np.array(list(self.output_channels), dtype=np.int16)
            board = (board[...,None] & (1 << shift)) >> shift
        return board

    def step(self, action):
        assert self.game is not None, "Game state is not initialized."
        action_name = self.action_names[action]
        reward = self.game.execute_action(action_name)
        self.game.advance_board()
        new_game_value = self.game.current_points()
        reward += new_game_value - self._old_game_value
        self._old_game_value = new_game_value
        self._num_steps += 1
        self.game.update_exit_colors()

        return self.get_obs(), reward, self.game.game_over, {
            'board': self.game.board,
            'goals': self.game.goals,
            'agent_loc': self.game.agent_loc,
        }

    def reset(self):
        self.game = next(self.level_iterator)
        self.game.revert()
        self.game.update_exit_colors()
        self._old_game_value = self.game.current_points()
        self._num_steps = 0
        return self.get_obs()

    def render(self, mode='ansi'):
        if mode == 'ansi':
            from .render_text import render_game
            return render_game(self.game, view_size=self.view_shape)
        else:
            from .render_graphics import render_game
            return render_game(self.game)

    def close(self):
        pass


def test_run(logdir=None):
    """
    A quick test to show that the environment is working properly
    """
    import shutil
    import os
    from training import wrappers
    from . import file_finder

    if logdir is None:
        logdir = os.path.abspath(os.path.join(__file__, '../../data/gym-test/'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    env = wrappers.VideoMonitor(SafeLifeEnv(file_finder.safelife_loader()), logdir)
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == "__main__":
    test_run()
