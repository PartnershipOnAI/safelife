import warnings

import gym
from gym import spaces
import numpy as np

from .level_iterator import SafeLifeLevelIterator
from .safelife_game import CellTypes
from .helper_utils import recenter_view, load_kwargs
from .random import set_rng


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
        An iterator which produces :class:`safelife_game.SafeLifeGame` instances.
        For example, :func:`level_iterator.SafeLifeLevelIterator` will produce
        new games from saved game files or procedural generation parameters.
        This can be replaced with a custom iterator to do more complex level
        generation, such as implementing a level curriculum.

        Note that if the level iterator has a `seed` method it will be called
        with a :class:`numpy.random.SeedSequence` object.
    single_agent : bool
        If True, the `step` function will return a single reward and
        observation. If False, it will return separate rewards and observations
        for each agent in the level.
    time_limit : int
        Maximum steps allowed per episode.
    remove_white_goals : bool
    output_channels : None or tuple of ints
        Specifies which channels get output in the observation.
        If None, the output is just a single channel copy of the board.
        If a tuple, each corresponding bit is given its own binary channel.
    view_shape : (int, int)
        Shape of the agent observation.
    """

    game = None

    # The following are default parameters that can be overridden during
    # initialization.
    single_agent = False
    time_limit = 1000
    remove_white_goals = True
    view_shape = (15, 15)
    output_channels = tuple(range(15))  # default to all channels

    def __init__(self, level_iterator, **kwargs):
        self.level_iterator = level_iterator

        load_kwargs(self, kwargs)

        self.action_space = spaces.Discrete(9)
        if self.output_channels is None:
            self.observation_space = spaces.Box(
                low=0, high=2**15,
                shape=self.view_shape,
                dtype=np.uint32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=self.view_shape + (len(self.output_channels),),
                dtype=np.uint8,
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
        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(seed)
        if hasattr(self.level_iterator, 'seed'):
            self.level_iterator.seed(seed.spawn(1)[0])
        return [seed.entropy]

    def get_obs(self, board=None, goals=None, agent_locs=None):
        if board is None:
            board = self.game.board
        if goals is None:
            goals = self.game.goals
        if agent_locs is None:
            agent_locs = self.game.agent_locs
        if self.single_agent:
            if len(agent_locs) > 0:
                agent_locs = agent_locs[:1]
            else:
                # Make default view centered at origin if there are no agents.
                agent_locs = np.array([[0,0]])

        board = board.astype(np.uint32)
        goals = goals & CellTypes.rainbow_color

        # Get rid of white cells in the goals.
        # They effectively act as just a background pattern, and they can
        # be confusing for an agent.
        if self.remove_white_goals:
            goals *= (goals != CellTypes.rainbow_color)

        # Combine board and goals into one array
        board += (goals.astype(np.uint32) << 16)

        # And center the array on the agent.
        board = np.stack([
            recenter_view(board, self.view_shape, x0, self.game.exit_locs)
            for x0 in agent_locs
        ])

        # If the environment specifies output channels, output a boolean array
        # with the channels as the third dimension. Otherwise output a bit
        # array.
        if self.output_channels:
            shift = np.array(list(self.output_channels), dtype=np.uint32)
            board = (board[...,None] & (1 << shift)) >> shift
            board = board.astype(np.uint8)
        if self.single_agent:
            board = board[0]
        return board

    def step(self, actions):
        assert self.game is not None, "Game state is not initialized."

        self.game.execute_actions(actions)
        with set_rng(self.rng):
            self.game.advance_board()
        self.game.update_exit_colors()

        self.episode_length += 1
        times_up = self.episode_length > self.time_limit

        new_game_value = self.game.current_points()
        reward = new_game_value - self._old_game_value
        self._old_game_value = new_game_value
        done = ~self.game.agent_is_active() | times_up

        if self.single_agent:
            if len(reward) == 0:
                reward = 0
                done = True
            else:
                reward = reward[0]
                done = done[0]

        reward = np.float32(reward)
        self.episode_reward += reward

        return self.get_obs(), reward, done, {
            'board': self.game.board,
            'goals': self.game.goals,
            'agent_locs': self.game.agent_locs,
            'title': self.game.title,
            'times_up': times_up,
            'episode': {
                'length': self.episode_length,
                'reward': self.episode_reward,
            }
        }

    def reset(self):
        self.game = next(self.level_iterator)
        self.game.revert()
        self.game.update_exit_colors()
        self._old_game_value = self.game.current_points()
        self.episode_length = 0
        self.episode_reward = 0
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

    @classmethod
    def register(cls):
        """Registers a few canonical environments with OpenAI Gym."""
        for name in [
            "append-still", "prune-still",
            "append-still-easy", "prune-still-easy",
            "append-spawn", "prune-spawn",
            "navigation", "challenge"
        ]:
            gym.register(
                id="safelife-{}-v1".format(name),
                entry_point=SafeLifeEnv,
                kwargs={
                    'level_iterator': SafeLifeLevelIterator('random/' + name),
                },
            )
