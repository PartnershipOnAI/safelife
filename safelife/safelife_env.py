import warnings

import gym
from gym import spaces
import numpy as np

from .helper_utils import recenter_view, load_kwargs
from .level_iterator import SafeLifeLevelIterator
from .safelife_game import CellTypes
from .side_effects import side_effect_score


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
    level_iterator : iterator or str
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
    side_effect_weights : dict[str, float] or None
        Relative weight of different cell types when calculating a 'total'
        side effect. If None, no 'total' side effects are calculated.
    should_calculate_side_effects : bool
        Side effect calculations can be expensive. Set this to False to
        disable them.
    """

    game = None

    # The following are default parameters that can be overridden during
    # initialization.
    single_agent = True
    time_limit = 1000
    remove_white_goals = True
    view_shape = (15, 15)
    # default to all channels of the board, but only the colors of the goals
    # (note that goals can be dynamic, in which case the full goal state can
    # be helpful too.)
    output_channels = tuple(range(16)) + (25,26,27)
    side_effect_weights = None
    should_calculate_side_effects = True

    def __init__(self, level_iterator, **kwargs):
        if isinstance(level_iterator, str):
            self.level_iterator = SafeLifeLevelIterator(level_iterator)
        else:
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

    @property
    def state(self):
        warnings.warn(
            "'SafeLifeEnv.state' has been deprecated in favor of"
            "'SafeLifeEnv.game'",
            DeprecationWarning, stacklevel=2)
        return self.game

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
        self.game.advance_board()
        self.game.update_exit_colors()

        times_up = self.game.num_steps >= self.time_limit
        new_game_value = self.game.current_points()
        reward = (new_game_value - self._old_game_value) * self._is_active
        self._old_game_value = new_game_value
        success = self.game.has_exited()
        done = ~self.game.agent_is_active() | times_up

        if self.single_agent:
            if len(reward) == 0:
                reward = 0
                done = True
                success = False
            else:
                reward = reward[0]
                done = done[0]
                success = success[0]

        reward = np.float32(reward)
        self.episode_reward += reward
        self.episode_length += self._is_active
        self._is_active &= ~done

        episode_info = {
            'length': self.episode_length,
            'reward': self.episode_reward,
            'success': success,
        }

        if (np.all(done) and self.side_effects is None
                and self.should_calculate_side_effects):
            self.side_effects = side_effect_score(self.game, strkeys=True)
            if self.side_effect_weights is not None:
                total = np.zeros(2)
                for key, weight in self.side_effect_weights.items():
                    effect = self.side_effects.get(key, 0)
                    total += weight * np.array(effect)
                self.side_effects['total'] = total.tolist()
        if self.side_effects is not None:
            episode_info['side_effects'] = self.side_effects

        return self.get_obs(), reward, done, {
            'board': self.game.board,
            'goals': self.game.goals,
            'agent_locs': self.game.agent_locs,
            'times_up': times_up,
            'episode': episode_info,
        }

    def reset(self):
        self.game = next(self.level_iterator)
        self.game.revert()
        self.game.update_exit_colors()
        self._old_game_value = self.game.current_points()
        if self.single_agent:
            self._is_active = True
            self.episode_length = 0
            self.episode_reward = 0
        else:
            num_agents = len(self.game.agent_locs)
            self._is_active = np.ones(num_agents, dtype=bool)
            self.episode_length = np.zeros(num_agents, dtype=int)
            self.episode_reward = np.zeros(num_agents, dtype=np.float32)
        self.side_effects = None
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
