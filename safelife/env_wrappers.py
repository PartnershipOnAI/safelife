import queue
import logging
import numpy as np

from gym import Wrapper
from .safelife_game import CellTypes
from .helper_utils import load_kwargs
from .speedups import advance_board

logger = logging.getLogger(__name__)


def call(x):
    return x() if callable(x) else x


class BaseWrapper(Wrapper):
    """
    Minor convenience class to make it easier to set attributes during init.
    """
    def __init__(self, env, **kwargs):
        super().__init__(env)
        load_kwargs(self, kwargs)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class MovementBonusWrapper(BaseWrapper):
    """
    Adds a bonus reward to incentivize agent movement.

    Without this, the agent will more easily get stuck. For example, the
    agent could find itself in a situation where any movement causes a pattern
    to collapse and the agent to lose points. Without the movement bonus,
    many agents will decide to forgo and prevent an immediate point loss.

    Attributes
    ----------
    movement_bonus : float
        Coefficients for the movement bonus. The agent's speed is calculated
        simply as the distance traveled divided by the time taken to travel it.
    movement_bonus_period : int
        The number of steps over which the movement bonus is calculated.
        By setting this to a larger number, one encourages the agent to
        maintain a particular bearing rather than circling back to where it
        was previously.
    movement_bonus_power : float
        Exponent applied to the movement bonus. Larger exponents will better
        reward maximal speed, while very small exponents will encourage any
        movement at all, even if not very fast.
    as_penalty : bool
        If True, the incentive is applied as a penalty for standing still
        rather than a bonus for movement. This shifts all rewards by
        a constant amount. This can be useful for episodic environments so
        that the agent does not receive a bonus for dallying and not reaching
        the level exit.
    """
    movement_bonus = 0.1
    movement_bonus_power = 1e-100
    movement_bonus_period = 4
    as_penalty = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Calculate the movement bonus
        p0 = self.game.agent_locs
        n = self.movement_bonus_period
        if len(self._prior_positions) >= n:
            p1 = self._prior_positions[-n]
            dist = np.sum(np.abs(p0-p1), axis=-1)
        elif len(self._prior_positions) > 0:
            p1 = self._prior_positions[0]
            dist = np.sum(np.abs(p0-p1), axis=-1)
            # If we're at the beginning of an episode, treat the
            # agent as if it were moving continuously before entering.
            dist += n - len(self._prior_positions)
        else:
            dist = n
        speed = dist / n
        if self.single_agent:  # convert to a scalar
            speed = np.sum(speed[:1])
        reward += self.movement_bonus * speed**self.movement_bonus_power
        if self.as_penalty:
            reward -= self.movement_bonus
        self._prior_positions.append(self.game.agent_locs.copy())

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._prior_positions = queue.deque(
            [self.game.agent_locs.copy()], self.movement_bonus_period)
        return obs


class ContinuingEnv(Wrapper):
    """
    Change to a continuing (rather than episodic) environment.

    The episode only ever ends if the 'times_up' flag gets set to True.

    Only suitable for single-agent environments.
    """
    def reset(self):
        assert self.single_agent, "ContinuingEnv requires single_agent = True"
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and not info['times_up']:
            done = False
            obs = self.env.reset()
        return obs, reward, done, info


class ExtraExitBonus(BaseWrapper):
    bonus = 0.5

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not info['times_up']:
            reward += done * call(self.bonus) * self.episode_reward
        return obs, reward, done, info


class MinPerformanceScheduler(BaseWrapper):
    """
    Provide a mechanism to set the `min_performance` for each episode.

    The `min_performance` specifies how much of the episode needs to be
    completed before the agent is allowed to leave through the level exit.
    The benchmark levels typically have `min_performance = 0.5`, but it can
    be helpful to start learning at a much lower value.
    """
    min_performance_fraction = 1

    def reset(self):
        obs = self.env.reset()
        self.game.min_performance *= call(self.min_performance_fraction)
        return obs


class SimpleSideEffectPenalty(BaseWrapper):
    """
    Penalize departures from starting or inaction state.

    Parameters
    ----------
    penalty_coef : float
        The magnitude of the side effect impact penalty.
    baseline : "starting-state" or "inaction"
        The starting-state baseline penalizes the agent for any departures
        from the starting state of the level. The inaction baseline penalizes
        the agent for any departures from the counterfactual state in which
        the agent had not acted.
    ignore_reward_cells : bool
        If True, the agent is only penalized for cell changes that do not
        otherwise result in a positive reward. This is (nearly) equivalent to
        applying the penalty to all changes, but then modifying the goal cells
        so that they are worth extra points.
    """
    penalty_coef = 0.0
    baseline = "starting-state"  # or "inaction"
    ignore_reward_cells = False

    def reset(self):
        obs = self.env.reset()
        self.last_side_effect = 0
        self.baseline_board = self.game.board.copy()
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.baseline == 'inaction':
            self.baseline_board = advance_board(self.baseline_board, self.game.spawn_prob)

        # Ignore the player's attributes so that moving around doesn't result
        # in a penalty. This also means that we ignore the destructible
        # attribute, so if a life cells switches to indestructible (which can
        # automatically happen for certain oscillators) that doesn't cause a
        # penalty either.
        # Note that this only works for uncolored (gray) players.
        board = self.game.board & ~CellTypes.player
        baseline_board = self.baseline_board & ~CellTypes.player

        # Also ignore exit locations (they change color when they open up)
        i1, i2 = self.game.exit_locs
        board[i1,i2] = baseline_board[i1,i2]

        # Finally, ignore any cells that are part of the reward.
        # This takes into account red cells and blue goals, but not other
        # potential rewards (other colors). Suitable for most training levels.
        unchanged = board == baseline_board
        if self.ignore_reward_cells:
            red_life = CellTypes.alive | CellTypes.color_r
            start_red = baseline_board & red_life == red_life
            end_red = board & red_life == red_life
            goal_cell = self.game.goals & CellTypes.rainbow_color == CellTypes.color_b
            end_alive = board & red_life == CellTypes.alive
            non_effects = unchanged | (start_red & ~end_red) | (goal_cell & end_alive)
            side_effect = np.sum(~non_effects)
        else:
            side_effect = np.sum(~unchanged)

        delta_effect = side_effect - self.last_side_effect
        reward -= delta_effect * call(self.penalty_coef)
        self.last_side_effect = side_effect
        return observation, reward, done, info
