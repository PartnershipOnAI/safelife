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
    movement_bonus_power = 0.01
    movement_bonus_period = 4
    as_penalty = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Calculate the movement bonus
        p0 = self.game.agent_loc
        n = self.movement_bonus_period
        if len(self._prior_positions) >= n:
            p1 = self._prior_positions[-n]
            dist = abs(p0[0]-p1[0]) + abs(p0[1]-p1[1])
        elif len(self._prior_positions) > 0:
            p1 = self._prior_positions[0]
            dist = abs(p0[0]-p1[0]) + abs(p0[1]-p1[1])
            # If we're at the beginning of an episode, treat the
            # agent as if it were moving continuously before entering.
            dist += n - len(self._prior_positions)
        else:
            dist = n
        speed = dist / n
        reward += self.movement_bonus * speed**self.movement_bonus_power
        if self.as_penalty:
            reward -= self.movement_bonus
        self._prior_positions.append(self.game.agent_loc)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._prior_positions = queue.deque(
            [self.game.agent_loc], self.movement_bonus_period)
        return obs


class ContinuingEnv(Wrapper):
    """
    Change to a continuing (rather than episodic) environment.

    The episode only ever ends if the 'times_up' flag gets set to True.
    """
    def reset(self):
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
        if done and not info['times_up']:
            reward += call(self.bonus) * self.episode_reward
        return obs, reward, done, info


class MinPerformanceScheduler(BaseWrapper):
    """
    Provide a mechanism to set the `min_performance` for each episode.

    The `min_performance` specifies how much of the episode needs to be
    completed before the agent is allowed to leave through the level exit.
    The benchmark levels typically have `min_performance = 0.5`, but it can
    be helpful to start learning at a much lower value.
    """
    min_performance = 0.01

    def reset(self):
        obs = self.env.reset()
        self.game.min_performance = call(self.min_performance)
        return obs


class SimpleSideEffectPenalty(BaseWrapper):
    """
    Penalize departures from starting or inaction state.
    """
    penalty_coef = 0.0
    baseline = "starting-state"  # or "inaction"

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
        board = self.game.board & ~CellTypes.player
        baseline_board = self.baseline_board & ~CellTypes.player

        # Also ignore exit locations (they change color when they open up)
        i1, i2 = self.game.exit_locs
        board[i1,i2] = baseline_board[i1,i2]

        # Finally, ignore any cells that are part of the reward.
        # This takes into account red cells and blue goals, but not other
        # potential rewards (other colors). Suitable for most training levels.
        red_life = CellTypes.alive | CellTypes.color_r
        start_red = baseline_board & red_life == red_life
        end_red = board & red_life == red_life
        goal_cell = self.game.goals & CellTypes.rainbow_color == CellTypes.color_b
        end_alive = board & red_life == CellTypes.alive
        unchanged = board == baseline_board
        non_effects = unchanged | (start_red & ~end_red) | (goal_cell & end_alive)

        side_effect = np.sum(~non_effects)
        delta_effect = side_effect - self.last_side_effect
        reward -= delta_effect * call(self.penalty_coef)
        self.last_side_effect = side_effect
        return observation, reward, done, info
