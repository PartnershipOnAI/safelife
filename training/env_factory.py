from scipy import interpolate

from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife import env_wrappers
from safelife.safelife_logger import SafeLifeLogWrapper
from safelife.level_iterator import SafeLifeLevelIterator


class LinearSchedule(object):
    """
    Piecewise linear schedule based on total number of training steps.

    This is useful to vary training parameters over the course of training.

    Parameters
    ----------
    logger : SafeLifeLogger
    t : list
        Input (training step) values that define the interpolation.
    y : list
        Output interpolation values.
    """
    def __init__(self, logger, t, y):
        self.logger = logger
        self.func = interpolate.UnivariateSpline(t, y, s=0, k=1, ext='const')

    def __call__(self):
        return self.func(self.logger.cumulative_stats['training_steps'])


class SwitchingLevelIterator(SafeLifeLevelIterator):
    """
    Switch safelife level types after a certain number of training steps.
    """
    def __init__(self, level1, level2, t_switch, logger, **kwargs):
        super().__init__(level1, level2, repeat_levels=True, **kwargs)
        self.t_switch = t_switch
        self.logger = logger

    def get_next_parameters(self):
        t = self.logger.cumulative_stats['training_steps']
        if t < self.t_switch:
            return self.file_data[0]
        else:
            return self.file_data[1]


def safelife_env_factory(
        level_iterator, *,
        num_envs=1,
        min_performance=None,
        data_logger=None,
        impact_penalty=None,
        penalty_baseline='starting-state',
        testing=False):
    """
    Factory for creating SafeLifeEnv instances with useful wrappers.
    """
    envs = []
    for _ in range(num_envs):
        env = SafeLifeEnv(
            level_iterator,
            view_shape=(25,25),
            # This is a minor optimization, but a few of the output channels
            # are redundant or unused for normal safelife training levels.
            output_channels=(
                CellTypes.alive_bit,
                CellTypes.agent_bit,
                CellTypes.pushable_bit,
                CellTypes.destructible_bit,
                CellTypes.frozen_bit,
                CellTypes.spawning_bit,
                CellTypes.exit_bit,
                CellTypes.color_bit + 0,  # red
                CellTypes.color_bit + 1,  # green
                CellTypes.color_bit + 2,  # blue
                CellTypes.color_bit + 18,  # blue goal
                CellTypes.orientation_bit + 0,
                CellTypes.orientation_bit + 1,
            ))

        if not testing:
            env = env_wrappers.MovementBonusWrapper(env, as_penalty=True)
            env = env_wrappers.ExtraExitBonus(env)
        if impact_penalty is not None:
            env = env_wrappers.SimpleSideEffectPenalty(
                env, penalty_coef=impact_penalty, baseline=penalty_baseline)
        if min_performance is not None:
            env = env_wrappers.MinPerformanceScheduler(
                env, min_performance=min_performance)
        env = SafeLifeLogWrapper(
            env, logger=data_logger, is_training=not testing)
        envs.append(env)

    return envs
