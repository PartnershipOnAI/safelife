from scipy import interpolate

import numpy as np
import numpy.random as npr

from safelife import env_wrappers
from safelife.random import coinflip
from safelife.level_iterator import SafeLifeLevelIterator
from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife.safelife_logger import SafeLifeLogWrapper


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


class CurricularLevelIterator(SafeLifeLevelIterator):
    """
    Iterate through a curriculum of [typically increasingly challenging] level tyepes

    Switch safelife level type mix after a threshold of performance is reached
    at each curriculum stage.
    """
    curr_progression_mid = 0.47
    curr_progression_span = 0.25
    progression_lottery_ticket = 0.3  # max chance of progression per epoch
    revision_param = 2.0              # pareto param, lower -> more revision of past curriculum grades

    def __init__(self, levels, logger, **kwargs):
        super().__init__(*levels, repeat_levels=True, **kwargs)
        self.logger = logger
        self.curriculum_stage = 0
        self.max_stage = len(levels) - 1
        self.curr_currently_playing = 0
        self.best_score_by_level = {}
        self.best_perf_by_level = {}
        self.just_advanced = False

    def get_next_parameters(self):
        if len(self.results) > 0:
            _data, result = self.results[-1]
            game = result.get()
            performance = game.current_points()
        else:
            performance = 0.0

        self.just_advanced = False  # watch out for timing between this and self.results.append()
        pop = self.probability_of_progression(performance)
        if self.curr_currently_playing == self.curriculum_stage:  # we played at the curriculum frontier
            if coinflip(pop):
                if self.curriculum_stage < self.max_stage:
                    self.curriculum_stage += 1
                    self.best_score_by_level[self.curriculum_stage] = 0
                    self.best_perf_by_level[self.curriculum_stage] = 0.
                    self.logger.info("Curriculum advanced to level %d" % self.curriculum_stage)
                    self.just_advanced = True
        revision = int(np.clip(npr.pareto(self.revision_param), 0, self.curriculum_stage))
        self.curr_currently_playing = self.curriculum_stage - revision  # pick next stage; current = next
        # Tensorflow-dependent logging...
        # summary = tf.Summary()
        # stage = self.curriculum_stage
        # summary.value.add(tag='episode/performance', simple_value=performance)
        # summary.value.add(tag='curriculum/stage', simple_value=stage)
        # summary.value.add(tag='curriculum/pr_progression', simple_value=pop)
        # summary.value.add(tag='curriculum/level', simple_value=self.level)
        # summary.value.add(tag='curriculum/best_score_this_lvl',
        #                   simple_value=self.best_score_by_level[stage])
        # summary.value.add(tag='curriculum/best_perf_this_lvl',
        #                   simple_value=self.best_perf_by_level[stage])
        # self.logger.add_summary(summary, self.num_steps)

        return self.file_data[self.curr_currently_playing]

    def probability_of_progression(self, score):
        """
        The probability of graduating to the next curriculum stage is a sigmoid over peformance
        on the current one, active between curr_progression_mid +/- span.
        """
        def sigmoid(x):
            return 1.0 / (1 + np.exp(-x))

        rel_score = (score - self.curr_progression_mid) * 6.0 / (self.curr_progression_span)

        return sigmoid(rel_score) * self.progression_lottery_ticket


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
                CellTypes.color_bit + 16,  # red goal
                CellTypes.color_bit + 17,  # green goal
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
