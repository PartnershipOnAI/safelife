from scipy import interpolate

import numpy as np
import numpy.random as npr

import logging
import os
from collections import defaultdict

from safelife import env_wrappers
from safelife.helper_utils import load_kwargs
from safelife.level_iterator import SafeLifeLevelIterator
from safelife.random import coinflip
from safelife.render_graphics import render_file
from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife.safelife_logger import SafeLifeLogWrapper


logger = logging.getLogger(__name__)


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
    progression_lottery_ticket = 0.9  # max chance of progression per epoch
    revision_param = 2.0              # pareto param, lower -> more revision of past curriculum grades
    eval_lookback = 10
    eval_nth_best = 3

    def progression_statistic(self, results):
        n = self.eval_lookback
        if len(results) < n:
            return 0
        # return the 3rd best result from the past ten episodes
        pool = np.array(results[-n:])
        return np.quantile(pool, 1 - (self.eval_nth_best / n))

    def __init__(self, *levels, logger, curriculum_params={}, **kwargs):
        super().__init__(*levels, repeat_levels=True, **kwargs)
        self.logger = logger
        self.curriculum_stage = 0
        self.max_stage = len(levels) - 1
        self.curr_currently_playing = 0
        self.just_advanced = False
        self.perf_records = defaultdict(lambda: [0.0])  # map level to history of performance
        self.pops = defaultdict(lambda: [0.0])
        load_kwargs(self, curriculum_params)

    def get_last_results(self):
        """
        Extract results of the most recently completed episode

        Returns
        -------
        results: dict
            log_data for the run
        performance: float
            proportion of maximum possible reward
        logstring: str
            brief description of whether a score was obtained
        """
        results = self.logger.last_data
        filename = None
        if results is not None:
            reward = np.array(results['reward'])
            reward_possible = np.array(results['reward_possible'])
            filename = self.logger.last_game.file_name
            if reward.size > 0:
                performance = np.average(reward / reward_possible)
                logstring = "Scoring from result {}".format(performance)
                self.perf_records[filename].append(performance)
                pop = self.probability_of_progression(performance)
                self.pops[filename].append(pop)
            else:
                performance = 0.0
                logstring = "Null score, using {}".format(performance)
        else:
            logstring = "Skipped result"
            performance = 0.0
        return filename, logstring

    def get_next_parameters(self):
        "Get the next level to play, managing curriculum progression along the way."
        filename, scorelog = self.get_last_results()

        self.just_advanced = False  # watch out for timing between this and self.results.append()
        pop = self.progression_statistic(self.pops[filename])
        print(filename, self.file_data[0])
        if filename:
            # if we played at the curriculum frontier
            if filename == self.file_data[self.curriculum_stage][0]:
                # and we scored high enough, progress to the next curriculum stage
                if coinflip(pop) and self.curriculum_stage < self.max_stage:
                    self.curriculum_stage += 1
                    self.just_advanced = True

        if self.just_advanced:
            # create a video of the episode that caused curriculum progression
            assert self.logger.last_game, "Mysterious advancement"
            assert self.logger.last_history, "Historical amnesia"
            filename = "curricular_advancement{}.npz".format(self.curriculum_stage - 1)
            path = os.path.join(self.logger.logdir, filename)
            np.savez_compressed(path, **self.logger.last_history)
            render_file(path, movie_format="mp4")
            logger.info("{}; Curriculum advanced to stage {} on POP {:0%}".format(
                        scorelog, self.curriculum_stage, pop))
        else:
            logger.info("{}; No curriculum advance; POP is {:.0%}".format(scorelog, pop))

        revision = int(np.clip(npr.pareto(self.revision_param), 0, self.curriculum_stage))
        self.curr_currently_playing = self.curriculum_stage - revision  # pick next stage; current = next
        return self.file_data[self.curr_currently_playing]

    def probability_of_progression(self, score):
        """
        The probability of graduating to the next curriculum stage is a sigmoid over peformance
        on the current one, active between curr_progression_mid +/- span.
        """
        # XXX Maybe switch to a beta distribution for finite support?
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


task_types = {
    'append-still': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/append-still-easy'],
        'test_levels': 'benchmarks/v1.0/append-still.npz',
        'schedule': [1e6, 2e6],
    },
    'multi-build-coop': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/multi-agent/build-coop'],
        'schedule': [1.5e6, 3e6],
    },
    'multi-build-compete': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/multi-agent/build-compete'],
        'schedule': [1.5e6, 3e6],
    },
    'multi-build-parallel': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/multi-agent/build-parallel'],
        'schedule': [1.5e6, 3e6],
    },
    'prune-still': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/prune-still-easy'],
        'test_levels': 'benchmarks/v1.0/prune-still.npz',
        'schedule': [0.5e6, 1.5e6],
    },
    'multi-prune': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/prune-still', 'random/multi-agent/prune-still'],
        'schedule': [1.5e6, 3e6],
    },
    'append-spawn': {
        'iter_class': SwitchingLevelIterator,
        'train_levels': ['random/append-still-easy', 'random/append-spawn'],
        'test_levels': 'benchmarks/v1.0/append-spawn.npz',
        'schedule': [1e6, 2e6],
        't_switch': 1.5e6,
    },
    'prune-spawn': {
        'iter_class': SwitchingLevelIterator,
        'train_levels': ['random/prune-still-easy', 'random/prune-spawn'],
        'test_levels': 'benchmarks/v1.0/prune-spawn.npz',
        'schedule': [0.5e6, 2e6],
        't_switch': 1.5e6,
    },
    'curriculum-append-spawn': {
        'iter_class': CurricularLevelIterator,
        'train_levels': ['random/append-still-easy', 'random/append-spawn'],
        'test_levels': 'benchmarks/v1.0/append-spawn.npz',
        'schedule': [1e6, 2e6],
    },
    'navigate': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/navigation'],
        'test_levels': 'benchmarks/v1.0/navigation.npz',
        'schedule': [1e6, 2e6],
    },
}


def build_environments(
        task, run_type='train', seed=None, data_logger=None,
        impact_penalty=None, penalty_baseline='starting-state'):
    assert task in task_types, "'%s' is not a recognized task" % (task,)

    task_data = task_types[task]
    iter_class = task_data.get('iter_class', SafeLifeLevelIterator)
    iter_args = {'seed': seed}
    if iter_class is SwitchingLevelIterator:
        iter_args['t_switch'] = task_data['t_switch']
        iter_args['logger'] = data_logger
    elif iter_class is CurricularLevelIterator:
        iter_args['logger'] = data_logger

    training_iter = iter_class(*task_data['train_levels'], **iter_args)
    schedule = task_data['schedule']
    if impact_penalty is not None:
        impact_penalty = LinearSchedule(data_logger, schedule, [0, impact_penalty])
    training_envs = safelife_env_factory(
        training_iter, data_logger=data_logger, num_envs=16,
        impact_penalty=impact_penalty, penalty_baseline=penalty_baseline,
        min_performance=LinearSchedule(data_logger, schedule, [0.01, 0.5]),
    )

    test_levels = task_data.get('test_levels')
    if run_type == "benchmark" and test_levels:
        testing_envs = safelife_env_factory(
            data_logger=data_logger, num_envs=20, testing=True,
            level_iterator=SafeLifeLevelIterator(
                test_levels, repeat_levels=True)
        )
    elif test_levels:
        testing_envs = safelife_env_factory(
            data_logger=data_logger, num_envs=5, testing=True,
            level_iterator=SafeLifeLevelIterator(
                test_levels, distinct_levels=5, repeat_levels=True)
        )
    else:
        testing_envs = None

    return training_envs, testing_envs
