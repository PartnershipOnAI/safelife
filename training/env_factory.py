import logging
import os
from collections import defaultdict
from functools import partial

from scipy import interpolate
from scipy.special import softmax

import numpy as np
import numpy.random as npr


from safelife import env_wrappers
from safelife.helper_utils import load_kwargs
from safelife.level_iterator import SafeLifeLevelIterator
from safelife.random import coinflip

from safelife.render_graphics import render_file
from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife.safelife_logger import SafeLifeLogWrapper

from .logging_setup import setup_data_logger


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
    Iterate through a curriculum of [typically increasingly challenging] level types

    Switch safelife level type mix after a threshold of performance is reached
    at each curriculum stage.
    """
    curr_progression_mid = 0.47
    curr_progression_span = 0.25
    progression_lottery_ticket = 0.9  # max chance of progression per epoch
    revision_param = 2.0              # pareto param, lower -> more revision of past curriculum grades
    eval_lookback = 10
    eval_nth_best = 3
    lookback = 100  # base performance estimates on the last 100 episodes of each level
    curriculum_distribution = "progress_estimate"  # or "uniform"

    def __init__(self, *levels, logger, curriculum_params={}, **kwargs):
        super().__init__(*levels, repeat_levels=True, **kwargs)
        self.logger = logger
        self.curriculum_stage = 0
        self.max_stage = len(levels) - 1
        self.curr_currently_playing = 0
        self.just_advanced = False
        self.perf_records = defaultdict(lambda: [0.0])  # map level to history of performance
        self.best = defaultdict(lambda: 0)
        load_kwargs(self, curriculum_params)

    def progression_statistic(self, results):
        n = self.eval_lookback
        if len(results) < n:
            return 0
        # return the 3rd best result from the past ten episodes
        pool = np.array(results[-n:])
        return np.quantile(pool, 1 - (self.eval_nth_best / n))

    def update_result_records(self):
        "Housekeeping with results of the most recently completed episode."
        results = self.logger.last_data
        filename = None
        if results is not None:
            reward = np.array(results['reward'])
            reward_possible = np.array(results['reward_possible'])
            filename = self.logger.last_game.file_name
            if reward.size > 0:
                performance = np.average(reward / reward_possible)
                if np.isnan(performance) or np.isinf(performance):
                    performance = 0
                    logger.info("perf was nan-y")
                self.perf_records[filename].append(performance)
                if performance > self.best[filename]:
                    self.best[filename] = performance
                    self.record_video(os.path.basename(filename), performance)

    def get_next_parameters(self):
        "Choose a next level to play based on softmax'd estimates of dperf/dtrain"

        self.update_result_records()
        # Default to a large estimate when there isn't enough information
        # about training performance on a level: 20% performance gained in
        # [lookback] levels would be a very large perf gain
        training_progress = 0.2 * np.ones(self.max_stage + 1) / self.lookback

        for i, entry in enumerate(self.file_data):
            level = entry[0]
            if len(self.perf_records[level]) >= self.lookback:
                dom = np.arange(self.lookback)
                m, c = np.polyfit(dom, self.perf_records[level][-self.lookback:], 1)
                training_progress[i] = 10 * m

        logger.debug("Progress: %s", training_progress)
        scale = np.min(np.abs(training_progress))
        training_progress = training_progress.clip(0, None)
        training_progress = training_progress / scale
        exploding = np.isnan(training_progress) | np.isinf(training_progress)
        training_progress[exploding] = 0.0
        if self.curriculum_distribution == "progress_estimate":
            probabilities = softmax(training_progress)
        elif self.curriculum_distribution == "uniform":
            probabilities = np.ones(self.max_stage + 1) / (self.max_stage + 1)
        else:
            raise ValueError("invalid curriculum distribution type")
        choice = npr.choice(self.max_stage + 1, p=probabilities)
        logger.debug("Probabilities: %s, chose %s", probabilities, choice)

        record = {}
        for i, entry in enumerate(self.file_data):
            level = entry[0]
            record["normalised_progress_lvl{}".format(i)] = training_progress[i]
            record["probability_lvl{}".format(i)] = probabilities[i]
            record["best_perf_lvl{}".format(i)] = self.best[level]
            recent = self.perf_records[level][-self.lookback:]
            rperf = np.average(recent) if len(recent) > 0 else 0.0
            record["recent{}_perf_lvl{}".format(self.lookback, i)] = rperf
        self.logger.log_scalars(record)

        return self.file_data[choice]

    def record_video(self, lvl, perf):
        filename = "best_score-{}-{}.npz".format(lvl, perf)
        path = os.path.join(self.logger.logdir, filename)
        np.savez_compressed(path, **self.logger.last_history)
        render_file(path, movie_format="mp4")


class SwitchingLevelIterator(SafeLifeLevelIterator):
    """
    Switch safelife level types based on a coin flip.

    Parameters
    ----------
    level1, level2 : str
        Level files.
    p_switch : callable
        Probability of picking level 2 instead of level 1.
    """
    def __init__(self, level1, level2, p_switch, **kwargs):
        super().__init__(level1, level2, repeat_levels=True, **kwargs)
        self.p_switch = p_switch

    def get_next_parameters(self):
        if coinflip(self.p_switch()):
            return self.file_data[1]
        else:
            return self.file_data[0]


task_types = {
    # Single-agent tasks:
    'append-still': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/append-still-easy'],
        'validation_levels': ['random/append-still'],
        'benchmark_levels': 'benchmarks/v1.0/append-still.npz',
    },
    'prune-still': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/prune-still'],
        'validation_levels': ['random/prune-still'],
        'benchmark_levels': 'benchmarks/v1.0/prune-still.npz',
    },
    'append-spawn': {
        'iter_class': SwitchingLevelIterator,
        'train_levels': ['random/append-still-easy', 'random/append-spawn'],
        'validation_levels': ['random/append-spawn'],
        'benchmark_levels': 'benchmarks/v1.0/append-spawn.npz',
    },
    'prune-spawn': {
        'iter_class': SwitchingLevelIterator,
        'train_levels': ['random/prune-still', 'random/prune-spawn'],
        'validation_levels': ['random/prune-spawn'],
        'benchmark_levels': 'benchmarks/v1.0/prune-spawn.npz',
    },
    'curriculum-append-spawn': {
        'iter_class': CurricularLevelIterator,
        'train_levels': ['random/append-still-easy', 'random/append-spawn'],
        'validation_levels': ['random/append-spawn'],
        'benchmark_levels': 'benchmarks/v1.0/append-spawn.npz',
    },
    'navigate': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/navigation'],
        'validation_levels': ['random/navigate'],
        'benchmark_levels': 'benchmarks/v1.0/navigation.npz',
    },

    # Multi-agent tasks:
    'asym1': {
        'iter_class': CurricularLevelIterator,
        'train_levels': ['random/multi-agent/asym1'],
        'validation_levels': ['random/multi-agent/asym1'],
        'multiagent': True,
    },
    'curriculum-asym1': {
        'iter_class': CurricularLevelIterator,
        'train_levels': [
            'random/multi-agent/asym1',
            'random/multi-agent/asym1-pretrain-cyanonly',
            'random/multi-agent/asym1-pretrain-redonly'],
        'validation_levels': ['random/multi-agent/asym1'],
        'multiagent': True,
    },
    'multi-build-coop': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/multi-agent/build-coop'],
        'validation_levels': ['random/multi-agent/build-coop'],
        'multiagent': True,
    },
    'multi-build-compete': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/multi-agent/build-compete'],
        'validation_levels': ['random/multi-agent/build-compete'],
        'multiagent': True,
    },
    'multi-build-parallel': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/multi-agent/build-parallel'],
        'validation_levels': ['random/multi-agent/build-parallel'],
        'multiagent': True,
    },
    'multi-prune': {
        'iter_class': SafeLifeLevelIterator,
        'train_levels': ['random/prune-still', 'random/multi-agent/prune-still'],
        'validation_levels': ['random/multi-agent/prune-still'],
        'multiagent': True,
    },
}


def safelife_env_factory(
        level_iterator, *,
        num_envs=1,
        env_args={},
        data_logger=None,
        training=True,
        exit_difficulty=1.0,
        se_baseline='starting-state',
        se_penalty=0.0):
    """
    Factory for creating SafeLifeEnv instances with useful wrappers.
    """
    envs = []
    for _ in range(num_envs):
        env = SafeLifeEnv(level_iterator, **env_args)

        if training:
            env = env_wrappers.MovementBonusWrapper(env, as_penalty=True)
            env = env_wrappers.ExtraExitBonus(env)
            env = env_wrappers.SimpleSideEffectPenalty(env,
                baseline=se_baseline, penalty_coef=se_penalty)
            env = env_wrappers.MinPerformanceScheduler(env,
                min_performance_fraction=exit_difficulty)
        env = SafeLifeLogWrapper(env, logger=data_logger)
        envs.append(env)

    return envs


def build_environments(config, seed=None, data_dir=None):
    task = config['env_type']

    assert task in task_types, "'%s' is not a recognized task" % (task,)

    if not isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(seed)
    training_seed, benchmark_seed = seed.spawn(2)

    task_data = task_types[task]

    # common arguments for all environments
    view_size = config.setdefault('env.view_size', 25)
    common_env_args = {
        'single_agent': not task_data.get('multiagent'),
        'view_shape': (view_size, view_size),
        'side_effect_weights': {
            'life-green': 1.0,
            'spawner-yellow': 2.0,
        },
        # This is a minor optimization, but a few of the output channels
        # are redundant or unused for normal safelife training levels.
        'output_channels': (
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
        )
    }

    # Training environments

    training_logger = setup_data_logger(data_dir, 'training')
    schedule = partial(LinearSchedule, training_logger)

    iter_class = task_data.get('iter_class', SafeLifeLevelIterator)
    iter_args = {'seed': training_seed}

    if iter_class is CurricularLevelIterator:
        iter_args['logger'] = training_logger
        iter_args['curriculum_params'] = {
            'curriculum_distribution': config.setdefault(
                'env.curriculum', 'progress_estimate')
        }
    elif iter_class is SwitchingLevelIterator:
        task_schedule = config.setdefault('env.task_switch', {
            't': [1e5, 1.5e6],
            'y': [0.1, 1.0],
        })
        iter_args['p_switch'] = schedule(**task_schedule)

    training_iter = iter_class(*task_data['train_levels'], **iter_args)

    # Side effect penalty for training
    se_penalty = config.setdefault('side_effect.penalty', 0.0)
    se_baseline = config.setdefault('side_effect.baseline', 'starting-state')
    se_schedule = config.setdefault('side_effect.schedule', {
        't': [1e6, 2e6],
        'y': [0, 1.0],
    }).copy()
    se_schedule['y'] = np.array(se_schedule['y']) * se_penalty

    # Exit difficulty for training.
    exit_difficulty = config.setdefault('env.exit_difficulty', {
        't': [5e5, 2e6],
        'y': [0.001, 1.0],
    })

    envs = {}
    envs['training'] = safelife_env_factory(
        training_iter, num_envs=16, training=True, env_args=common_env_args,
        data_logger=training_logger,
        se_baseline=se_baseline, se_penalty=schedule(**se_schedule),
        exit_difficulty=schedule(**exit_difficulty),
    )

    # Validation environments

    # Note that we typically want to keep the seed for testing environments
    # constant so that we get the same validation levels for multiple runs.
    # The number chosen here is just a random number. Nothing special about it.
    validation_seed = config.setdefault('validation.env_seed', 732230218323780641)

    validation_levels = task_data.get('validation_levels')
    num_validation_levels = config.setdefault('validation.num_levels', 5)
    if validation_levels:
        envs['validation'] = safelife_env_factory(
            num_envs=num_validation_levels, training=False,
            env_args=common_env_args,
            data_logger=setup_data_logger(data_dir, 'validation'),
            level_iterator=SafeLifeLevelIterator(
                *validation_levels, seed=validation_seed, num_workers=0,
                repeat_levels=True, distinct_levels=num_validation_levels))

    # Benchmark environments

    # These are only run at the very end of training.
    # The seed only matters for stochastic dynamics, because the benchmark
    # levels themselves are fixed. The seed is spawned off of the main seed
    # and will generally be different from run to run.

    benchmark_levels = task_data.get('benchmark_levels')
    if benchmark_levels:
        envs['benchmark'] = safelife_env_factory(
            num_envs=20, training=False, env_args=common_env_args,
            data_logger=setup_data_logger(data_dir, 'benchmark'),
            level_iterator=SafeLifeLevelIterator(
                benchmark_levels, seed=benchmark_seed, num_workers=0,
                repeat_levels=True))

    return envs
