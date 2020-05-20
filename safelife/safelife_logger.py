"""
Logging utilities for recording SafeLife episodes and episode statistics.

This module contains a number of classes to make logging in SafeLife easier.
The `SafeLifeLogger` class does the bulk of the actual logging work: it
maintains handles and writes to test and training logs, writes data to
tensorboard, and records agent trajectories as movies and data archives.

There are two main functions that `SafeLifeLogger`, and, more generally, the
`BaseLogger` base class, implement. The `log_episode()` function logs
statistics for a single SafeLife episode, and is generally called by instances
of the `SafeLifeLogWrapper` class. The `log_scalars()` function logs arbitrary
scalar statistics to tensorboard. This can be used from within training
algorithms to monitor training progress (loss, value functions, etc.).
There is also a `cumulative_stats` attribute that contains the total number of
training episodes and steps taken, which can be helpful for setting
hyperparameter training schedules in the training algorithm or for setting a
curriculum for the environment itself.

The `RemoteSafeLifeLogger` class has the same interface, but it's suitable
for use in multiprocessing environments that use Ray. The actual logging work
is delegated to a remote actor with `RemoteSafeLifeLogger` instances holding on
to references to that actor. Importantly, this means that `RemoteSafeLifeLogger`
instances can be copied within or between processes without competing for
access to a single open log or tensorboard file.

Finally, the `SafeLifeLogWrapper` class can wrap `SafeLifeEnv` environment
instances to automatically log episodes upon completion. With this wrapper in
place, the training algorithms themselves don't actually need to log any extra
episode statistics; they just need to run episodes in the environment.
"""

import os
import time
import json
import textwrap
import logging
import logging.config
from datetime import datetime
from collections import defaultdict

import gym
import numpy as np

try:
    import ray
    ray_remote = ray.remote
except ImportError:
    ray = None
    def ray_remote(func): return func

from .helper_utils import load_kwargs
from .side_effects import side_effect_score
from .render_text import cell_name
from .render_graphics import render_file

logger = logging.getLogger(__name__)


class StreamingJSONWriter(object):
    """
    Serialize streaming data to JSON.

    This class holds onto an open file reference to which it carefully
    appends new JSON data. Individual entries are input in a list, and
    after every entry the list is closed so that it remains valid JSON.
    When a new item is added, the file cursor is moved backwards to overwrite
    the list closing bracket.
    """
    def __init__(self, filename, encoder=json.JSONEncoder):
        if os.path.exists(filename):
            self.file = open(filename, 'r+')
            self.delimeter = ','
        else:
            self.file = open(filename, 'w')
            self.delimeter = '['
        self.encoder = encoder

    def dump(self, obj):
        """
        Dump a JSON-serializable object to file.
        """
        data = json.dumps(obj, cls=self.encoder)
        close_str = "\n]\n"
        self.file.seek(max(self.file.seek(0, os.SEEK_END) - len(close_str), 0))
        self.file.write("%s\n    %s%s" % (self.delimeter, data, close_str))
        self.file.flush()
        self.delimeter = ','

    def close(self):
        self.file.close()


class BaseLogger(object):
    """
    Defines the interface for SafeLife loggers, both local and remote.
    """
    def __init__(self, logdir):
        self.logdir = logdir
        self.cumulative_stats = {
            'training_episodes': 0,
            'training_steps': 0,
            'testing_episodes': 0,
        }

    def log_episode(self, game, info={}, history=None, training=True):
        raise NotImplementedError

    def log_scalars(self, data, global_step=None, tag=None):
        raise NotImplementedError


class SafeLifeLogger(BaseLogger):
    """
    Logs episode statistics for SafeLife.

    Attributes
    ----------
    logdir : str
        Directory to save log data.
    cumulative_stats : dict
        Cumulative statistics for the training run. Includes
        ``training_steps``, ``training_episodes``, and ``testing_epsodes``.
        Note that this dictionary gets updated in place, so it can easily be
        passed to other functions to do e.g. hyperparameter annealing.
    training_video_name : str
        Format string for the training video files.
    testing_video_name : str
        Format string for the testing video files.
    training_video_interval : int
        Interval at which to save training videos. If 1, every episode is saved.
    testing_video_interval : int
        Interval at which to save testing videos.
    record_side_effects : bool
        If true (default), side effects are calculated at the end of each
        episode.
    summary_writer : tensorboardX.SummaryWriter
        Writes data to tensorboard. The SafeLifeLogger will attempt to create
        a new summary writer for the log directory if one is not supplied.
    """

    logdir = None
    cumulative_stats = None
    summary_writer = None

    training_video_name = "training-e{training_episodes}-s{training_steps}"
    testing_video_name = "testing-s{training_steps}-{level_name}"
    training_video_interval = 100
    testing_video_interval = 1

    training_log = "training-log.json"
    testing_log = "testing-log.json"

    record_side_effects = True

    _testing_log = None
    _training_log = None
    summary_writer = None

    console_training_msg = textwrap.dedent("""
        Training episode completed.
            level name: {level_name}
            episode #{training_episodes};  training steps = {training_steps}
            clock: {time}
            length: {length}
            reward: {reward} / {reward_possible} (exit cutoff = {reward_needed})
    """[1:-1])
    console_testing_msg = textwrap.dedent("""
        Testing episode completed.
            level name: {level_name}
            clock: {time}
            length: {length}
            reward: {reward} / {reward_possible} (exit cutoff = {reward_needed})
    """[1:-1])

    def __init__(self, logdir, **kwargs):
        load_kwargs(self, kwargs)

        self.logdir = logdir
        self.cumulative_stats = {
            'training_episodes': 0,
            'training_steps': 0,
            'testing_episodes': 0,
        }
        self._has_init = False

    def init_logdir(self):
        if not self._has_init and self.logdir:
            if self.testing_log:
                self._testing_log = StreamingJSONWriter(
                    os.path.join(self.logdir, self.testing_log))
            if self.training_log:
                self._training_log = StreamingJSONWriter(
                    os.path.join(self.logdir, self.training_log))
            if self.summary_writer is None:
                try:
                    from tensorboardX import SummaryWriter
                    self.summary_writer = SummaryWriter(self.logdir)
                except ImportError:
                    logger.error(
                        "Could not import tensorboardX. "
                        "SafeLifeLogger will not write data to tensorboard.")
        self._has_init = True

    def log_episode(self, game, info={}, history=None, training=True):
        """
        Log an episode. Outputs (potentially) to file, tensorboard, and video.

        Parameters
        ----------
        game : SafeLifeGame
        info : dict
            Episode data to log. Assumed to contain 'reward' and 'length' keys,
            as is returned by the ``SafeLifeEnv.step()`` function.
        history : dict
            Trajectory of the episode. Should contain keys 'board', 'goals',
            and 'orientations'.
        training : bool
            Whether to log output as a training or testing episode.
        """
        self.init_logdir()  # init if needed

        if training:
            self.cumulative_stats['training_episodes'] += 1
            self.cumulative_stats['training_steps'] += info.get('length', 0)
            num_episodes = self.cumulative_stats['training_episodes']
            history_name = (
                self.training_video_interval > 0 and
                (num_episodes - 1) % self.training_video_interval == 0 and
                self.training_video_name
            )
            console_msg = self.console_training_msg
        else:
            self.cumulative_stats['testing_episodes'] += 1
            num_episodes = self.cumulative_stats['testing_episodes']
            history_name = (
                self.testing_video_interval > 0 and
                (num_episodes - 1) % self.testing_video_interval == 0 and
                self.testing_video_name
            )
            console_msg = self.console_testing_msg

        # First, log to screen.
        log_data = info.copy()
        log_data.setdefault('reward', 0)
        log_data.setdefault('length', 0)
        log_data['level_name'] = game.title
        log_data['reward'] = float(log_data['reward'])
        log_data['reward_possible'] = float(game.initial_available_points)
        log_data['reward_needed'] = game.required_points()
        log_data['time'] = datetime.utcnow().isoformat()
        logger.info(console_msg.format(**log_data, **self.cumulative_stats))

        # Then log to file.
        if self.record_side_effects:
            log_data['side_effects'] = side_effects = {
                cell_name(cell): effect
                for cell, effect in side_effect_score(game).items()
            }
        if training and self._training_log is not None:
            self._training_log.dump(log_data)
        elif self._testing_log is not None:
            self._testing_log.dump(log_data)

        # Log to tensorboard.
        tb_data = info.copy()
        # Use a normalized reward
        tb_data['reward_frac'] = (
            log_data['reward'] / max(log_data['reward_possible'], 1))
        tb_data.pop('reward')
        if training:
            tb_data['total_episodes'] = self.cumulative_stats['training_episodes']
            tb_data['reward_frac_needed'] = game.min_performance
        if self.record_side_effects and 'life-green' in side_effects:
            amount, total = side_effects['life-green']
            tb_data['side_effects'] = amount / max(total, 1)
        tag = "training_runs" if training else "testing_runs"
        self.log_scalars(tb_data, tag=tag)

        # Finally, save a recording of the trajectory.
        if history is not None and self.logdir is not None and history_name:
            history_name = history_name.format(**log_data, **self.cumulative_stats)
            history_name = os.path.join(self.logdir, history_name) + '.npz'
            if not os.path.exists(history_name):
                np.savez_compressed(history_name, **history)
                render_file(history_name, movie_format="mp4")

    def log_scalars(self, data, global_step=None, tag=None):
        """
        Log scalar values to tensorboard.

        Parameters
        ----------
        data : dict
            Dictionary of key/value pairs to log to tensorboard.
        tag : str or None

        """
        self.init_logdir()  # init if needed

        if not self.summary_writer:
            return
        tag = "" if tag is None else tag + '/'
        if global_step is None:
            global_step = self.cumulative_stats['training_steps']
        for key, val in data.items():
            if np.isreal(val) and np.isscalar(val):
                self.summary_writer.add_scalar(tag + key, val, global_step)
        self.summary_writer.flush()


class RemoteSafeLifeLogger(BaseLogger):
    """
    Maintains a local interface to a remote logging object using ray.

    The remote logging object is a ray Actor that does lightweight wrapping
    of a SafeLifeLogger instance. This means that the same RemoteSafeLifeLogger
    can be copied to different processes while maintaining a link to the same
    actor, retrieving the same global state, and writing to the same open files.

    Note that the ``cumulative_stats`` in the local copy will generally lag
    what is available on the remote copy. It is only updated whenever an
    episode is logged, and even then it is updated asynchronously.

    Parameters
    ----------
    logdir : str
        The directory in which to log everything.
    config_dict : dict
        A dictionary of options to pass to ``logging.config.dictConfig``
        in the standard python logging library. Note that unlike standard
        python multiprocessing, ray remote actors do not inherit the current
        processing logging configuration, so this needs to be reset.
    """
    max_backlog = 50
    update_interval = 0.01

    @ray_remote
    class SafeLifeLoggingActor(object):
        def __init__(self, logger, config_dict):
            self.logger = logger
            logger.init_logdir()
            if config_dict is not None:
                logging.config.dictConfig(config_dict)

        def log_episode(self, game, info, history, training):
            self.logger.log_episode(game, info, history, training)
            return self.logger.cumulative_stats

        def log_scalars(self, data, step, tag):
            self.logger.log_scalars(data, step, tag)

        def update_stats(self, cstats):
            self.logger.cumulative_stats = cstats

    def __init__(self, logdir, config_dict=None, **kwargs):
        if ray is None:
            raise ImportError("No module named 'ray'.")
        logger = SafeLifeLogger(logdir, **kwargs)
        self.logdir = logdir
        self.actor = self.SafeLifeLoggingActor.remote(logger, config_dict)
        self._cstats = logger.cumulative_stats.copy()

        # _promises stores references to remote updates to cumulative_stats
        # that will be received in response to having sent a log item. There
        # is no point exposing this state because there is in general no way
        # to get up-to-date statistics to any thread, and therefore no benefit
        # from knowing whether you're waiting for an update.
        self._promises = []

        self._last_update = time.time()

    @property
    def cumulative_stats(self):
        next_update = self._last_update + self.update_interval
        if self._promises and time.time() > next_update:
            timeout = 0 if len(self._promises) < self.max_backlog else None
            ready, self._promises = ray.wait(
                self._promises, len(self._promises), timeout=timeout)
            if ready:
                self._cstats = ray.get(ready[-1])
            self._last_update = time.time()
        return self._cstats

    @cumulative_stats.setter
    def cumulative_stats(self, stats):
        self._cstats = stats.copy()
        self.actor.update_stats.remote(stats)

    def log_episode(self, game, info, history=None, training=True):
        self._promises.append(
            self.actor.log_episode.remote(game, info, history, training))

    def log_scalars(self, data, step=None, tag=None):
        self.actor.log_scalars.remote(data, step, tag)


class SafeLifeLogWrapper(gym.Wrapper):
    """
    Records episode data and (optionally) full agent trajectories.

    Parameters
    ----------
    logger : SafeLifeLogger
        The logger performs the actual writing to disk.
        It should be an instance of SafeLifeLogger, or any other class that
        implements a ``log_episode()`` function.
    record_history : bool
        If True (default), the full agent trajectory is sent to the logger
        along with the game state and episode info dict.
    is_training : bool
        Flag passed along to the logger. Training and testing environments
        get logged somewhat differently.
    """

    logger = None
    record_history = True
    is_training = True

    def __init__(self, env, **kwargs):
        super().__init__(env)
        load_kwargs(self, kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self.record_history and not self._did_log_episode:
            game = self.env.game
            self._episode_history['board'].append(game.board)
            self._episode_history['goals'].append(game.goals)
            self._episode_history['orientation'].append(game.orientation)

        if done and not self._did_log_episode and self.logger is not None:
            self._did_log_episode = True
            self.logger.log_episode(
                game, info.get('episode', {}),
                self._episode_history if self.record_history else None,
                self.is_training)

        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()

        self._did_log_episode = False
        self._episode_history = {
            'board': [],
            'goals': [],
            'orientation': []
        }

        return observation


def load_safelife_log(logfile, default_values={}):
    """
    Load a SafeLife log file as a dictionary of arrays.

    This is *much* more space efficient than the json format, and generally
    much easier to analyze.

    Note that the returned dictionary can be saved to a numpy archive for
    efficient storage and fast retrieval. E.g., ::

        data = load_safelife_log('training-log.json')
        numpy.savez_compressed('training-log.npz', **data)

    Missing data is filled in with NaN.

    Parameters
    ----------
    logfile : str or file-like object
        Path of the file to load, or the file itself.
    default_values : dict
        Default values for rows with missing data.
        Each key should receive it's own missing value.
    """
    if hasattr(logfile, 'read'):
        data = json.load(logfile)
    else:
        data = json.load(open(logfile))
    arrays = defaultdict(list)
    indicies = defaultdict(list)

    def flatten_dict(d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                out.update({
                    key + '.' + k:v
                    for k,v in flatten_dict(val).items()
                })
            elif key == 'time':
                out['time'] = np.datetime64(val)
            else:
                out[key] = val
        return out

    for n, datum in enumerate(data):
        for key, val in flatten_dict(datum).items():
            arrays[key].append(val)
            indicies[key].append(n)

    outdata = {}
    for key, arr in arrays.items():
        try:
            arr1 = np.array(arr)
        except Exception:
            logger.error("Cannot load key: %s", key)
            continue
        dtype = arr1.dtype
        if str(dtype).startswith('<U'):
            # dtype is a unicode string
            default_val = ''
        elif str(dtype).startswith('<M'):
            # dtype is a datetime
            default_val = np.datetime64('nat')
        elif str(dtype) == 'object':
            logger.error("Cannot load key: %s", key)
            continue
        else:
            default_val = 0
        default_val = default_values.get(key, default_val)
        arr2 = np.empty((len(data),) + arr1.shape[1:], dtype=dtype)
        arr2[:] = default_val
        arr2[indicies[key]] = arr1
        outdata[key] = arr2
    return outdata
