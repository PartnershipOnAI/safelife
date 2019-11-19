import os
from types import SimpleNamespace

import yaml
import numpy as np

from .file_finder import safelife_loader
from .safelife_env import SafeLifeEnv
from . import env_wrappers


def run_benchmark(
        name, policy, logfile, num_trials=1, record=True, num_env=10,
        env_factory=SafeLifeEnv):
    """
    Run benchmark levels for a specific policy.

    Parameters
    ----------
    name : str
        Benchmark name. E.g., "v1.0/append-still"
    policy : function
        Function that maps observations to policies.
        Inputs will be a list of observations and a list of prior RNN states.
        Output should be a list of policies (probabilities of each action)
        and a list of next RNN states. If the policy function is stateless,
        then output can just be a list of dummy values of the same length as
        the observations.
    logfile : str
        The file in which to store the benchmark output.
    num_trials : int, optional
        Number of times to test each benchmark level.
    record : bool, optional
        If True, every trial is video recorded and saved in the same directory
        as the log file.
    num_env : int, optional
        Number of environments to run simultaneously.
    env_factory : function
        Function to build new SafeLifeEnv instances. Must accept at least two
        arguments: ``level_iterator`` and ``global_counter``. See
        :class:`safelife_env.SafeLifeEnv` for more details.
    """
    logfile = os.path.abspath(os.path.expanduser(logfile))
    logdir = os.path.split(logfile)[0]
    if os.path.exists(logfile):
        logfile = open(logfile, 'a')  # append to it
    else:
        os.makedirs(logdir, exist_ok=True)
        logfile = open(logfile, 'w')
        logfile.write("# SafeLife benchmark data\n---\n")
        logfile.flush()

    levels = safelife_loader(
        os.path.join("benchmarks", "v1.0", name), repeat=num_trials)
    counter = SimpleNamespace(
        episodes_started=0,
        episodes_completed=0,
        num_steps=0
    )
    if record:
        video_name = "benchmark-{level_title} ({episode_num})"
        video_name = os.path.join(logdir, video_name)
    else:
        video_name = None

    envs = []
    obs = []
    rnn_state = [None] * num_env
    for k in range(num_env):
        env = env_factory(level_iterator=levels, global_counter=counter)
        # Note that basically all the logging happens in the wrapper.
        env = env_wrappers.RecordingSafeLifeWrapper(
            env, log_file=logfile, video_name=video_name, video_recording_freq=1)
        try:
            obs.append(env.reset())
        except StopIteration:
            break
        envs.append(env)

    # Now we just run the environments until they run out.
    envs0 = envs
    try:
        t = 0
        while envs:
            policies, rnn_state = policy(obs, rnn_state)
            new_obs = []
            new_rnn_state = []
            new_envs = []
            t += 1
            print("t = %i, episodes completed = %i" % (
                t, counter.episodes_completed))
            for p, env, state in zip(policies, envs, rnn_state):
                action = np.random.choice(len(p), p=p)
                ob, r, done, info = env.step(action)
                if done:
                    try:
                        ob = env.reset()
                        state = None
                    except StopIteration:
                        continue
                new_obs.append(ob)
                new_envs.append(env)
                new_rnn_state.append(state)
            obs = new_obs
            envs = new_envs
            rnn_state = new_rnn_state
    finally:
        logfile.close()
        for env in envs0:
            env.close()


def load_benchmarks(logfile):
    """
    Load benchmark data into a dictionary of numpy arrays.
    """
    with open(logfile) as logfile:
        data = yaml.safe_load(logfile)
    keys = set()
    side_keys = set()
    for episode in data:
        keys.update(episode.keys())
        side_keys.update(episode.get('side effects', {}).keys())
    keys.discard('side effects')
    stats = {key:[] for key in keys}
    side_effects = {key:[] for key in side_keys}
    for episode in data:
        for key in stats:
            stats[key].append(episode.get(key, 0))
        episode_effects = episode.get('side effects', {})
        for key in side_effects:
            side_effects[key].append(episode_effects.get(key, [0.0,0.0]))
    for key in stats:
        stats[key] = np.array(stats[key])
    for key in side_effects:
        side_effects[key] = np.array(side_effects[key])
    stats['side effects'] = side_effects
    return stats
