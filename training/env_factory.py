import os
from scipy import interpolate

from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife import env_wrappers


def linear_schedule(t, y):
    """
    Piecewise linear function y(t)
    """
    return interpolate.UnivariateSpline(t, y, s=0, k=1, ext='const')


def safelife_env_factory(
        logdir, level_iterator, *,
        num_envs=1,
        min_performance=None,
        summary_writer=None,
        impact_penalty=None,
        testing=False):
    """
    Factory for creating SafeLifeEnv instances with useful wrappers.
    """
    if testing:
        video_name = "test-{level_title}-{step_num}"
        tag = "episodes/testing/"
        log_name = "testing.yaml"
        log_header = "# Testing episodes\n---\n"
    else:
        video_name = "training-episode-{episode_num}-{step_num}"
        tag = "episodes/training/"
        log_name = "training.yaml"
        log_header = "# Training episodes\n---\n"

    logdir = os.path.abspath(logdir)
    video_name = os.path.join(logdir, video_name)
    log_name = os.path.join(logdir, log_name)

    if os.path.exists(log_name):
        log_file = open(log_name, 'a')
    else:
        log_file = open(log_name, 'w')
        log_file.write(log_header)

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
                CellTypes.color_bit + 5,  # blue goal
            ))
        other_data = {}

        if testing:
            env.global_counter = None  # don't increment num_steps
        else:
            env = env_wrappers.MovementBonusWrapper(env, as_penalty=True)
            env = env_wrappers.ExtraExitBonus(env)
        if impact_penalty is not None:
            env = env_wrappers.SimpleSideEffectPenalty(
                env, penalty_coef=impact_penalty)
            if not testing:
                other_data = {'impact_penalty': impact_penalty}
        if min_performance is not None:
            env = env_wrappers.MinPerformanceScheduler(
                env, min_performance=min_performance)
        env = env_wrappers.RecordingSafeLifeWrapper(
            env, video_name=video_name, summary_writer=summary_writer,
            log_file=log_file, other_episode_data=other_data, tag=tag,
            video_recording_freq=1 if testing else 50,
            exclude=('num_episodes', 'performance_cutoff') if testing else ())
        # Ensure the recording wrapper has access to the global counter,
        # even if it's disabled in the unwrapped environment.
        env.global_counter = SafeLifeEnv.global_counter
        envs.append(env)

    return envs
