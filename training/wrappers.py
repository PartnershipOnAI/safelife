import os
import logging
import numpy as np

from gym import Wrapper
from gym.wrappers.monitoring import video_recorder

logger = logging.getLogger(__name__)


class SafeLifeRecorder(video_recorder.VideoRecorder):
    """
    Record agent trajectories and videos for SafeLife.

    Note that this is pretty particular to the SafeLife environment, as it
    also outputs a numpy array of states that the agent traverses.
    """

    def __init__(self, env, enabled=True, base_path=None):
        super().__init__(env, enabled=enabled, base_path=base_path)
        self.base_path = base_path
        if self.enabled:
            name = os.path.split(self.path)[1]
            logger.info("Starting video: %s", name)
            self.trajectory = {
                "orientation": [],
                "board": [],
                "goals": []
            }

    def write_metadata(self):
        # The metadata file is pretty useless, so don't write it.
        pass

    def capture_frame(self):
        super().capture_frame()
        # Also capture the state in numpy mode to make it easy to analyze
        # or re-render the trajectory later.
        if self.enabled:
            state = self.env.unwrapped.state
            self.trajectory['orientation'].append(state.orientation)
            self.trajectory['board'].append(state.board.copy())
            self.trajectory['goals'].append(state.goals.copy())

    def close(self):
        if self.enabled:
            name = os.path.split(self.path)[1]
            logger.info("Ending video: %s", name)
            np.savez(self.base_path + '.npz', **self.trajectory)
        super().close()


class SafeLifeWrapper(Wrapper):
    """
    A wrapper for the SafeLife environment to handle recording and parameter
    updating.

    Parameters
    ----------
    env : SafeLifeEnv instance
    reset_callback : callable
        Called whenever the environment is about to reset, using `self` as the
        single argument. Useful for updating either the name of the video
        recording or any underlying parameters of the SafeLife environment.
    video_name : str
        If set, the environment will record a video and save the raw trajectory
        information for the next episode. This attribute can be changed between
        episodes (via `reset_callback`) to either disable recording or give the
        recording a new name.
    on_name_conflict : str
        What to do if the video name conflicts with an existing file.
        One of "abort" (don't record a new video), "overwrite", or
        "change_name".
    """
    def __init__(
            self, env, reset_callback=None, video_name=None,
            on_name_conflict="change_name"):
        if reset_callback is not None and not callable(reset_callback):
            raise ValueError("'reset_callback' must be a callable")
        self.reset_callback = reset_callback
        self.video_name = video_name
        self.video_idx = 1  # used only if the name is duplicated
        self.video_recorder = None
        self.on_name_conflict = on_name_conflict
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.video_recorder is not None:
            self.video_recorder.capture_frame()
        return observation, reward, done, info

    def reset(self, **kwargs):
        if self.reset_callback is not None:
            self.reset_callback(self)
        observation = self.env.reset(**kwargs)
        self.reset_video_recorder()
        return observation

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super().close()

    def reset_video_recorder(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        if self.video_name:
            path = p0 = os.path.abspath(self.video_name)
            directory = os.path.split(path)[0]
            if not os.path.exists(directory):
                os.makedirs(directory)
            if self.on_name_conflict == "change_name":
                while os.path.exists(path + '.npz'):
                    # If the video name already exists, add a counter to it.
                    path = p0 + '-' + str(self.video_idx)
                    self.video_idx += 1
            elif self.on_name_conflict == "abort":
                if os.path.exists(path + '.npz'):
                    return
            elif self.on_name_conflict != "overwrite":
                raise ValueError("Invalid value for 'on_name_conflict'")
            self.video_recorder = SafeLifeRecorder(env=self.env, base_path=path)
            self.video_recorder.capture_frame()

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()


class AutoResetWrapper(Wrapper):
    """
    A top-level wrapper that automatically resets an environment when done
    and does some basic logging of episode rewards.
    """
    def __init__(self, env, reset_callback=None):
        super().__init__(env)
        self._obs = None
        self.num_episodes = -1
        self.reset_callback = reset_callback

    @property
    def obs(self):
        if self._obs is None:
            self._obs = self.reset()
        return self._obs

    def reset(self, **kwargs):
        self.episode_length = 0
        self.episode_reward = 0.0
        self.num_episodes += 1
        if self.reset_callback is not None:
            self.reset_callback(self)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_length += 1
        self.episode_reward += info.get('base_reward', reward)
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        info['num_episodes'] = self.num_episodes
        if done:
            obs = self.reset()
        self._obs = obs
        return obs, reward, done, info
