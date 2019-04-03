import os
import logging

from gym import Wrapper
from gym.wrappers.monitoring import video_recorder

logger = logging.getLogger(__name__)


class VideoRecorder(video_recorder.VideoRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enabled:
            logger.info("Starting video: %s", self.path)

    def write_metadata(self):
        # The metadata file is pretty useless, so don't write it.
        pass

    def close(self):
        if self.enabled:
            logger.info("Ending video: %s", self.path)
        super().close()


class VideoMonitor(Wrapper):
    def __init__(self, env, directory, video_name_callback=None):
        super().__init__(env)

        self.video_recorder = None
        self.directory = os.path.abspath(directory)

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            os.makedirs(directory, exist_ok=True)
        if not callable(video_name_callback):
            raise ValueError("'video_name_callback' must in fact be callable.")
        self.video_name_callback = video_name_callback

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.video_recorder is not None:
            self.video_recorder.capture_frame()
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.reset_video_recorder()
        return observation

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super().close()

    def reset_video_recorder(self):
        video_name = self.video_name_callback()
        if self.video_recorder is not None:
            self.video_recorder.close()
        if video_name:
            logger.info("Starting new video: {}".format(video_name))
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path=os.path.join(self.directory, video_name)
            )
            self.video_recorder.capture_frame()
        else:
            self.video_recorder = None

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()


class AutoResetWrapper(Wrapper):
    """
    A top-level wrapper that automatically resets an environment when done
    and does some basic logging of episode rewards.
    """
    def __init__(self, env):
        super().__init__(env)
        self._state = None
        self.num_episodes = -1

    @property
    def state(self):
        if self._state is None:
            self._state = self.reset()
        return self._state

    def reset(self, **kwargs):
        self.episode_length = 0
        self.episode_reward = 0.0
        self.num_episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_length += 1
        self.episode_reward += reward
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        info['num_episodes'] = self.num_episodes
        if done:
            obs = self.reset()
        self._state = obs
        return obs, reward, done, info
