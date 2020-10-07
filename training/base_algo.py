import os
import glob
import logging

import torch
import numpy as np

from .utils import nested_getattr, nested_setattr, named_output

logger = logging.getLogger(__name__)


class BaseAlgo(object):
    """
    Common methods for model checkpointing in pytorch.

    Attributes
    ----------
    checkpoint_directory : str
        The directory where checkpoints are stored. If not set, the checkpoint
        directory will be taken from ``self.data_logger.logdir``.
    data_logger : object
    num_steps : int
        Total number of training steps. It's assumed that subclasses will
        increment this in their training loops.
    checkpoint_interval : int
        Interval between subsequent checkpoints
    num_checkpoints : int
        Total number of checkpoints to maintain the logging directory.
        Older checkpoints that exceed this number are deleted.
    checkpoint_attribs : list
        List of attributes on the algorithm that ought to be saved at each
        checkpoint. This should be overridden by subclasses.
        Note that this implicitly contains ``num_steps``.
    """
    data_logger = None

    num_steps = 0

    checkpoint_interval = 100000
    max_checkpoints = 3
    checkpoint_attribs = []

    _last_checkpoint = -1
    _checkpoint_directory = None

    @property
    def checkpoint_directory(self):
        return self._checkpoint_directory or (
            self.data_logger and self.data_logger.logdir)

    @checkpoint_directory.setter
    def checkpoint_directory(self, value):
        self._checkpoint_directory = value

    def get_all_checkpoints(self):
        """
        Return a sorted list of all checkpoints in the log directory.
        """
        chkpt_dir = self.checkpoint_directory
        if not chkpt_dir:
            return []
        files = glob.glob(os.path.join(chkpt_dir, 'checkpoint-*.data'))

        def step_from_checkpoint(f):
            try:
                return int(os.path.basename(f)[11:-5])
            except ValueError:
                return -1

        files = [f for f in files if step_from_checkpoint(f) >= 0]
        return sorted(files, key=step_from_checkpoint)

    def save_checkpoint_if_needed(self):
        if self._last_checkpoint < 0:
            self.save_checkpoint()
        elif self._last_checkpoint + self.checkpoint_interval < self.num_steps:
            self.save_checkpoint()
        else:
            pass  # already have a recent checkpoint

    def save_checkpoint(self, filename=None):
        chkpt_dir = self.checkpoint_directory
        if not chkpt_dir:
            return

        data = {'num_steps': self.num_steps}
        for attrib in self.checkpoint_attribs:
            try:
                val = nested_getattr(self, attrib)
            except AttributeError:
                logger.error("Cannot save attribute '%s'", attrib)
                continue
            if hasattr(val, 'state_dict'):
                val = val.state_dict()
            data[attrib] = val

        if not filename:
            filename = 'checkpoint-%i.data' % self.num_steps
        path = os.path.join(chkpt_dir, filename)
        torch.save(data, path)
        logger.info("Saving checkpoint: '%s'", path)

        old_checkpoints = self.get_all_checkpoints()
        for old_checkpoint in old_checkpoints[:-self.max_checkpoints]:
            os.remove(old_checkpoint)

        self._last_checkpoint = self.num_steps

    def load_checkpoint(self, checkpoint_name=None):
        chkpt_dir = self.checkpoint_directory
        if checkpoint_name and os.path.dirname(checkpoint_name):
            # Path includes a directory.
            # Treat it as a complete path name and ignore chkpt_dir
            path = checkpoint_name
        elif chkpt_dir and checkpoint_name:
            path = os.path.join(chkpt_dir, checkpoint_name)
        else:
            checkpoints = self.get_all_checkpoints()
            path = checkpoints and checkpoints[-1]
        if not path or not os.path.exists(path):
            return

        logger.info("Loading checkpoint: %s", path)

        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))

        for key, val in checkpoint.items():
            orig_val = nested_getattr(self, key, None)
            if hasattr(orig_val, 'load_state_dict'):
                orig_val.load_state_dict(val)
            else:
                try:
                    nested_setattr(self, key, val)
                except AttributeError:
                    logger.error("Cannot load key '%s'", key)

        self._last_checkpoint = self.num_steps

    def tensor(self, data, dtype):
        """
        Shorthand for creating a tensor with the current compute device.

        Note that this is *much* faster than passing data in list form to
        ``torch.tensor`` directly, at least as of torch v1.3.
        See https://github.com/pytorch/pytorch/issues/13918 for more details.
        """
        data = np.asanyarray(data)
        return torch.as_tensor(data, device=self.compute_device, dtype=dtype)

    def obs_for_envs(self, envs):
        """
        Return current observations and agent ids for a list of environments.

        If the environments are multi-agent, then the number of returned
        observations will not generally match the number of environments because
        there can be more than (or fewer than) one agent per environment.

        This should be used in conjunction with `act_on_envs()`.
        Note that together they add attributes `last_obs`, `last_done`, and
        `num_resets` to the environment itself.
        """
        obs_list = []
        active = []
        agent_ids = []
        for env in envs:
            if hasattr(env, 'last_obs'):
                obs = env.last_obs
                done = env.last_done
            else:
                obs = env.reset()
                if getattr(env, 'single_agent', True):
                    obs = np.asanyarray(obs)[np.newaxis]
                env.last_done = done = np.tile(False, len(obs))
                env.num_resets = 0
            for k in range(len(obs)):
                agent_ids.append((id(env), env.num_resets, k))
            obs_list.append(obs)
            active.append(~done)

        obs_list = np.concatenate(obs_list)
        active = np.concatenate(active)
        # Make an array of agent ids, but keep each element of the array
        # a tuple so that they can be used as dictionary keys.
        agent_id_arr = np.zeros(len(agent_ids), dtype=object)
        agent_id_arr[:] = agent_ids

        return obs_list[active], agent_id_arr[active]

    def act_on_envs(self, envs, actions):
        """
        Return observations, rewards, and done flags for each environment.

        The number of actions should match the total number of active agents
        in each environment, which should also match the number of observations
        returned by `obs_for_envs()`.

        This should be used in conjunction with `obs_for_envs()`.
        Note that together they add attributes `last_obs`, `last_done`, and
        `num_resets` to the environment itself.
        """
        obs_list = []
        reward_list = []
        done_list = []

        k = 0
        for env in envs:
            single_agent = getattr(env, 'single_agent', True)
            active = ~env.last_done
            num_active = np.sum(active)
            if num_active == 0:
                continue
            active_actions = actions[k:k+num_active]
            assert len(active_actions) == num_active
            action_shape = (len(active),) + np.asanyarray(active_actions[0]).shape
            env_actions = np.zeros_like(active_actions[0], shape=action_shape)
            env_actions[active] = active_actions
            k += num_active
            if single_agent:
                obs, reward, done, info = env.step(env_actions[0])
                obs = np.asanyarray(obs)[np.newaxis]
                reward = np.array([reward])
                done = np.array([done])
            else:
                obs, reward, done, info = env.step(env_actions)
            obs_list.append(obs[active])
            reward_list.append(reward[active])
            done_list.append(done[active])

            if np.all(done):
                obs = env.reset()
                if getattr(env, 'single_agent', True):
                    obs = np.asanyarray(obs)[np.newaxis]
                done = np.tile(False, len(obs))
                env.num_resets += 1
            env.last_obs = obs
            env.last_done = done

        return (
            np.concatenate(obs_list),
            np.concatenate(reward_list),
            np.concatenate(done_list),
        )

    @named_output('obs actions rewards done next_obs agent_ids')
    def take_one_step(self, envs):
        """
        Take one step in each of the environments.

        This returns a set of arrays, with one value for each agent.
        Environments can contain more than one agent (or no agents at all),
        so the number of items in each array won't generally match the number
        of environments.

        This function should be implemented by subclasses to execute the
        subclass's policy.

        Returns
        -------
        obs : list
        actions : list
        rewards : list
        done : list
            Whether or not each environment reached its end this step.
        next_obs : list
        agent_ids : list
            A unique identifier for each agent. This can be used to string
            multiple steps together.
        """
        # Example:
        # obs, agent_ids = self.obs_for_envs(envs)
        # (calculate actions from the observations)
        # next_obs, rewards, done = self.act_on_envs(envs, actions)
        # return obs, actions, rewards, done, agent_ids
        raise NotImplementedError

    def run_episodes(self, envs, num_episodes=None, validation=False):
        """
        Run each environment to completion.

        Note that no data is logged in this method. It's instead assumed
        that each environment has a wrapper which takes care of the logging.

        Parameters
        ----------
        envs : list
            List of environments to run in parallel.
        num_episodes : int
            Total number of episodes to run. Defaults to the same as number
            of environments.
        validation : bool
            True if these are validation episodes & we should do associated
            housekeeping.
        """
        if not envs:
            return
        if num_episodes is None:
            num_episodes = len(envs)
        num_completed = 0

        sl_logger = getattr(envs[0], 'logger', None)
        if sl_logger is not None:
            sl_logger.reset_summary()

        while num_completed < num_episodes:
            data = self.take_one_step(envs)
            num_in_progress = len(envs)
            new_envs = []
            for env, done in zip(envs, data.done):
                done = np.all(done)
                if done:
                    num_completed += 1
                if done and num_in_progress + num_completed > num_episodes:
                    num_in_progress -= 1
                else:
                    new_envs.append(env)
            envs = new_envs
            if num_completed == 1:
                config.check_for_unused_hyperparams()

        if sl_logger is not None:
            sl_logger.log_summary()

        # Keep the agent that performs best on the validation ("test") levels
        if validation:
            best_so_far = sl_logger.check_for_best_agent()
            if best_so_far is not False:
                logger.info("New best performance on validation levels: %f", best_so_far)
                self.save_checkpoint(filename="best-validation-agent.data")
