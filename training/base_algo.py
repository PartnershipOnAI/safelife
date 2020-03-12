import os
import glob
import logging

import torch

from .utils import nested_getattr, nested_setattr

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
    checkpoint_directory = None
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
        self,_checkpoint_directory = value

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

    def save_checkpoint(self):
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

        path = os.path.join(chkpt_dir, 'checkpoint-%i.data' % self.num_steps)
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

        checkpoint = torch.load(path)

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

    def take_one_step(self, envs):
        """
        Take one step in each of the environments.

        Returns
        -------
        states : list
        actions : list
        rewards : list
        done : list
            Whether or not each environment reached its end this step.
        """
        raise NotImplementedError

    def run_episodes(self, envs, num_episodes=None):
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
        """
        if num_episodes is None:
            num_episodes = len(envs)
        num_completed = 0

        while num_completed < num_episodes:
            data = self.take_one_step(envs)
            num_in_progress = len(envs)
            new_envs = []
            for env, done in zip(envs, data.done):
                if done:
                    num_completed += 1
                if done and num_in_progress + num_completed > num_episodes:
                    num_in_progress -= 1
                else:
                    new_envs.append(env)
            envs = new_envs
