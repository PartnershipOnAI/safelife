import os
import glob
import logging

import torch

logger = logging.getLogger(__name__)


class BaseAlgo(object):
    """
    Common methods for model checkpointing in pytorch.

    Attributes
    ----------
    data_logger : SafeLifeLogger
        The data logger points to the logging directory and contains a
        dictionary of ``cumulative_stats`` that should be saved along with
        any model attributes.
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
    """
    data_logger = None

    num_steps = 0

    checkpoint_interval = 100000
    max_checkpoints = 3
    checkpoint_attribs = []

    _last_checkpoint = -1

    def get_all_checkpoints(self):
        """
        Return a sorted list of all checkpoints in the log directory.
        """
        logdir = self.data_logger and self.data_logger.logdir
        if not logdir:
            return []
        files = glob.glob(os.path.join(logdir, 'checkpoint-*.data'))

        def step_from_checkpoint(f):
            try:
                return int(os.path.basename(f)[11:-5])
            except ValueError:
                return -1

        files = [f for f in files if step_from_checkpoint(f) >= 0]
        return sorted(files, key=step_from_checkpoint)

    def save_checkpoint(self):
        if self.data_logger is None:
            return
        if (self._last_checkpoint >= 0 and
                self.num_steps < self._last_checkpoint + self.checkpoint_interval):
            # Already have a recent checkpoint.
            return

        logdir = self.data_logger.logdir
        path = os.path.join(logdir, 'checkpoint-%i.data' % self.num_steps)

        data = {'logger_stats': self.data_logger.cumulative_stats}
        data['logger_stats']['training_steps'] = self.num_steps

        for attrib in self.checkpoint_attribs:
            val = getattr(self, attrib)
            if hasattr(val, 'state_dict'):
                val = val.state_dict()
            data[attrib] = val
        torch.save(data, path)
        logger.info("Saving checkpoint: '%s'", path)

        old_checkpoints = self.get_all_checkpoints()
        for old_checkpoint in old_checkpoints[:-self.max_checkpoints]:
            os.remove(old_checkpoint)

        self._last_checkpoint = self.num_steps

    def load_checkpoint(self, checkpoint_name=None):
        if self.data_logger is None:
            return
        logdir = self.data_logger.logdir
        if checkpoint_name is not None:
            path = os.path.join(logdir, checkpoint_name)
        else:
            checkpoints = self.get_all_checkpoints()
            path = checkpoints and checkpoints[-1]
        if not path or not os.path.exists(path):
            return

        checkpoint = torch.load(path)
        self.data_logger.cumulative_stats = checkpoint['logger_stats']
        self.num_steps = checkpoint['logger_stats']['training_steps']

        for key, val in checkpoint.items():
            orig_val = getattr(self, key, None)
            if hasattr(orig_val, 'load_state_dict'):
                orig_val.load_state_dict(val)
            else:
                setattr(self, key, val)

        self._last_checkpoint = self.num_steps
