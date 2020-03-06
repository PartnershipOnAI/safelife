"""
A short set of utilities for saving pytorch models for SafeLife.
"""

import os
import glob
import logging

import torch

logger = logging.getLogger(__name__)


def get_all_checkpoints(logdir):
    files = glob.glob(os.path.join(logdir, 'checkpoint-*.data'))

    def step_from_checkpoint(f):
        try:
            return int(os.path.basename(f)[11:-5])
        except ValueError:
            return -1

    files = [f for f in files if step_from_checkpoint(f) >= 0]
    return sorted(files, key=step_from_checkpoint)


def save_checkpoint(safelife_logger, obj, attribs, max_checkpoints=3):
    if safelife_logger is None:
        return
    num_steps = safelife_logger.cumulative_stats['training_steps']
    logdir = safelife_logger.logdir
    path = os.path.join(logdir, 'checkpoint-%i.data' % num_steps)

    data = {'logger_stats': safelife_logger.cumulative_stats}
    for attrib in attribs:
        val = getattr(obj, attrib)
        if hasattr(val, 'state_dict'):
            val = val.state_dict()
        data[attrib] = val
    torch.save(data, path)
    logger.info("Saving checkpoint: '%s'", path)

    for old_checkpoint in get_all_checkpoints(logdir)[:-max_checkpoints]:
        os.remove(old_checkpoint)


def load_checkpoint(safelife_logger, obj, checkpoint_name=None):
    if safelife_logger is None:
        return
    logdir = safelife_logger.logdir
    if checkpoint_name is not None:
        path = os.path.join(logdir, checkpoint_name)
    else:
        checkpoints = get_all_checkpoints(logdir)
        path = checkpoints and checkpoints[-1]
    if not path or not os.path.exists(path):
        return

    checkpoint = torch.load(path)
    safelife_logger.cumulative_stats = checkpoint['logger_stats']

    for key, val in checkpoint.items():
        orig_val = getattr(obj, key, None)
        if hasattr(orig_val, 'load_state_dict'):
            orig_val.load_state_dict(val)
        else:
            setattr(obj, key, val)
