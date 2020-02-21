"""
A short set of utilities for saving pytorch models for SafeLife.

These use the global counter on SafeLifeEnv, but are otherwise SafeLife
agnostic.
"""

import os
import glob
import logging

import torch

from safelife.safelife_env import SafeLifeEnv

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


def save_checkpoint(path, obj, attribs, max_checkpoints=3):
    num_steps = SafeLifeEnv.global_counter.num_steps
    if os.path.isdir(path):
        logdir = path
        path = os.path.join(path, 'checkpoint-%i.data' % num_steps)
    else:
        logdir = os.path.dirname(path)

    data = {
        'num_steps': num_steps,
        'num_episodes': SafeLifeEnv.global_counter.episodes_completed,
    }
    for attrib in attribs:
        val = getattr(obj, attrib)
        if hasattr(val, 'state_dict'):
            val = val.state_dict()
        data[attrib] = val
    torch.save(data, path)
    logger.info("Saving checkpoint: '%s'", path)

    for old_checkpoint in get_all_checkpoints(logdir)[:-max_checkpoints]:
        os.remove(old_checkpoint)


def load_checkpoint(path, obj):
    if os.path.isdir(path):
        checkpoints = get_all_checkpoints(path)
        path = checkpoints and checkpoints[-1]
    if not path or not os.path.exists(path):
        return

    checkpoint = torch.load(path)

    for key, val in checkpoint.items():
        orig_val = getattr(obj, key, None)
        if hasattr(orig_val, 'load_state_dict'):
            orig_val.load_state_dict(val)
        else:
            setattr(obj, key, val)

    SafeLifeEnv.global_counter.num_steps = checkpoint['num_steps']
    SafeLifeEnv.global_counter.episodes_started = checkpoint['num_episodes']
    SafeLifeEnv.global_counter.episodes_completed = checkpoint['num_episodes']
