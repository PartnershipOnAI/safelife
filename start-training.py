#!/usr/bin/env python3

"""
Main entry point for starting a training job.
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


parser = argparse.ArgumentParser(description="""
    Run agent training using proximal policy optimization.

    This will set up the data/log directories, optionally install any needed
    dependencies, start tensorboard, configure loggers, and start the actual
    training loop. If the data directory already exists, it will prompt for
    whether the existing data should be overwritten or appended. The latter
    allows for training to be restarted if interrupted.
    """)
parser.add_argument('data_dir', nargs='?',
    help="the directory in which to store this run's data")
parser.add_argument('--run-type', choices=('train', 'benchmark', 'inspect'),
    default='train',
    help="What to do once the algorithm and environments have been loaded. "
    "If 'train', train the model. If 'benchmark', run the model on testing "
    "environments. If 'inspect', load an ipython prompt for interactive "
    "debugging.")
parser.add_argument('--algo', choices=('ppo', 'dqn'), default='ppo')
parser.add_argument('-e', '--env-type', default='append-spawn')
parser.add_argument('-s', '--steps', type=float, default=6e6,
    help='Length of training in steps (default: 6e6).')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--deterministic', action="store_true",
    help="If set, uses deterministic cudnn routines. This may slow "
    "down training, but it should make the results reproducable.")

parser.add_argument('-p', '--impact-penalty', type=float)
parser.add_argument('--penalty-baseline',
    choices=('starting-state', 'inaction'), default='starting-state')
parser.add_argument('--curriculum', default="progress_estimate", type=str,
    help='Curriculum type ("uniform" or "progress_estimate")')

parser.add_argument('--port', type=int,
    help="Port on which to run tensorboard.")
parser.add_argument('-w', '--wandb', action='store_true',
    help='Use wandb for analytics.')
parser.add_argument('--ensure-gpu', action='store_true',
    help="Check that the machine we're running on has CUDA support")
args = parser.parse_args()


if args.seed is None:
    # Make sure the seed can be represented by floating point exactly.
    # This is just because we may want to pass it over the web, and javascript
    # doesn't have 64 bit integers.
    args.seed = np.random.randint(2**53)

assert args.wandb or args.data_dir, ("Either a data directory must be set or "
    "the wandb flag must be set. If wandb is set but there is no data "
    "directory, then a run name will be picked automatically.")

if args.ensure_gpu:
    assert torch.cuda.is_available(), "CUDA support requested but not available!"


# Build the C extensions and only _then_ import safelife modules.

safety_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(1, safety_dir)  # ensure current directory is on the path
subprocess.run([
    "python3", os.path.join(safety_dir, "setup.py"),
    "build_ext", "--inplace"
])

from safelife.random import set_rng  # noqa
from safelife.safelife_logger import SafeLifeLogger, summarize_run # noqa
from training import logging_setup  # noqa
from training import models  # noqa
from training.env_factory import build_environments  # noqa
from training.global_config import config  # noqa


# Check to see if the data directory is already in use

if args.data_dir is not None:
    data_dir = os.path.realpath(args.data_dir)
    job_name = os.path.basename(data_dir)
    if os.path.exists(data_dir) and args.run_type == 'train':
        print("The directory '%s' already exists." % data_dir)
        print("Would you like to overwrite the old data, append to it, or abort?")
        response = 'overwrite' if job_name.startswith('tmp') else None
        while response not in ('overwrite', 'append', 'abort'):
            response = input("(overwrite / append / abort) > ")
        if response == 'overwrite':
            print("Overwriting old data.")
            shutil.rmtree(data_dir)
        elif response == 'abort':
            print("Aborting.")
            exit()
else:
    data_dir = job_name = None


# Setup wandb and initialize logging

if args.wandb:
    import wandb
    if wandb.login():
        run_notes = os.path.join(safety_dir, 'run-notes.txt')
        if os.path.exists(run_notes):
            run_notes = open(run_notes).read()
        else:
            run_notes = None

        # Remove some args from the wandb config, just to make things neater.
        # These args don't actually affect the run output.
        wandb.init(name=job_name, notes=run_notes, config={
            k: v for k, v in vars(args).items() if k not in
            ['port', 'wandb', 'ensure_gpu']
        })
        # Note that wandb config can contain different and/or new keys that
        # aren't in the command-line arguments. This is especially true for
        # wandb sweeps.
        config.update(wandb.config._items)

        if job_name is None:
            job_name = wandb.run.name
            data_dir = os.path.join(
                safety_dir, 'data', time.strftime("%Y-%m-%d-") + wandb.run.id)

        logging_setup.save_code_to_wandb()
else:
    wandb = None
    config.update(vars(args))

os.makedirs(data_dir, exist_ok=True)
logger = logging_setup.setup_logging(
    data_dir, debug=(config['run_type'] == 'inspect'))
logger.info("COMMAND ARGUMENTS: %s", ' '.join(sys.argv))
logger.info("TRAINING RUN: %s", job_name)
logger.info("ON HOST: %s", platform.node())


# Set the global seed

main_seed = np.random.SeedSequence(config['seed'])
logger.info("SETTING GLOBAL SEED: %i", main_seed.entropy)
set_rng(np.random.default_rng(main_seed))
torch.manual_seed(main_seed.entropy & (2**31 - 1))
if config['deterministic']:
    # Note that this may slow down performance
    # See https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    torch.backends.cudnn.deterministic = True


# Run tensorboard

if args.port:
    tb_proc = subprocess.Popen([
        "tensorboard", "--logdir_spec",
        job_name + ':' + data_dir, '--port', str(args.port)])
else:
    tb_proc = None


# Start training!

try:
    envs = build_environments(config, main_seed, data_dir)
    obs_shape = envs['training'][0].observation_space.shape

    algo_args = {
        'training_envs': envs['training'],
        'testing_envs': envs.get('testing'),
        'data_logger': logging_setup.setup_data_logger(data_dir, 'training'),
    }

    if config['algo'] == 'ppo':
        from training.ppo import PPO as algo_cls
        algo_args['model'] = models.SafeLifePolicyNetwork(obs_shape)
    elif config['algo'] == 'dqn':
        from training.dqn import DQN as algo_cls
        algo_args['training_model'] = models.SafeLifeQNetwork(obs_shape)
        algo_args['target_model'] = models.SafeLifeQNetwork(obs_shape)
    else:
        logging.error("Unexpected algorithm type '%s'", config['algo'])
        raise ValueError("unexpected algorithm type")

    algo = algo_cls(**algo_args)

    if args.wandb:
        # Before we start running things, save the config object back to wandb.
        config2 = config.copy()
        config2.pop('_wandb', None)
        wandb.config.update(config2)

    if config['run_type'] == "train":
        algo.train(int(config['steps']))
        if 'benchmark' in envs:
            algo.run_episodes(envs['benchmark'], num_episodes=1000)
    elif config['run_type'] == "benchmark" and "benchmark" in envs:
        algo.run_episodes(envs['benchmark'], num_episodes=1000)
    elif config['run_type'] == "inspect":
        from IPython import embed
        print('')
        embed()

    if config['run_type'] in ['train', 'benchmark'] and wandb:
        benchmark_file = os.path.join(data_dir, 'benchmark-data.json')
        summarize_run(benchmark_file, wandb.run)
        wandb.run.summary['env_type'] = config['env_type']


except Exception:
    logging.exception("Ran into an unexpected error. Aborting training.")
    raise
finally:
    if tb_proc is not None:
        tb_proc.kill()
