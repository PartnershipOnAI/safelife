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
import json

import torch
import numpy as np

logger = logging.getLogger('training')
safety_dir = os.path.realpath(os.path.dirname(__file__))


def parse_args(argv=sys.argv):
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
    parser.add_argument('--algo', choices=('ppo', 'mppo', 'dqn'), default='ppo')
    parser.add_argument('-e', '--env-type', default='append-spawn')
    parser.add_argument('-s', '--steps', type=float, default=6e6,
        help='Length of training in steps (default: 6e6).')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--deterministic', action="store_true",
        help="If set, uses deterministic cudnn routines. This may slow "
        "down training, but it should make the results reproducable.")
    parser.add_argument('--port', type=int,
        help="Port on which to run tensorboard.")
    parser.add_argument('-w', '--wandb', action='store_true',
        help='Use wandb for analytics.')
    parser.add_argument('--project', default=None,
        help='[Entity and] project for wandb. '
        'Eg: "safelife/multiagent" or "multiagent"')
    parser.add_argument('--shutdown', action="store_true",
        help="Shut down the system when the job is complete"
        "(helpful for running remotely).")
    parser.add_argument('--ensure-gpu', action='store_true',
        help="Check that the machine we're running on has CUDA support")
    parser.add_argument('-x', '--extra-params', default=None,
        help="Extra config values/hyperparameters. Should be loadable as JSON.")

    args = parser.parse_args(sys.argv[1:])

    if args.extra_params:
        try:
            args.extra_params = json.loads(args.extra_params)
            assert isinstance(args.extra_params, dict)
        except (json.JSONDecodeError, AssertionError):
            print(f"'{args.extra_params}' is not a valid JSON dictionary. "
                "Make sure to escape your quotes!")
            exit(1)

    assert args.wandb or args.data_dir or args.run_type == 'inspect', (
        "Either a data directory must be set or the wandb flag must be set. "
        "If wandb is set but there is no data directory, then a run name will be "
        "picked automatically.")

    if args.ensure_gpu:
        assert torch.cuda.is_available(), "CUDA support requested but not available!"

    return args


def build_c_extensions():
    subprocess.run([
        "python3", os.path.join(safety_dir, "setup.py"),
        "build_ext", "--inplace"
    ])


def setup_config_and_wandb(args):
    """
    Setup wandb, the logging directory, and update the global config object.

    There is a two-way sync between wandb and the parameter config.
    Any parameters that are set in args are passed to wandb, and any parameters
    that are set in wandb (if, e.g., this run is part of a parameter sweep) are
    loaded back into the config.

    Parameters
    ----------
    args : Namespace
        Values returned from parse_args

    Returns
    -------
    config : GlobalConfig
        The global configuration object. This is a singleton and can be
        imported directly, but it's helpful in this file to pass it around and
        so that function inputs are made explicit.
    job_name : str
    data_dir : str
        The directory where data gets written.
    """

    from training import logging_setup
    from training.global_config import config

    # Check to see if the data directory is already in use
    # If it is, prompt the user if they want to overwrite it.
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
        job_name = data_dir = None

    # Remove some args from the config, just to make things neater.
    # These args don't actually affect the run output.
    base_config = {
        k: v for k, v in vars(args).items() if k not in
        ['port', 'wandb', 'ensure_gpu', 'project', 'shutdown', 'extra_params']
    }

    # tag any hyperparams from the commandline
    if args.extra_params is not None:
        config.add_hyperparams(args.extra_params)

    if args.wandb:
        import wandb
        if wandb.login():
            run_notes = os.path.join(safety_dir, 'run-notes.txt')
            if os.path.exists(run_notes):
                run_notes = open(run_notes).read()
            else:
                run_notes = None

            if args.project and '/' in args.project:
                entity, project = args.project.split("/", 1)
            elif args.project:
                entity, project = None, args.project
            else:
                entity = project = None  # use values from wandb/settings

            wandb.init(
                name=job_name, notes=run_notes, project=project, entity=entity,
                config=base_config)
            # Note that wandb config can contain different and/or new keys that
            # aren't in the command-line arguments. This is especially true for
            # wandb sweeps.
            config.update(wandb.config._items)

            # Save the environment type to the wandb summary data.
            # This allows env_type show up in the benchmark table.
            wandb.run.summary['env_type'] = config['env_type']

            if job_name is None:
                job_name = wandb.run.name
                data_dir = os.path.join(
                    safety_dir, 'data', time.strftime("%Y-%m-%d-") + wandb.run.id)

            logging_setup.save_code_to_wandb()
    else:
        config.update(base_config)

    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)

    logging_setup.setup_logging(
        data_dir, debug=(config['run_type'] == 'inspect'))
    logger.info("COMMAND ARGUMENTS: %s", ' '.join(sys.argv))
    logger.info("TRAINING RUN: %s", job_name)
    logger.info("ON HOST: %s", platform.node())

    return config, job_name, data_dir


def set_global_seed(config):
    from safelife.random import set_rng

    # Make sure the seed can be represented by floating point exactly.
    # This is just because we want to pass it over the web, and javascript
    # doesn't have 64 bit integers.
    if config.get('seed') is None:
        config['seed'] = np.random.randint(2**53)
    seed = np.random.SeedSequence(config['seed'])
    logger.info("SETTING GLOBAL SEED: %i", seed.entropy)
    set_rng(np.random.default_rng(seed))
    torch.manual_seed(seed.entropy & (2**31 - 1))

    if config['deterministic']:
        # Note that this may slow down performance
        # See https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True


def launch_tensorboard(job_name, data_dir, port):
    """
    Launch tensorboard as a subprocess.

    Note that this process must be killed before the script exits.
    """
    if port and data_dir is not None:
        return subprocess.Popen([
            "tensorboard", "--logdir_spec",
            job_name + ':' + data_dir, '--port', str(port)])
    else:
        return None


def launch_training(config, data_dir):
    from training import logging_setup
    from training import models
    from training.env_factory import build_environments

    envs = build_environments(config, data_dir)
    obs_shape = envs['training'][0].observation_space.shape

    algo_args = {
        'training_envs': envs['training'],
        'testing_envs': envs.get('validation'),
        'data_logger': logging_setup.setup_data_logger(data_dir, 'training'),
    }

    if config['algo'] == 'ppo':
        from training.ppo import PPO as algo_cls
        algo_args['model'] = models.SafeLifePolicyNetwork(obs_shape)
    elif config['algo'] == 'dqn':
        from training.dqn import DQN as algo_cls
        algo_args['training_model'] = models.SafeLifeQNetwork(obs_shape)
        algo_args['target_model'] = models.SafeLifeQNetwork(obs_shape)
    elif config['algo'] == 'mppo':
        from training.ppo import LSTM_PPO as algo_cls
        algo_args['model'] = models.SafeLifeLSTMPolicyNetwork(obs_shape)
    else:
        logger.error("Unexpected algorithm type '%s'", config['algo'])
        raise ValueError("unexpected algorithm type")

    algo = algo_cls(**algo_args)

    if config.get('_wandb') is not None:
        # Before we start running things, save the config object back to wandb.
        import wandb
        config2 = config.copy()
        config2.pop('_wandb', None)
        wandb.config.update(config2, allow_val_change=True)

    print("")
    logger.info("Hyperparameters: %s", config)
    # config.check_for_unused_hyperparams()
    print("")

    if config['run_type'] == "train":
        algo.train(int(config['steps']))
        if 'benchmark' in envs:
            # "Early stopping"... load the best agent from training for the benchmark run
            algo.load_checkpoint("best-validation-agent.data")
            algo.run_episodes(envs['benchmark'], num_episodes=1000)
    elif config['run_type'] == "benchmark" and "benchmark" in envs:
        algo.run_episodes(envs['benchmark'], num_episodes=1000)
    elif config['run_type'] == "inspect":
        from IPython import embed
        print('')
        embed()


def cleanup(config, data_dir, tb_proc, shutdown):
    from safelife.safelife_logger import summarize_run

    if config.get('_wandb') is not None:
        import wandb
        wandb_run = wandb.run
    else:
        wandb_run = None

    try:
        if config['run_type'] in ['train', 'benchmark']:
            summarize_run(data_dir, wandb_run)
    except:  # noqa
        import traceback
        logger.warn("Exception during summarization:\n", traceback.format_exc())

    if wandb_run is not None:
        wandb_run.finish()

    if tb_proc is not None:
        tb_proc.kill()

    if shutdown:
        # Shutdown in 3 minutes.
        # Enough time to recover if it crashed at the start.
        subprocess.run("sudo shutdown +3", shell=True)
        logger.critical("Shutdown commenced, but keeping ssh available...")
        subprocess.run("sudo rm -f /run/nologin", shell=True)


def main():
    # We'll be importing safelife modules, so we need to make
    # sure that the safelife package is on the import path
    sys.path.insert(1, safety_dir)

    args = parse_args()
    build_c_extensions()
    config, job_name, data_dir = setup_config_and_wandb(args)
    set_global_seed(config)
    tb_proc = launch_tensorboard(job_name, data_dir, args.port)

    try:
        launch_training(config, data_dir)
    except KeyboardInterrupt:
        logger.critical("Keyboard Interrupt. Ending early.\n")
    except Exception:
        logger.exception("Ran into an unexpected error. Aborting training.\n")
    finally:
        cleanup(config, data_dir, tb_proc, args.shutdown)


if __name__ == "__main__":
    main()
