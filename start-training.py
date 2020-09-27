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

import numpy as np
import torch


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

    args = parser.parse_args(argv)

    if args.extra_params:
        try:
            args.extra_params = json.loads(args.extra_params)
            assert isinstance(args.extra_params, dict)
        except (json.JSONDecodeError, AssertionError):
            print(f"'{args.extra_params}' is not a valid JSON dictionary. "
                "Make sure to escape your quotes!")
            exit(1)
    return args


def build_and_import_modules(safety_dir):
    "Build the C extensions and only _then_ import safelife modules."

    subprocess.run([
        "python3", os.path.join(safety_dir, "setup.py"),
        "build_ext", "--inplace"
    ])

    global config, set_rng, SafeLifeLogger, summarize_run, logging_setup, models, build_environments

    from safelife.random import set_rng  # noqa
    from safelife.safelife_logger import SafeLifeLogger, summarize_run # noqa
    from training import logging_setup  # noqa
    from training import models  # noqa
    from training.env_factory import build_environments  # noqa
    from training.global_config import config  # noqa


def setup_config_and_wandb(args, launch_ingredients):
    "Setup wandb and finalise config object logging."

    # config is a global, from training.global_config, but we fill it out here
    # if wandb is in use, the config is synchronized with it, except these
    # parameters:

    base_config = {
        k: v for k, v in vars(args).items() if k not in
        ['port', 'wandb', 'ensure_gpu', 'project', 'shutdown', 'extra_params']
    }
    global wandb
    if args.wandb:
        import wandb
        if wandb.login():
            run_notes = os.path.join(launch_ingredients.safety_dir, 'run-notes.txt')
            if os.path.exists(run_notes):
                run_notes = open(run_notes).read()
            else:
                run_notes = None

            # Remove some args from the wandb config, just to make things neater.
            # These args don't actually affect the run output.
            if args.project and '/' in args.project:
                entity, project = args.project.split("/", 1)
            elif args.project:
                entity, project = None, args.project
            else:
                entity = project = None  # use values from wandb/settings

            wandb.init(
                name=launch_ingredients.job_name, notes=run_notes, project=project, entity=entity,
                config=base_config)
            # Note that wandb config can contain different and/or new keys that
            # aren't in the command-line arguments. This is especially true for
            # wandb sweeps.
            config.update(wandb.config._items)

            # Save the environment type to the wandb summary data.
            # This allows env_type show up in the benchmark table.
            wandb.run.summary['env_type'] = config['env_type']

            if launch_ingredients.job_name is None:
                launch_ingredients.job_name = wandb.run.name
                launch_ingredients.data_dir = os.path.join(
                    launch_ingredients.safety_dir, 'data', time.strftime("%Y-%m-%d-") + wandb.run.id)

            logging_setup.save_code_to_wandb()
    else:
        wandb = None
        config.update(base_config)


def setup_from_args(args):

    # launch_ingredients is a place to store variables that are derived from
    # the config but not part of it, and that are the result of our setup
    # process
    li = argparse.Namespace()  # aka launch_ingredients

    if args.seed is None:
        # Make sure the seed can be represented by floating point exactly.
        # This is just because we may want to pass it over the web, and javascript
        # doesn't have 64 bit integers.
        args.seed = np.random.randint(2**53)

    assert args.wandb or args.data_dir or args.run_type == 'inspect', (
        "Either a data directory must be set or the wandb flag must be set. "
        "If wandb is set but there is no data directory, then a run name will be "
        "picked automatically.")

    if args.ensure_gpu:
        assert torch.cuda.is_available(), "CUDA support requested but not available!"

    li.safety_dir = os.path.realpath(os.path.dirname(__file__))
    sys.path.insert(1, li.safety_dir)  # ensure current directory is on the path
    build_and_import_modules(li.safety_dir)

    # Check to see if the data directory is already in use

    if args.data_dir is not None:
        li.data_dir = os.path.realpath(args.data_dir)
        li.job_name = os.path.basename(li.data_dir)
        if os.path.exists(li.data_dir) and args.run_type == 'train':
            print("The directory '%s' already exists." % li.data_dir)
            print("Would you like to overwrite the old data, append to it, or abort?")
            response = 'overwrite' if li.job_name.startswith('tmp') else None
            while response not in ('overwrite', 'append', 'abort'):
                response = input("(overwrite / append / abort) > ")
            if response == 'overwrite':
                print("Overwriting old data.")
                shutil.rmtree(li.data_dir)
            elif response == 'abort':
                print("Aborting.")
                exit()
    else:
        li.data_dir = li.job_name = None

    setup_config_and_wandb(args, li)

    # tag any hyperparams from the commandline
    if args.extra_params is not None:
        config.add_hyperparams(args.extra_params)

    if li.data_dir is not None:
        os.makedirs(li.data_dir, exist_ok=True)
    li.logger = logging_setup.setup_logging(
        li.data_dir, debug=(config['run_type'] == 'inspect'))
    li.logger.info("COMMAND ARGUMENTS: %s", ' '.join(sys.argv))
    li.logger.info("TRAINING RUN: %s", li.job_name)
    li.logger.info("ON HOST: %s", platform.node())

    # Set the global seed

    li.main_seed = np.random.SeedSequence(config['seed'])
    li.logger.info("SETTING GLOBAL SEED: %i", li.main_seed.entropy)
    set_rng(np.random.default_rng(li.main_seed))
    torch.manual_seed(li.main_seed.entropy & (2**31 - 1))
    if config['deterministic']:
        # Note that this may slow down performance
        # See https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True

    # Run tensorboard

    if args.port and li.data_dir is not None:
        li.tb_proc = subprocess.Popen([
            "tensorboard", "--logdir_spec",
            li.job_name + ':' + li.data_dir, '--port', str(args.port)])
    else:
        li.tb_proc = None

    return li


# Start training!

def launch_training(args, launch_ingredients):

    try:
        envs = build_environments(config, launch_ingredients.main_seed, launch_ingredients.data_dir)
        obs_shape = envs['training'][0].observation_space.shape

        algo_args = {
            'training_envs': envs['training'],
            'testing_envs': envs.get('testing'),
            'data_logger': logging_setup.setup_data_logger(launch_ingredients.data_dir, 'training'),
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
            logging.error("Unexpected algorithm type '%s'", config['algo'])
            raise ValueError("unexpected algorithm type")

        algo = algo_cls(**algo_args)

        if args.wandb:
            # Before we start running things, save the config object back to wandb.
            config2 = config.copy()
            config2.pop('_wandb', None)
            wandb.config.update(config2)

        print("")
        logging.info("Hyperparameters: %s", config)
        config.check_for_unused_hyperparams()
        print("")

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

    except KeyboardInterrupt:
        logging.critical("Keyboard Interrupt. Ending early.\n")
    except Exception:
        logging.exception("Ran into an unexpected error. Aborting training.\n")
    finally:
        if config['run_type'] in ['train', 'benchmark']:
            summarize_run(launch_ingredients.data_dir, wandb and wandb.run)
        if args.wandb:
            wandb.run.finish()
        if launch_ingredients.tb_proc is not None:
            launch_ingredients.tb_proc.kill()
        if args.shutdown:
            # Shutdown in 3 minutes.
            # Enough time to recover if it crashed at the start.
            subprocess.run("sudo shutdown +3", shell=True)
            logging.critical("Shutdown commenced, but keeping ssh available...")
            subprocess.run("sudo rm -f /run/nologin", shell=True)


def main(argv=sys.argv):
    # There are three main pieces of state here:
    #
    #   args               ->  as seen by the commandline
    #   launch_ingredients ->  derived from args in preparation for training
    #                          (eg, directories that definitely exist)
    #   cofig (global)     ->  args prepared for interaction with wandb;
    #                          wandb can add settings (eg for sweeps) and
    #                          variables of local-only relevance are removed
    args = parse_args(argv)
    launch_ingredients = setup_from_args(args)
    launch_training(args, launch_ingredients)


if __name__ == "__main__":
    main()
