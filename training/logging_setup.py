import os
import logging
import logging.config
from functools import lru_cache
import subprocess as sp

from safelife.safelife_logger import SafeLifeLogger

from .global_config import config


def setup_logging(data_dir, debug=False):
    logfile = os.path.join(data_dir, 'training.log')

    if not os.path.exists(logfile):
        open(logfile, 'w').close()  # write an empty file
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '{levelname:8s} {message}',
                'style': '{',
            },
            'dated': {
                'format': '{asctime} {levelname} ({filename}:{lineno}) {message}',
                'style': '{',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'stream': 'ext://sys.stdout',
                'formatter': 'simple',
            },
            'logfile': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'dated',
                'filename': logfile,
            }
        },
        'loggers': {
            'training': {
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': False,
                'handlers': ['console', 'logfile'],
            },
            'safelife': {
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': False,
                'handlers': ['console', 'logfile'],
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console', 'logfile'],
        }
    }
    logging.config.dictConfig(logging_config)

    return logging.getLogger('training')


@lru_cache(maxsize=128)  # reuse the same data logger if called multiple times
def setup_data_logger(data_dir, episode_type):
    os.makedirs(data_dir, exist_ok=True)

    if config.get('_wandb'):
        import wandb
        summary_writer = False
    elif config['run_type'] == 'train':
        wandb = None
        summary_writer = 'auto'
    else:
        wandb = None
        summary_writer = False

    return SafeLifeLogger(
        data_dir, episode_type,
        summary_writer=summary_writer,
        wandb=wandb)


def save_code_to_wandb():
    """
    Save all code that's version controlled in git to a wandb artifact.

    Note that this assumes that we're running from root of the git directory.
    """
    import wandb
    logger = logging.getLogger('training')

    # First, get all of the tracked files.
    result = sp.run(
        "git ls-tree --full-tree -r --name-only HEAD",
        shell=True, stdout=sp.PIPE)
    if result.returncode != 0:
        logger.error("Could not retrieve list of tracked files.")
    files = result.stdout.decode().strip().splitlines()
    safelife_files = wandb.Artifact('safelife_core', type='code')
    training_files = wandb.Artifact('safelife_training', type='code')
    for file in files:
        if file.rpartition('.')[2] in ('py', 'c', 'cpp', 'h', 'yaml'):
            if file.startswith('safelife'):  # core safelife code
                safelife_files.add_file(file, name=file)
            else:
                training_files.add_file(file, name=file)
    wandb.run.log_artifact(safelife_files)
    wandb.run.log_artifact(training_files)
