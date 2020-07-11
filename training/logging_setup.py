import os
import logging
import logging.config


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


def setup_data_logger(data_dir, run_type='train'):
    # Delayed import so that running from command line returns an error
    # faster for bad inputs.
    from safelife.safelife_logger import SafeLifeLogger

    os.makedirs(data_dir, exist_ok=True)

    if run_type == "benchmark":
        data_logger = SafeLifeLogger(
            data_dir,
            summary_writer=False,
            training_log=False,
            testing_video_name="benchmark-{level_name}",
            testing_log="benchmark-data.json")
    elif run_type == "train":
        data_logger = SafeLifeLogger(data_dir)
    else:
        data_logger = SafeLifeLogger(
            data_dir,
            summary_writer=False,
            training_log=False,
            testing_log=False,
            training_video_name=False,
            testing_video_name=False)
    return data_logger
