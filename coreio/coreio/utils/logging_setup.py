import logging
import os
from logging.handlers import RotatingFileHandler

import colorlog


def setup_logging(level: int | str = logging.INFO, log_file: str = '') -> logging.Logger:
    # Create logger - set to DEBUG to allow all messages to flow to handlers
    # Individual handlers will then filter based on their own levels
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with color - uses passed-in level for user control
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s%(reset)s: %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation - always logs at DEBUG level for comprehensive file logging
    os.makedirs('outputs/coreio', exist_ok=True)
    file_handler = RotatingFileHandler(f'outputs/coreio/coreio_{log_file}.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("asyncua").setLevel(logging.WARN)
    logging.getLogger("asyncuagds").setLevel(logging.WARN)

    return logging.getLogger(__name__)
