import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog

from coreio.config import CoreIOConfig


def get_log_file_name(cfg: CoreIOConfig):
    if cfg.log_file is not None:
        return cfg.log_file
    return f"outputs/coreio/coreio_{cfg.coreio_origin.split(":")[-1]}.log"

def setup_logging(level: int | str = logging.INFO, log_file: str = "outputs/coreio/coreio.log") -> logging.Logger:
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

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("asyncua").setLevel(logging.WARN)
    logging.getLogger("asyncuagds").setLevel(logging.WARN)

    return logging.getLogger(__name__)
