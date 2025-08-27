import logging

import colorlog


def setup_logging(level: int | str = logging.INFO):
    colorlog.basicConfig(
        level=level,
        format='%(log_color)s%(levelname)s%(reset)s: %(asctime)s %(message)s',
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

    # Set specific logger levels
    logging.getLogger("asyncua").setLevel(logging.WARNING)
    logging.getLogger("asyncuagds").setLevel(logging.WARNING)

    return logging.getLogger(__name__)
