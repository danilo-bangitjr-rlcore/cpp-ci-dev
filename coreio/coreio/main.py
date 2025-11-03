#!/usr/bin/env python3
# CoreIO is async first

import sys

from lib_config.loader import load_config

from coreio.service import CoreIOService
from coreio.utils.config_schemas import MainConfigAdapter
from coreio.utils.logging_setup import get_log_file_name, setup_logging


@load_config(MainConfigAdapter)
def main(cfg: MainConfigAdapter):
    log_file = get_log_file_name(cfg.coreio)
    logger = setup_logging(cfg.coreio.log_level, log_file)
    logger.info(f"Initialized logger with log file: {log_file}")

    service = CoreIOService(cfg)
    service.run_forever(max_retries=5, retry_window_hours=1, enable_retry=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("Received keyboard interrupt, shutting down\n")
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR during coreio startup: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
