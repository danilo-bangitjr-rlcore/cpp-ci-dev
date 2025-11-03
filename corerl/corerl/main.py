#!/usr/bin/env python3

import logging
import sys
import traceback

from lib_config.loader import load_config

from corerl.config import MainConfig
from corerl.service import CoreRLService
from corerl.tags.validate_tag_configs import validate_tag_configs


@load_config(MainConfig)
def main(cfg: MainConfig):
    validate_tag_configs(cfg)

    log_fmt = "[%(asctime)s][%(levelname)s] - %(message)s"
    logging.basicConfig(
        format=log_fmt,
        encoding="utf-8",
        level=logging.INFO if not cfg.silent else logging.WARN,
    )
    logging.getLogger('asyncua').setLevel(logging.CRITICAL)

    service = CoreRLService(cfg)

    enable_retry = not cfg.is_simulation and cfg.max_steps is None
    service.run_forever(
        max_retries=5,
        retry_window_hours=1,
        enable_retry=enable_retry,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR during corerl startup: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
