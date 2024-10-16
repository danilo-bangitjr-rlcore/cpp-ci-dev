import hydra
import asyncio
import logging
import subprocess
import datetime as dt

from omegaconf import DictConfig
from corerl.utils.time import as_seconds


def try_to_execute(cfg: DictConfig):
    exec = str(cfg.deployment.python_executable).split(' ')
    python_entrypoint = cfg.deployment.python_entrypoint
    config_name = cfg.deployment.config_name
    options = cfg.deployment.options

    try:
        return subprocess.run(
            [*exec, python_entrypoint, '--config-name', config_name, *options],
        )

    # Note: BaseException here is broader than Exception
    # and will include things like keyboard interrupts.
    # This makes the manager process hard-to-kill (which is the goal!)
    # however, also makes it hard to work with.
    except BaseException:
        logging.exception('Caught an exception executing the agent application!')


async def async_main(cfg: DictConfig):
    attempts = 0

    while True:
        # this should be an indefinitely blocking call
        # if we move on from this, it means an exception was caught
        proc = try_to_execute(cfg)

        if proc is not None and proc.returncode == 0:
            break

        attempts += 1
        sleep = min(2**attempts, as_seconds(dt.timedelta(hours=1)))

        logging.error(f'Agent code has terminated unexpectedly <{attempts}> times. Restarting in {sleep} seconds.')

        await asyncio.sleep(sleep)


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg))


if __name__ == '__main__':
    main()
