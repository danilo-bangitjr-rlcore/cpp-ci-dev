import asyncio
import logging
import subprocess
import datetime as dt

from corerl.utils.time import Seconds, as_seconds

log = logging.getLogger(__name__)


async def keep_alive(
    cmd: list[str],
    base_backoff: Seconds = 2,
    max_backoff: Seconds | dt.timedelta = dt.timedelta(hours=1),
):
    max_backoff = as_seconds(max_backoff)
    attempts = 0

    while True:
        proc = try_to_execute(cmd)

        if proc is not None and proc.returncode == 0:
            break

        attempts += 1
        sleep = min(base_backoff**attempts, max_backoff)

        log.error(f'<{cmd}> has terminated unexpectedly <{attempts}> times. Restarting in {sleep} seconds.')
        await asyncio.sleep(sleep)


def try_to_execute(cmd: list[str]):
    try:
        return subprocess.run(cmd)

    # Note: BaseException here is broader than Exception
    # and will include things like keyboard interrupts.
    # This makes the manager process hard-to-kill (which is the goal!)
    # however, also makes it hard to work with.
    except BaseException:
        log.exception('Caught an exception executing the agent application!')
