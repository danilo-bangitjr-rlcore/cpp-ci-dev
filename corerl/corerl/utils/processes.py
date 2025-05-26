import asyncio
import atexit
import datetime as dt
import logging

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
        proc = await try_to_execute(cmd)

        if proc is not None and proc == 0:
            break

        attempts += 1
        sleep = min(base_backoff**attempts, max_backoff)

        log.error(f'<{cmd}> has terminated unexpectedly <{attempts}> times. Restarting in {sleep} seconds.')
        await asyncio.sleep(sleep)


async def try_to_execute(cmd: list[str]):
    proc = await asyncio.create_subprocess_exec(*cmd)
    exit_code = None
    atexit.register(proc.kill)

    try:
        exit_code = await proc.wait()

    except asyncio.CancelledError:
        proc.kill()
        exit_code = 0

    except Exception:
        log.exception('Caught an exception executing the agent application!')

    atexit.unregister(proc.kill)
    return exit_code
