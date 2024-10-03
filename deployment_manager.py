import asyncio
import logging
import subprocess

def MINS(x: float):
    return x * 60

def HOURS(x: float):
    return 60 * MINS(x)

def try_to_execute():
    try:
        subprocess.run(
            # TODO: replace this with actual deployment command
            ['python', 'main.py', '--config-name', 'saturation.yaml'],
        )

    # Note: BaseException here is broader than Exception
    # and will include things like keyboard interrupts.
    # This makes the manager process hard-to-kill (which is the goal!)
    # however, also makes it hard to work with.
    except BaseException:
        logging.exception('Caught an exception executing the agent application!')


async def main():
    attempts = 0

    while True:
        # this should be an indefinitely blocking call
        # if we move on from this, it means an exception was caught
        try_to_execute()

        attempts += 1
        sleep = min(2**attempts, HOURS(1))

        logging.error(f'Agent code has terminated unexpectedly <{attempts}> times. Restarting in {sleep} seconds.')

        await asyncio.sleep(sleep)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
