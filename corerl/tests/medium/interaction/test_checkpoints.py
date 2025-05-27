import math
from datetime import datetime, timedelta
from math import log2
from pathlib import Path

from corerl.interaction.deployment_interaction import prune_checkpoints


def is_power_of_two(n: int):
    if n <= 0:
        return False
    return math.log2(n).is_integer()

def test_prune_checkpoints():

    cliff_steps = 4
    now = datetime.now()
    checkpoint_freq = timedelta(hours=1)
    cliff_delta = timedelta(hours=cliff_steps)

    n = 10
    steps = 2**n

    chkpoints = []
    times = []

    for step in range(steps):
        new_chk = Path(f"{now.isoformat().replace(':', '_')}")
        chkpoints.append(new_chk)

        times = [datetime.fromisoformat(chk.name.replace('_',':')) for chk in chkpoints]
        cliff = now - cliff_delta
        to_delete = prune_checkpoints(chkpoints, times, cliff, checkpoint_freq)

        for chk in to_delete:
            chkpoints.remove(chk)
        now += checkpoint_freq

        # Ensure the number of checkpoints scales logarithmically with steps after cliff
        steps_after_cliff = max(0, step - cliff_steps)
        if steps_after_cliff > 0:
            assert len(chkpoints) <= (cliff_delta / checkpoint_freq) + 1 + 2*log2(steps_after_cliff+1)

        # Ensure all chkpoints at powers of two from the cliff are maintained
        if is_power_of_two(steps_after_cliff):
            times = [datetime.fromisoformat(chk.name.replace('_',':')) for chk in chkpoints]
            ages_after_cliff = [int((cliff - t)/checkpoint_freq) for t in times]

            assert check_powers_of_two(ages_after_cliff, steps_after_cliff)


def check_powers_of_two(numbers: list[int], max_power: int):
    """
    Check if the list contains all powers of 2 smaller than 2^max_power.
    """
    # Find the exponent n where 2^n = max_power
    n = 0
    temp = max_power
    while temp > 1:
        temp //= 2
        n += 1

    # Check for each power of 2 less than max_power
    for i in range(n):
        power = 2 ** i
        if power not in numbers:
            return False

    return True
