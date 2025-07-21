import logging
import math
import shutil
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from lib_utils.list import sort_by

from corerl.interaction.configs import InteractionConfig

logger = logging.getLogger(__name__)


class Checkpointable(Protocol):
    def save(self, path: Path) -> Any: ...
    def load(self, path: Path) -> Any: ...


def next_power_of_2(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def prev_power_of_2(x: int):
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def periods_since(start: datetime, end: datetime, period: timedelta):
    return math.floor((end - start) / period)


def prune_checkpoints(
    chkpoints: list[Path],
    times: list[datetime],
    cliff: datetime,
    checkpoint_freq: timedelta,
) -> list[Path]:

    to_delete = []
    for i, chk in enumerate(chkpoints):
        # keep latest and first checkpoint
        if i in (0, len(chkpoints) - 1):
            continue

        # keep all checkpoints more recent than the cliff
        if times[i] > cliff:
            continue

        periods_since_cliff_chk = periods_since(times[i], cliff, checkpoint_freq)
        periods_since_cliff_prev_chk = periods_since(times[i-1], cliff, checkpoint_freq)
        periods_since_cliff_next_chk = periods_since(times[i+1], cliff, checkpoint_freq)

        # having checkpoints at powers of two is our goal. Get the next and previous powers of two in periods
        next_power_2 = next_power_of_2(periods_since_cliff_chk)
        prev_power_2 = prev_power_of_2(periods_since_cliff_chk)

        # we will delete a checkpoint if there is an older checkpoint closer to the next power of two
        # and there is a younger checkpoint closer to the previous power of two
        if periods_since_cliff_prev_chk <= next_power_2 and periods_since_cliff_next_chk >= prev_power_2:
            to_delete.append(chk)
    return to_delete


def checkpoint(
    now: datetime,
    cfg: InteractionConfig,
    last_checkpoint: datetime,
    checkpoint_cliff: timedelta,
    checkpoint_freq: timedelta,
    elements: Sequence[Checkpointable],
):
    """
    Checkpoints and removes old checkpoints to maintain a set of checkpoints that get increasingly sparse with age.
    """
    path = cfg.checkpoint_path / f'{str(now).replace(":","_")}'
    path.mkdir(exist_ok=True, parents=True)

    for element in elements:
        element.save(path)

    last_checkpoint = now

    chkpoints = list(cfg.checkpoint_path.glob('*'))
    times = [datetime.fromisoformat(chk.name.replace('_',':')) for chk in chkpoints]
    chkpoints, times = sort_by(chkpoints, times) # sorted oldest to youngest

    # keep all checkpoints more recent than the cliff
    cliff = now - checkpoint_cliff
    to_delete = prune_checkpoints(chkpoints, times, cliff, checkpoint_freq)

    for chk in to_delete:
        shutil.rmtree(chk)

    return last_checkpoint


def restore_checkpoint(
    cfg: InteractionConfig,
    elements: Sequence[Checkpointable],
):
    if not cfg.restore_checkpoint:
        return

    chkpoints = list(cfg.checkpoint_path.glob('*'))
    if len(chkpoints) == 0:
        return

    # get latest checkpoint
    checkpoint = sorted(chkpoints)[-1]
    logger.info(f"Loading agent weights from checkpoint {checkpoint}")

    for element in elements:
        element.load(checkpoint)
