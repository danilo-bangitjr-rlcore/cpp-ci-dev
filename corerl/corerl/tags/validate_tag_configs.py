from functools import partial
from itertools import product
from typing import Any

import numpy as np
from lib_utils.maybe import Maybe

from corerl.tags.components.bounds import SafetyZonedTag, get_tag_bounds
from corerl.tags.tag_config import TagConfig


def get_tag_value_permutations(tags: list[str], tag_cfgs: list[TagConfig]) -> list[tuple[Any,...]]:
    tag_vals: list[np.ndarray] = [np.empty(1)] * len(tags)
    for ind, tag_name in enumerate(tags):
        bounds = (
            Maybe.find(lambda tag_cfg, tag_name=tag_name: tag_cfg.name == tag_name, tag_cfgs)
            .is_instance(SafetyZonedTag)
            .map(partial(get_tag_bounds, row=None))
            .expect(f'Was unable to find tag config for tag: {tag_name}')
        )
        lo = bounds[0].expect(f'Was unable to find a lower bound for tag: {tag_name}')
        hi = bounds[1].expect(f'Was unable to find an upper bound for tag: {tag_name}')
        tag_vals[ind] = np.linspace(start=lo, stop=hi, num=11, endpoint=True)

    return list(product(*tag_vals))
