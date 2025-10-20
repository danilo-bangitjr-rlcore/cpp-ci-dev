from typing import Any, cast

from pydantic.dataclasses import rebuild_dataclass

from corerl.configs.data_pipeline.transforms import SplitConfig, SympyConfig
from corerl.configs.data_pipeline.transforms import TransformConfig as TransformConfig
from corerl.data_pipeline.transforms.base import transform_group


def register_dispatchers():
    # Import all transform modules to ensure their dispatchers are registered
    from corerl.data_pipeline.transforms import add_raw, bounds, delta, identity, norm, nuke, trace
    from corerl.data_pipeline.transforms.split import SplitTransform
    from corerl.data_pipeline.transforms.sympy import SympyTransform

    # These are registered at module level, just need to ensure imports
    _ = (add_raw, bounds, delta, identity, norm, nuke, trace)

    transform_group.dispatcher(SplitTransform)
    transform_group.dispatcher(SympyTransform)

    # Because TransformConfig was only partially known when
    # pydantic first parsed these schemas, rebuild them
    # now that they are completely known.
    rebuild_dataclass(cast(Any, SplitConfig))
    rebuild_dataclass(cast(Any, SympyConfig))
