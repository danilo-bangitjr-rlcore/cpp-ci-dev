from typing import Any

from lib_config.config import MISSING, config, list_
from lib_defs.type_defs.base_events import EventType

from corerl.configs.data_pipeline.imputers.per_tag import ImputerConfig
from corerl.configs.data_pipeline.oddity_filters import OddityFilterConfig
from corerl.configs.data_pipeline.transforms import NormalizerConfig, NukeConfig, TransformConfig


# -----------------
# -- Tag Trigger --
# -----------------
@config()
class TagTriggerConfig:
    condition: list[TransformConfig] = MISSING
    event: EventType | list[EventType] = MISSING


@config()
class GlobalTagAttributes:
    name: Any = MISSING

    is_endogenous: bool = True
    """
    Kind: optional external

    Whether the tag can be controlled (even indirectly) by the agent. In the control theory
    field, this is sometimes called a "controlled" variable. This may be used for plant modeling
    to simplify counterfactual reasoning.
    """

    outlier: list[OddityFilterConfig] | None = None
    """
    Kind: internal

    A per-tag configuration for outlier detection. Particularly useful for
    lab-tests or other tags with wildly different frequencies.
    """

    imputer: ImputerConfig | None = None
    """
    Kind: internal

    A per-tag configuration for imputation. Used when tags have unusual
    temporal or spatial structure, for example lab-tests.
    """

    preprocess: list[TransformConfig] = list_([NormalizerConfig()])
    """
    Kind: internal

    Specifies a pipeline of preprocessing steps to apply to the tag.
    Is not designed to be front-end compatible and exposed to the user.
    Instead, modifications of this configuration should be made through
    computed configurations based on tag_type or other user-exposed toggles.
    """

    reward_constructor: list[TransformConfig] = list_([NukeConfig()])
    """
    Kind: internal

    Specifies a transformation pipeline to produce rewards from tags.
    Exposed primarily for testing.
    """

    state_constructor: list[TransformConfig] | None = None
    """
    Kind: internal

    Specifies a transformation pipeline to produce states from tags.
    Used to produce a sufficient history of interaction to satisfy
    the Markov property.
    """

    trigger: TagTriggerConfig | None = None
    """
    Kind: optional external

    Allows triggering an event based on a transformation pipeline that
    evaluates truthy.
    """
