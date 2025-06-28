from typing import Any

from lib_config.config import MISSING, config, list_, post_processor
from lib_utils.list import find_instance
from lib_utils.maybe import Maybe

from corerl.data_pipeline.imputers.per_tag.factory import ImputerConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.transforms import NormalizerConfig, NukeConfig, TransformConfig
from corerl.messages.events import EventType

BoundsElem = float | str | None

Bounds = tuple[BoundsElem, BoundsElem]
FloatBounds = tuple[float | None, float | None]

# -----------------
# -- Tag Trigger --
# -----------------
@config()
class TagTriggerConfig:
    condition: list[TransformConfig] = MISSING
    event: EventType | list[EventType] = MISSING


@config()
class BaseTagConfig:
    # NOTE: being non-prescriptive about the type of the tag name
    # so that children can do type-narrowing.
    name: Any = MISSING

    is_endogenous: bool = True
    """
    Kind: optional external

    Whether the tag can be controlled (even indirectly) by the agent. In the control theory
    field, this is sometimes called a "controlled" variable. This may be used for plant modeling
    to simplify counterfactual reasoning.
    """

    operating_range: FloatBounds | None = None
    """
    Kind: optional external

    The maximal range of values that the tag can take. Often called the "engineering range".
    If specified on an AI-controlled setpoint, this determines the range of values that the agent
    can select.
    """

    expected_range: FloatBounds | None = None
    """
    Kind: optional external

    The range of values that the tag is expected to take. If specified, this range controls
    the min/max for normalization and reward scaling.
    """


    # ----------------------
    # -- Pipeline Configs --
    # ----------------------

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

    # per-tag constructors
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

    filter: list[TransformConfig] | None = None
    """
    Kind: optional external

    Specifies a pipeline producing a boolean mask from transformations
    on other tags. If the pipeline produces True for a given timestep,
    then the value of this tag is converted to NaN.

    Used when certain operating modes, such as maintenance mode, are
    a clear signal of data degradation.
    """

    trigger: TagTriggerConfig | None = None
    """
    Kind: optional external

    Allows triggering an event based on a transformation pipeline that
    evaluates truthy.
    """


    # -----------------
    # -- Validations --
    # -----------------
    @post_processor
    def _set_normalization_bounds(self, _: object):
        lo, hi = self.get_normalization_bounds()

        norm_cfg = find_instance(NormalizerConfig, self.preprocess)
        if norm_cfg is None:
            return

        norm_cfg.min = Maybe(norm_cfg.min).otherwise(lambda: lo).unwrap()
        norm_cfg.max = Maybe(norm_cfg.max).otherwise(lambda: hi).unwrap()

        if norm_cfg.min is None or norm_cfg.max is None:
            norm_cfg.from_data = True


    # -----------------------
    # -- Utility Functions --
    # -----------------------

    def get_normalization_bounds(self) -> FloatBounds:
        def _get_bound(idx: int):
            return (
                Maybe[float](self.expected_range and self.expected_range[idx])
                .otherwise(lambda: self.operating_range and self.operating_range[idx])
                .unwrap()
            )

        lo = _get_bound(0)
        hi = _get_bound(1)

        return lo, hi
