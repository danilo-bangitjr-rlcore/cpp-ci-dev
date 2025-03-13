from __future__ import annotations

from datetime import timedelta
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from pydantic import Field

from corerl.configs.config import MISSING, config, list_, post_processor
from corerl.data_pipeline.imputers.per_tag.factory import ImputerConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.transforms import NormalizerConfig, NullConfig, TransformConfig
from corerl.utils.list import find_instance
from corerl.utils.maybe import Maybe

if TYPE_CHECKING:
    from corerl.config import MainConfig

Bounds = tuple[float | None, float | None]

class Agg(StrEnum):
    avg = auto()
    last = auto()
    bool_or = auto()


# -----------------------
# -- Bounds Scheduling --
# -----------------------
@config()
class GuardrailScheduleConfig:
    """
    Kind: optional external

    A schedule for slowly widening the bounds of an AI-controlled
    setpoint tag. Starts within the starting_range and slowly approaches
    the full operating range over the duration.
    """
    starting_range: Bounds = MISSING
    duration: timedelta = MISSING


@config()
class TagConfig:
    """
    Kind: required external

    Configuration for a tag, representing a single variable from the plant's OPC server.
    Tags have a few strictly required fields, however other fields are optional and
    internal only.

    Tags are used to construct RL-specific concepts, such as states, actions, and rewards.
    Tags themselves, however, are not directly states, actions, or rewards even if in some
    scenarios a 1-1 mapping exists.
    """

    # tag metadata
    name: str = MISSING
    """
    Kind: required external

    The human-readable name of the tag. This is used for all internal references to the tag,
    such as in the reward construction.
    """

    node_identifier: str | int | None = None # prefer full opc node_id path (specified in web GUI)
    """
    Kind: optional external

    The long-form OPC node identifier for the tag. This is used in communication with the OPC
    server. If unspecified, the tag name is used instead.
    """

    is_endogenous: bool = True
    """
    Kind: optional external

    Whether the tag can be controlled (even indirectly) by the agent. In the control theory
    field, this is sometimes called a "controlled" variable. This may be used for plant modeling
    to simplify counterfactual reasoning.
    """

    is_meta: bool = False
    """
    Kind: internal

    Exposed only for the purposes of tests in order to pass meta-test information through
    the pipeline. For example truncation/termination or environment synchronization step.
    """

    # tag zones
    operating_range: Bounds | None = None
    """
    Kind: optional external

    The maximal range of values that the tag can take. Often called the "engineering range".
    If specified on an AI-controlled setpoint, this determines the range of values that the agent
    can select.
    """

    red_bounds: Bounds | None = None
    """
    Kind: optional external

    The interior endpoints of the two red zones. The lower value specifies a red zone
    between:
        (operating_range[0] and red_bounds[0])

    The upper value specifies a red zone between:
        (red_bounds[1] and operating_range[1])

    See also:
    https://docs.google.com/document/d/1Inm7dMHIRvIGvM7KByrRhxHsV7uCIZSNsddPTrqUcOU/edit?tab=t.c8rez4g44ssc#heading=h.qru0qq73sjyw
    """

    yellow_bounds: Bounds | None = None
    """
    Kind: optional external

    The interior endpoints of the two yellow zones. The lower value specifies a yellow zone
    between:
        (red_bounds[0] and yellow_bounds[0])

    The upper value specifies a yellow zone between:
        (yellow_bounds[1] and red_bounds[1])

    If a corresponding red zone is not specified, the yellow zone's exterior endpoint
    is determined by the operating range.

    See also:
    https://docs.google.com/document/d/1Inm7dMHIRvIGvM7KByrRhxHsV7uCIZSNsddPTrqUcOU/edit?tab=t.c8rez4g44ssc#heading=h.qru0qq73sjyw
    """

    change_bounds: tuple[float, float] | None = None
    """
    Kind: required external
    Requires: feature_flags.delta_actions


    The maximal change between consecutive AI-controlled setpoint values, specified
    as a percentage change between [-1, 1] inclusive.
    If specified, this tag _must_ be an AI-controlled setpoint.
    """

    guardrail_schedule: GuardrailScheduleConfig | None = None
    """
    Kind: optional external
    """

    # per-tag pipeline configuration
    agg: Agg = Agg.avg
    """
    Kind: internal

    The temporal aggregation strategy used when querying timescale db. For most tags,
    this should be Agg.avg. For setpoints, this should be Agg.last.
    """

    outlier: OddityFilterConfig = Field(default_factory=IdentityFilterConfig)
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

    reward_constructor: list[TransformConfig] = list_([NullConfig()])
    """
    Kind: internal

    Specifies a transformation pipeline to produce rewards from tags.
    Exposed primarily for testing.
    """

    action_constructor: list[TransformConfig] | None = None
    """
    Kind: internal

    Specifies a transformation pipeline to produce actions from tags.
    Used to decouple AI-controlled setpoints (i.e. the outputs) from
    the effects on the plant (i.e. the inputs). In most cases, these
    will be the same.
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

    @post_processor
    def _default_normalize_preprocessor(self, cfg: MainConfig):
        lo, hi = get_tag_bounds(self)

        # although each constructor type may _also_ have a normalizer
        # only automatically set the preprocessor normalizer bounds
        norm_cfg = find_instance(NormalizerConfig, self.preprocess)
        if norm_cfg is None:
            return

        norm_cfg.min = (
            Maybe(norm_cfg.min)
            .flat_otherwise(lambda: lo)
            .unwrap()
        )

        norm_cfg.max = (
            Maybe(norm_cfg.max)
            .flat_otherwise(lambda: hi)
            .unwrap()
        )

        if norm_cfg.min is None or norm_cfg.max is None:
            norm_cfg.from_data = True


    @post_processor
    def _additional_validations(self, cfg: MainConfig):
        # it is only valid to specify a guardrail schedule if the tag is an AI-controlled setpoint
        if self.guardrail_schedule is not None:
            assert self.change_bounds is not None or self.action_constructor is not None, \
                "A guardrail schedule was specified, but the tag is not an AI-controlled setpoint."


def get_tag_bounds(cfg: TagConfig) -> tuple[Maybe[float], Maybe[float]]:
    # each bound type is fully optional
    # prefer to use red zone, fallback to black zone then yellow
    lo = (
        Maybe[float](cfg.red_bounds and cfg.red_bounds[0])
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[0])
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[0])
    )

    hi = (
        Maybe[float](cfg.red_bounds and cfg.red_bounds[1])
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[1])
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[1])
    )

    return lo, hi
