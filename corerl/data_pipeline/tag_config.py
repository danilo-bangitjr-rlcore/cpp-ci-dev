from __future__ import annotations

from datetime import timedelta
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, Annotated, Callable, Literal, assert_never

import pandas as pd
from pydantic import Field

from corerl.configs.config import MISSING, config, list_, post_processor
from corerl.data_pipeline.imputers.per_tag.factory import ImputerConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.transforms import DeltaConfig, NormalizerConfig, NullConfig, TransformConfig
from corerl.messages.events import EventType
from corerl.utils.list import find_index, find_instance
from corerl.utils.maybe import Maybe
from corerl.utils.sympy import is_affine, to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig
    from corerl.data_pipeline.pipeline import PipelineConfig

BoundsElem = float | str | None

Bounds = tuple[BoundsElem, BoundsElem]
FloatBounds = tuple[float | None, float | None]

BoundsFunction = tuple[Callable[..., float] | None, Callable[..., float] | None]
BoundsTags = tuple[list[str] | None, list[str] | None]


class Agg(StrEnum):
    avg = auto()
    last = auto()
    bool_or = auto()


class TagType(StrEnum):
    ai_setpoint = auto()
    meta = auto()
    default = auto()

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

    starting_range: FloatBounds = MISSING
    duration: timedelta = MISSING

# -----------------
# -- Tag Trigger --
# -----------------
@config()
class TagTriggerConfig:
    condition: list[TransformConfig] = MISSING
    event: EventType | list[EventType] = MISSING

# ----------------
# -- Tag Config --
# ----------------
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

    The UUID associated to the OPC-UA server connection, used within the CoreIO thin client.
    """

    connection_id: str | None = None
    """
    Kind: required external

    The full OPC-UA node identifier string (e.g. ns=#;i=?). This is used for all internal references to the tag,
    such as in the reward construction.
    """

    node_identifier: str | None = None
    """
    Kind: optional external

    The long-form OPC node identifier for the tag. This is used in communication with the OPC
    server. If unspecified, the tag name is used instead.
    """

    dtype: str = 'float'
    """
    Kind: optional external

    The datatype of the OPC data. Typically this will just be a float. In rare cases, this
    may be a boolean, integer, or string.
    """

    is_endogenous: bool = True
    """
    Kind: optional external

    Whether the tag can be controlled (even indirectly) by the agent. In the control theory
    field, this is sometimes called a "controlled" variable. This may be used for plant modeling
    to simplify counterfactual reasoning.
    """

    type: TagType = TagType.default
    """
    Kind: optional external

    The type of values that this tag represents -- i.e. AI-controlled setpoints, lab tests,
    process values, etc. Specifying this value allows the data pipeline to pick smarter
    defaults.
    """

    # tag zones
    operating_range: FloatBounds | None = None
    """
    Kind: optional external

    The maximal range of values that the tag can take. Often called the "engineering range".
    If specified on an AI-controlled setpoint, this determines the range of values that the agent
    can select.
    """

    action_bounds: Bounds | None = None
    """
    Kind: optional external

    The lower and upper bounds of values that the agent can write to this tag. This interval
    may be a subset of the operating range.
    """

    action_bounds_func: Annotated[BoundsFunction | None, Field(exclude=True)] = None
    action_bounds_tags: Annotated[BoundsTags | None, Field(exclude=True)] = None
    """
    Kind: computed internal

    In case that the action_bounds are specified as strings representing sympy functions,
    the action_bounds_function will hold the functions for computing the lower and/or upper ranges,
    and the action_bounds_tags will hold the lists of tags that those functions depend on.
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

    red_bounds_func: Annotated[BoundsFunction | None, Field(exclude=True)] = None
    red_bounds_tags: Annotated[BoundsTags | None, Field(exclude=True)] = None
    """
    Kind: computed internal

    In case that the red_bounds are specified as strings representing sympy functions,
    the red_bounds_function will hold the functions for computing the lower and/or upper ranges,
    and the red_bounds_tags will hold the lists of tags that those functions depend on.
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

    yellow_bounds_func: Annotated[BoundsFunction | None, Field(exclude=True)] = None
    yellow_bounds_tags: Annotated[BoundsTags | None, Field(exclude=True)] = None
    """
    Kind: computed internal

    In case that the yellow_bounds are specified as strings representing sympy functions,
    the yellow_bounds_function will hold the functions for computing the lower and/or upper ranges,
    and the yellow_bounds_tags will hold the lists of tags that those functions depend on.
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

    is_computed: bool = False
    """
    Kind: optional external

    Specifies whether this is a computed virtual tag.
    """

    value: str | None = None
    """
    Kind: optional external

    If this is a computed virtual tag, then a value string must be specified
    in order to construct the value of the tag as a function of other tags.
    """

    @post_processor
    def _initialize_bound_functions(self, cfg: MainConfig | PipelineConfig):
        # We can receive MainConfig or PipelineConfig
        if "pipeline" in cfg.__dict__.keys():  # Avoids `isinstance` check which causes circular import
            if TYPE_CHECKING:
                assert isinstance(cfg, MainConfig)
            known_tags = set(tag.name for tag in cfg.pipeline.tags)
        elif "tags" in cfg.__dict__.keys():  # Avoids `isinstance` check which causes circular import
            if TYPE_CHECKING:
                assert isinstance(cfg, PipelineConfig)
            known_tags = set(tag.name for tag in cfg.tags)
        else:
            raise ValueError("Unknown cfg type")

        if self.action_bounds is not None:
            self.action_bounds_func, self.action_bounds_tags = self._bounds_parse_sympy(
                self.action_bounds, known_tags, allow_circular=True
            )

        if self.red_bounds is not None:
            self.red_bounds_func, self.red_bounds_tags = self._bounds_parse_sympy(self.red_bounds, known_tags)

        if self.yellow_bounds is not None:
            self.yellow_bounds_func, self.yellow_bounds_tags = self._bounds_parse_sympy(self.yellow_bounds, known_tags)

    def _bounds_parse_sympy(
        self, input_bounds: Bounds, known_tags: set[str], allow_circular: bool = False
    ) -> tuple[BoundsFunction, BoundsTags]:
        lo_func, hi_func = None, None
        lo_tags, hi_tags = None, None

        if input_bounds is not None and isinstance(input_bounds[0], str):
            expression_lo, lo_func, lo_tags = to_sympy(input_bounds[0])
            for tag in lo_tags:
                assert tag in known_tags, f"Unknown tag name in lower bound or range expression of {self.name}"
                assert (
                    allow_circular or tag != self.name
                ), f"Circular definition in lower bound or rage expression of {self.name}"
            assert is_affine(expression_lo), f"Expression on the lower bound or range of {self.name} is not affine"

        if input_bounds is not None and isinstance(input_bounds[1], str):
            expression_hi, hi_func, hi_tags = to_sympy(input_bounds[1])
            for tag in hi_tags:
                assert tag in known_tags, f"Unknown tag name in upper bound or range expression of {self.name}"
                assert (
                    allow_circular or tag != self.name
                ), f"Circular definition in upper bound or rage expression of {self.name}"
            assert is_affine(expression_hi), f"Expression on the upper bound or range of {self.name} is not affine"

        bounds_func: BoundsFunction = (lo_func, hi_func)
        bounds_tags: BoundsTags = (lo_tags, hi_tags)

        return bounds_func, bounds_tags

    @post_processor
    def _default_for_tag_types(self, cfg: MainConfig):
        match self.type:
            case TagType.default: return
            case TagType.meta: return
            case TagType.ai_setpoint: set_ai_setpoint_defaults(self)
            case _: assert_never(self.type)


    @post_processor
    def _default_normalize_preprocessor(self, cfg: MainConfig):
        # Since we don't have a dataframe, the sympy expressions are treated as None
        lo, hi = get_tag_bounds_no_eval(self)

        # although each constructor type may _also_ have a normalizer
        # only automatically set the preprocessor normalizer bounds
        norm_cfg = find_instance(NormalizerConfig, self.preprocess)
        if norm_cfg is None:
            return

        norm_cfg.min = Maybe(norm_cfg.min).flat_otherwise(lambda: lo).unwrap()

        norm_cfg.max = Maybe(norm_cfg.max).flat_otherwise(lambda: hi).unwrap()

        if norm_cfg.min is None or norm_cfg.max is None:
            norm_cfg.from_data = True

    @post_processor
    def _optional_delta_preprocessor(self, cfg: MainConfig):
        # Make sure delta transform happens before normalization
        norm_ind = find_index(lambda x_form: isinstance(x_form, NormalizerConfig), self.preprocess)
        delta_ind = find_index(lambda x_form: isinstance(x_form, DeltaConfig), self.preprocess)
        assert (
            norm_ind is None or delta_ind is None or delta_ind < norm_ind
        ), f"{self.name} must have the delta transform before the normalization transform in the preprocess stage"

    @post_processor
    def _additional_validations(self, cfg: MainConfig):
        if self.type == TagType.ai_setpoint:
            assert (
                self.operating_range is not None
                and self.operating_range[0] is not None
                and self.operating_range[1] is not None
            ), "AI-controlled setpoints must have an operating range."

        if self.guardrail_schedule is not None:
            assert self.type == TagType.ai_setpoint, \
                "A guardrail schedule was specified, but the tag is not an AI-controlled setpoint."

            # clean error message already handled above
            assert self.operating_range is not None
            lo, hi = self.operating_range
            assert lo is not None and hi is not None

            if isinstance(lo, float) and self.guardrail_schedule.starting_range[0] is not None:
                assert (
                    self.guardrail_schedule.starting_range[0] >= lo
                ), "Guardrail starting range must be greater than or equal to the operating range."

            if isinstance(hi, float) and self.guardrail_schedule.starting_range[1] is not None:
                assert (
                    self.guardrail_schedule.starting_range[1] <= hi
                ), "Guardrail starting range must be less than or equal to the operating range."

        if self.is_computed:
            assert self.value is not None, \
                "A value string must be specified for computed virtual tags."

            known_tags = set(tag.name for tag in cfg.pipeline.tags)
            _, _, dependent_tags = to_sympy(self.value)

            for dep in dependent_tags:
                assert dep in known_tags, f"Virtual tag {self.name} depends on unknown tag {dep}."


def set_ai_setpoint_defaults(tag_cfg: TagConfig):
    tag_cfg.agg = Agg.last


def get_tag_bounds(cfg: TagConfig, row: pd.DataFrame) -> tuple[Maybe[float], Maybe[float]]:
    # each bound type is fully optional
    # prefer to use red zone, fallback to black zone then yellow
    lo = (
        Maybe[float | str](cfg.red_bounds and cfg.red_bounds[0])
        .map(partial(eval_bound, row, "lo", cfg.red_bounds_func, cfg.red_bounds_tags))
        .map(widen_bound_types)
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[0])
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[0])
        .map(partial(eval_bound, row, "lo", cfg.yellow_bounds_func, cfg.yellow_bounds_tags))
    )

    hi = (
        Maybe[float | str](cfg.red_bounds and cfg.red_bounds[1])
        .map(partial(eval_bound, row, "hi", cfg.red_bounds_func, cfg.red_bounds_tags))
        .map(widen_bound_types)
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[1])
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[1])
        .map(partial(eval_bound, row, "hi", cfg.yellow_bounds_func, cfg.yellow_bounds_tags))
    )

    return lo, hi


def get_action_bounds(cfg: TagConfig, row: pd.DataFrame) -> tuple[float, float]:
    lo = (
        # the next lowest bound is either the red zone if one exists
        Maybe(cfg.action_bounds and cfg.action_bounds[0])
        .map(partial(eval_bound, row, "lo", cfg.action_bounds_func, cfg.action_bounds_tags))
        # or the operating bound if one exists
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[0])
        # and if neither exists, we're in trouble
        .expect(f"Tag {cfg.name} is configured as an action, but no lower bound found")
    )
    hi = (
        # the next lowest bound is either the red zone if one exists
        Maybe(cfg.action_bounds and cfg.action_bounds[1])
        .map(partial(eval_bound, row, "hi", cfg.action_bounds_func, cfg.action_bounds_tags))
        # or the operating bound if one exists
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[1])
        # and if neither exists, we're in trouble
        .expect(f"Tag {cfg.name} is configured as an action, but no lower bound found")
    )
    return lo, hi


def get_tag_bounds_no_eval(cfg: TagConfig) -> tuple[Maybe[float], Maybe[float]]:
    """
    Note: If you have access to the `pf.data`, it is preferred to use `get_tag_bounds`.

    This function gets the tag bounds only if they are defined as float in the config file.
    If they are str (sympy expressions), it returns none.

    It is used to initialize other parts of the config.

    Each bound type is fully optional.
    Prefer to use red zone, fallback to black zone then yellow
    """
    lo = (
        Maybe[float | str](cfg.red_bounds and cfg.red_bounds[0])
        .is_instance(float)
        .map(widen_bound_types)
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[0])
        .is_instance(float)
        .map(widen_bound_types)
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[0])
        .is_instance(float)
    )

    hi = (
        Maybe[float | str](cfg.red_bounds and cfg.red_bounds[1])
        .is_instance(float)
        .map(widen_bound_types)
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[1])
        .is_instance(float)
        .map(widen_bound_types)
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[1])
        .is_instance(float)
    )

    return lo, hi


def eval_bound(
    data: pd.DataFrame,
    side: Literal["lo", "hi"],
    bounds_func: BoundsFunction | None,
    bounds_tags: BoundsTags | None,
    bound: BoundsElem,  # This is the last argument for cleaner mapping in Maybe with functools partial
) -> float | None:
    index = {"lo": 0, "hi": 1}[side]

    if isinstance(bound, str):
        assert bounds_func and bounds_tags  # Assertion for pyright
        res_func, res_tags = bounds_func[index], bounds_tags[index]
        assert res_func and res_tags  # Assertion for pyright

        values = [data[res_tag].item() for res_tag in res_tags]
        bound = res_func(*values)

    if bound is not None:
        bound = float(bound)

    return bound


def widen_bound_types(x: float | None) -> BoundsElem:
    return x
