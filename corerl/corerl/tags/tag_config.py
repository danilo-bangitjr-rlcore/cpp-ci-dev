from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, Annotated, assert_never

import pandas as pd
from lib_config.config import MISSING, config, post_processor
from lib_defs.config_defs.tag_config import TagType
from lib_utils.maybe import Maybe
from pydantic import Field

from corerl.data_pipeline.transforms import NukeConfig
from corerl.tags.components.bounds import (
    Bounds,
    BoundsFunction,
    BoundsTags,
    FloatBounds,
    SafetyZonedTag,
    eval_bound,
)
from corerl.utils.sympy import is_affine, to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig
    from corerl.data_pipeline.pipeline import PipelineConfig


class Agg(StrEnum):
    avg = auto()
    last = auto()
    bool_or = auto()

class ViolationDirection(StrEnum):
    upper_violation = auto()
    lower_violation = auto()

@config()
class CascadeConfig:
    """
    Kind: optional external

    Specifies how the value of this virtual tag should be computed.
    The value will copy the value of the "ai setpoint" or "operator setpoint"
    as a function of a third "mode" tag.

    If the mode takes a val other than op_mode_val or ai_mode_val, the computed value
    will be NaN.
    """

    mode: str = MISSING
    op_sp: str = MISSING
    ai_sp: str = MISSING
    op_mode_val: float | int | bool = MISSING # value of mode indicating operator control
    ai_mode_val: float | int | bool = MISSING # value of mode indicating ai control
    mode_is_bool: bool = False

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



# ----------------
# -- Tag Config --
# ----------------
@config()
class TagConfig(SafetyZonedTag):
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

    This is used for all internal references to the tag, such as in the reward construction.
    """


    connection_id: str | None = None
    """
    Kind: required external

    The UUID associated to the OPC-UA server connection, used within the CoreIO thin client.
    """

    node_identifier: str | None = None
    """
    Kind: required for ai_setpoint, external

    The full OPC-UA node identifier string (e.g. ns=#;i=?). This is used by coreio in
    communication with the OPC server.
    """

    dtype: str = 'float'
    """
    Kind: optional external

    The datatype of the OPC data. Typically this will just be a float. In rare cases, this
    may be a boolean, integer, or string.
    """

    type: TagType = TagType.default
    """
    Kind: optional external

    The type of values that this tag represents -- i.e. AI-controlled setpoints, lab tests,
    process values, etc. Specifying this value allows the data pipeline to pick smarter
    defaults.
    """

    nominal_setpoint: float | None = None
    """
    Kind: optional external

    The default setpoint for this tag. Can only be specified for tags of type `TagType.ai_setpoint`.
    """

    # tag zones
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

    cascade: CascadeConfig | None = None
    """
    Kind: optional external

    Specifies whether this tag should take the one of two values
    (the "ai setpoint" or "operator setpoint") as a function of a third "mode" tag.
    """

    @post_processor
    def _initialize_cascade_tag(self, cfg: MainConfig):
        if self.cascade is None:
            return

        # mark tag as computed and define piecewise expression
        self.is_computed = True
        self.value = (
            "Piecewise("
                f"({{{self.cascade.op_sp}}}, Eq({{{self.cascade.mode}}}, {self.cascade.op_mode_val})),"
                f"({{{self.cascade.ai_sp}}}, Eq({{{self.cascade.mode}}}, {self.cascade.ai_mode_val}))"
            ")"
        )

    @post_processor
    def _initialize_bound_functions(self, cfg: MainConfig | PipelineConfig):
        # We can receive MainConfig or PipelineConfig
        if "pipeline" in cfg.__dict__.keys():  # Avoids `isinstance` check which causes circular import
            if TYPE_CHECKING:
                assert isinstance(cfg, MainConfig)
            known_tags = {tag.name for tag in cfg.pipeline.tags}
        elif "tags" in cfg.__dict__.keys():  # Avoids `isinstance` check which causes circular import
            if TYPE_CHECKING:
                assert isinstance(cfg, PipelineConfig)
            known_tags = {tag.name for tag in cfg.tags}
        else:
            raise ValueError("Unknown cfg type")

        if self.action_bounds is not None:
            self.action_bounds_func, self.action_bounds_tags = self._bounds_parse_sympy(
                self.action_bounds, known_tags, allow_circular=True,
            )


    def _bounds_parse_sympy(
        self, input_bounds: Bounds, known_tags: set[str], allow_circular: bool = False,
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
            case TagType.seasonal: set_seasonal_tag_defaults(self)
            case TagType.delta: return
            case _: assert_never(self.type)


    @post_processor
    def _additional_validations(self, cfg: MainConfig):
        # --------------------------
        # -- Setpoint validations --
        # --------------------------
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

        # -----------------------------
        # -- Virtual tag validations --
        # -----------------------------
        if self.is_computed:
            assert self.value is not None, \
                "A value string must be specified for computed virtual tags."

            known_tags = {tag.name for tag in cfg.pipeline.tags}
            if self.cascade is not None:
                known_tags |= {self.cascade.mode, self.cascade.op_sp, self.cascade.ai_sp}
            _, _, dependent_tags = to_sympy(self.value)

            for dep in dependent_tags:
                assert dep in known_tags, f"Virtual tag {self.name} depends on unknown tag {dep}."



def set_ai_setpoint_defaults(tag_cfg: TagConfig):
    tag_cfg.agg = Agg.last

def set_seasonal_tag_defaults(tag_cfg: TagConfig):
    tag_cfg.preprocess = []
    tag_cfg.state_constructor = [NukeConfig()]


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


def get_scada_tags(cfgs: list[TagConfig]) -> list[TagConfig]:
    return [
        tag_cfg
        for tag_cfg in cfgs
        if tag_cfg.type != TagType.seasonal
        and not tag_cfg.is_computed
    ]

def in_taglist(name: str, taglist: Sequence[TagConfig]):
    for tc in taglist:
        if tc.name == name: return True
    return False
