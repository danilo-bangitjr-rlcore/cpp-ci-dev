from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import pandas as pd
from lib_config.config import MISSING, config, post_processor
from lib_defs.config_defs.tag_config import TagType
from lib_utils.maybe import Maybe

from corerl.configs.tags.components.bounds import (
    BoundInfo,
    Bounds,
    BoundsInfo,
    BoundType,
    FloatBounds,
    SafetyZonedTag,
    get_bound_with_data,
    get_maybe_bound_info,
    init_bounds_info,
)
from corerl.configs.tags.components.computed import ComputedTag
from corerl.configs.tags.components.opc import Agg, OPCTag
from corerl.utils.sympy import to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig



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


@config()
class SetpointTagConfig(
    SafetyZonedTag,
    ComputedTag,
    OPCTag,
):
    name: str = MISSING
    type: Literal[TagType.ai_setpoint] = TagType.ai_setpoint

    is_endogenous: bool = True
    agg: Agg = Agg.last

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

    action_bounds_info: BoundsInfo | None = None
    """
    Kind: computed internal

    If action_bounds is specified, action_bounds_info will store a BoundsInfo object containing information about the
    lower and upper bounds, including the functions and tags for computing the lower and/or upper ranges
    if the bounds are specified as strings representing sympy functions.
    """

    guardrail_schedule: GuardrailScheduleConfig | None = None
    """
    Kind: optional external
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
    def _initialize_cascade_tag(self, cfg: 'MainConfig'):
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

    # --------------
    # -- Defaults --
    # --------------
    @post_processor
    def _set_action_bounds(self, cfg: 'MainConfig'):
        tags = {tag.name for tag in cfg.pipeline.tags}

        if self.action_bounds is not None:
            self.action_bounds_info = init_bounds_info(self, self.action_bounds, BoundType.action_bound, tags)


    # -----------------
    # -- Validations --
    # -----------------
    @post_processor
    def _validate(self, cfg: 'MainConfig'):
        if self.operating_range is None:
            raise ValueError(f"Setpoint tag '{self.name}' must have an operating range defined.")


        if self.guardrail_schedule is not None:
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

        if self.nominal_setpoint is not None:
            # Want nominal setpoint to be specified as a raw value but it must then be converted to a normalized value
            assert self.operating_range[0] is not None
            assert self.operating_range[1] is not None
            assert (
                self.operating_range[0] <= self.nominal_setpoint <= self.operating_range[1]
            ), f"The nominal setpoint {self.nominal_setpoint} must be within the operating range:" \
               f"[{self.operating_range[0]}, {self.operating_range[1]}]."
            mi = self.operating_range[0]
            ma = self.operating_range[1]
            self.nominal_setpoint = (self.nominal_setpoint - mi) / (ma - mi)


def get_action_bounds(cfg: SetpointTagConfig, row: pd.DataFrame) -> tuple[float, float]:
    def _get_bound_info(lens: Callable[[BoundsInfo], BoundInfo | None]) -> Maybe[BoundInfo]:
        return (
            get_maybe_bound_info(cfg.action_bounds_info, lens)
            .flat_otherwise(lambda: get_maybe_bound_info(cfg.red_bounds_info, lens))
            .flat_otherwise(lambda: get_maybe_bound_info(cfg.operating_bounds_info, lens))
        )

    lo = (
        get_bound_with_data(_get_bound_info(lambda b: b.lower), row)
        .expect(f"Tag {cfg.name} is configured as an action, but no lower bound found")
    )
    hi = (
        get_bound_with_data(_get_bound_info(lambda b: b.upper), row)
        .expect(f"Tag {cfg.name} is configured as an action, but no lower bound found")
    )

    return lo, hi
