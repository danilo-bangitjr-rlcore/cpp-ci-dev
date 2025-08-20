import warnings
from collections.abc import Callable
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
from lib_config.config import MISSING, config, post_processor
from lib_utils.list import find_instance
from lib_utils.maybe import Maybe

from corerl.data_pipeline.transforms import NormalizerConfig
from corerl.tags.base import GlobalTagAttributes
from corerl.utils.sympy import is_affine, to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig

BoundsElem = float | str | None

Bounds = tuple[BoundsElem, BoundsElem]
FloatBounds = tuple[float | None, float | None]

BoundFunction = Callable[..., float]
BoundsFunctions = tuple[BoundFunction | None, BoundFunction | None]
BoundTags = list[str]
BoundsTags = tuple[BoundTags | None, BoundTags | None]


class BoundType(StrEnum):
    operating_range = auto()
    expected_range = auto()
    red_zone = auto()
    yellow_zone = auto()
    action_bound = auto()

class Direction(StrEnum):
    Lower = auto()
    Upper = auto()

class BoundInfo(NamedTuple):
    tag: str
    type: BoundType
    direction: Direction
    bound_elem: BoundsElem
    bound_func: BoundFunction | None
    bound_tags: BoundTags | None

class BoundsInfo(NamedTuple):
    lower: BoundInfo | None
    upper: BoundInfo | None

class ViolationDirection(StrEnum):
    upper_violation = auto()
    lower_violation = auto()


@config()
class RedZoneReflexConfig:
    """
    Kind: optional external

    Specifies the reaction to a red zone violation.
    """

    tag: str = MISSING
    """
    Kind: required external
    The tag to which the reaction applies.
    """

    kind: ViolationDirection = MISSING
    """
    Kind: required external

    The direction of the violation (upper or lower).
    """

    bounds: FloatBounds = MISSING
    """
    Kind: required external
    The bounds of the red zone. This is used to determine the
    reaction to the violation.
    """



@config()
class BoundedTag(GlobalTagAttributes):
    operating_range: FloatBounds | None = None
    """
    Kind: optional external

    The maximal range of values that the tag can take. Often called the "engineering range".
    If specified on an AI-controlled setpoint, this determines the range of values that the agent
    can select.
    """

    operating_bounds_info: BoundsInfo | None = None
    """
    Kind: computed internal

    If the operating_range is specified, operating_bounds_info will store a BoundsInfo object
    containing information about the lower and upper bounds.
    """

    expected_range: FloatBounds | None = None
    """
    Kind: optional external

    The range of values that the tag is expected to take. If specified, this range controls
    the min/max for normalization and reward scaling.
    """

    expected_bounds_info: BoundsInfo | None = None
    """
    Kind: computed internal

    If the expected_range is specified, expected_bounds_info will store a BoundsInfo object
    containing information about the lower and upper bounds.
    """

    operating_range_tol: float = 1e-10
    """
    Kind: internal

    The bound checker sets tag readings outside of the optionally defined operating_range to NaNs.
    BoundCheckerConfig enables you to customize the tolerance of the bounds on a per-tag basis.
    """

    @post_processor
    def _set_bounds_info(self, cfg: 'MainConfig'):
        tags = {tag.name for tag in cfg.pipeline.tags}

        if self.operating_range is not None:
            self.operating_bounds_info = init_bounds_info(self, self.operating_range, BoundType.operating_range, tags)

        if self.expected_range is not None:
            self.expected_bounds_info = init_bounds_info(self, self.expected_range, BoundType.expected_range, tags)

    @post_processor
    def _validate_bounds(self, cfg: 'MainConfig'):
        if self.operating_range is not None and self.expected_range is not None:
            if self.operating_range[0] is not None and self.expected_range[0] is not None:
                assert (
                    self.expected_range[0] >= self.operating_range[0]
                ), f"{self.name}'s lower bound of the expected range {self.expected_range[0]} " \
                   f"must be greater or equal to the lower bound of the operating range {self.operating_range[0]}"

            if self.operating_range[1] is not None and self.expected_range[1] is not None:
                assert (
                    self.expected_range[1] <= self.operating_range[1]
                ), f"{self.name}'s upper bound of the expected range {self.expected_range[1]} " \
                   f"must be smaller or equal to the upper bound of the operating range {self.operating_range[1]}"

    # -----------------------
    # -- Utility Functions --
    # -----------------------
    def get_normalization_bounds(self) -> FloatBounds:
        def _get_bound(lens: Callable[[BoundsInfo], Maybe[BoundInfo]]) -> float | None:
            return (
                get_maybe_float_bound_elem(self.expected_bounds_info, lens)
                .flat_otherwise(lambda: get_maybe_float_bound_elem(self.operating_bounds_info, lens))
                .unwrap()
            )

        lo = _get_bound(lambda b: Maybe(b.lower))
        hi = _get_bound(lambda b: Maybe(b.upper))

        return lo, hi

    # --------------
    # -- Defaults --
    # --------------
    @post_processor
    def _set_normalization_bounds(self, _: object):
        lo, hi = self.get_normalization_bounds()

        norm_cfg = find_instance(NormalizerConfig, self.preprocess)
        if norm_cfg is None:
            return

        norm_cfg.min = Maybe(norm_cfg.min).otherwise(lambda: lo).unwrap()
        norm_cfg.max = Maybe(norm_cfg.max).otherwise(lambda: hi).unwrap()

        norm_cfg.from_data = norm_cfg.min is None or norm_cfg.max is None



@config()
class SafetyZonedTag(BoundedTag):
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

    red_bounds_info: BoundsInfo | None = None
    """
    Kind: computed internal

    If red_bounds is specified, red_bounds_info will store a BoundsInfo object containing information about the lower
    and upper bounds, including the functions and tags for computing the lower and/or upper ranges
    if the bounds are specified as strings representing sympy functions.
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

    yellow_bounds_info: BoundsInfo | None = None
    """
    Kind: computed internal

    If yellow_bounds is specified, yellow_bounds_info will store a BoundsInfo object containing information about the
    lower and upper bounds, including the functions and tags for computing the lower and/or upper ranges
    if the bounds are specified as strings representing sympy functions.
    """

    red_zone_reaction: list[RedZoneReflexConfig] | None = None
    """
    Kind: optional external

    Specifies the reaction to a red zone violation.
    """

    # --------------
    # -- Defaults --
    # --------------
    @post_processor
    def _set_zone_bounds(self, cfg: 'MainConfig'):
        tags = {tag.name for tag in cfg.pipeline.tags}

        if self.red_bounds is not None:
            self.red_bounds_info = init_bounds_info(self, self.red_bounds, BoundType.red_zone, tags)

        if self.yellow_bounds is not None:
            self.yellow_bounds_info = init_bounds_info(self, self.yellow_bounds, BoundType.yellow_zone, tags)

    # -----------------------
    # -- Utility Functions --
    # -----------------------
    def get_normalization_bounds(self) -> FloatBounds:
        def _get_bound(lens: Callable[[BoundsInfo], Maybe[BoundInfo]]) -> float | None:
            return (
                get_maybe_float_bound_elem(self.expected_bounds_info, lens)
                .flat_otherwise(lambda: get_maybe_float_bound_elem(self.red_bounds_info, lens))
                .flat_otherwise(lambda: get_maybe_float_bound_elem(self.operating_bounds_info, lens))
                .flat_otherwise(lambda: get_maybe_float_bound_elem(self.yellow_bounds_info, lens))
                .unwrap()
            )

        lo = _get_bound(lambda b: Maybe(b.lower))
        hi = _get_bound(lambda b: Maybe(b.upper))

        return lo, hi

    @post_processor
    def _update_normalization_bounds(self, _: object):
        lo, hi = self.get_normalization_bounds()

        norm_cfg = find_instance(NormalizerConfig, self.preprocess)
        if norm_cfg is None:
            return

        norm_cfg.min = Maybe(norm_cfg.min).otherwise(lambda: lo).unwrap()
        norm_cfg.max = Maybe(norm_cfg.max).otherwise(lambda: hi).unwrap()

        norm_cfg.from_data = norm_cfg.min is None or norm_cfg.max is None


def parse_string_bounds(
    cfg: GlobalTagAttributes,
    input_bounds: Bounds,
    known_tags: set[str],
    allow_circular: bool = False,
) -> tuple[BoundsFunctions, BoundsTags]:
    def get_expr(bound: str):
        expression, func, tags = to_sympy(bound)

        for tag in tags:
            assert tag in known_tags, f"Unknown tag name in bound or range expression of {cfg.name}"
            assert (
                allow_circular or tag != cfg.name
            ), f"Circular definition in bound or range expression of {cfg.name}"
        assert is_affine(expression), f"Expression on the bound or range of {cfg.name} is not affine"

        return func, tags

    lo_func, lo_tags = (
        Maybe(input_bounds[0])
            .is_instance(str)
            .map(get_expr)
            .or_else((None, None))
    )

    hi_func, hi_tags = (
        Maybe(input_bounds[1])
            .is_instance(str)
            .map(get_expr)
            .or_else((None, None))
    )

    bounds_func: BoundsFunctions = (lo_func, hi_func)
    bounds_tags: BoundsTags = (lo_tags, hi_tags)

    return bounds_func, bounds_tags


def eval_bound(
    data: pd.DataFrame,
    bound_info: BoundInfo,  # This is the last argument for cleaner mapping in Maybe with functools partial
) -> Maybe[float]:
    def _inner() -> float | None:
        res_func, res_tags = bound_info.bound_func, bound_info.bound_tags
        assert res_func and res_tags  # Assertion for pyright

        if not set(res_tags).issubset(set(data.columns)):
            warnings.warn(
                message=f"Not all of the tags in {bound_info.tag}'s {bound_info.type} {bound_info.direction} bound "
                f"sympy function ({res_tags}) are in the dataframe passed to eval_bound()",
                stacklevel=2,
            )
            return None

        values = [data[res_tag].item() for res_tag in res_tags]
        return float(res_func(*values))

    bound = bound_info.bound_elem

    return (
        Maybe(bound).is_instance(str).map(lambda _: _inner())
        .otherwise(lambda: bound).is_instance(float)
    )


def get_tag_bounds(cfg: SafetyZonedTag, row: pd.DataFrame) -> tuple[Maybe[float], Maybe[float]]:
    def _get_bound_info(lens: Callable[[BoundsInfo], Maybe[BoundInfo]]) -> Maybe[BoundInfo]:
        return (
            get_maybe_bound_info(cfg.expected_bounds_info, lens)
            .flat_otherwise(lambda: get_maybe_bound_info(cfg.red_bounds_info, lens))
            .flat_otherwise(lambda: get_maybe_bound_info(cfg.operating_bounds_info, lens))
            .flat_otherwise(lambda: get_maybe_bound_info(cfg.yellow_bounds_info, lens))
        )

    lo = get_float_bound(_get_bound_info(lambda b: Maybe(b.lower)), row)
    hi = get_float_bound(_get_bound_info(lambda b: Maybe(b.upper)), row)

    return lo, hi

def init_bounds_info(
    cfg: GlobalTagAttributes,
    bounds: Bounds,
    bound_type: BoundType,
    known_tags: set[str],
) -> BoundsInfo:
    bounds_funcs, bounds_tags = parse_string_bounds(cfg, bounds, known_tags, allow_circular=True)

    lower_bound_info = None
    upper_bound_info = None

    if bounds[0] is not None:
        lower_bound_info = BoundInfo(
            tag=cfg.name,
            type=bound_type,
            direction=Direction.Lower,
            bound_elem=bounds[0],
            bound_func=bounds_funcs[0],
            bound_tags=bounds_tags[0],
        )

    if bounds[1] is not None:
        upper_bound_info = BoundInfo(
            tag=cfg.name,
            type=bound_type,
            direction=Direction.Upper,
            bound_elem=bounds[1],
            bound_func=bounds_funcs[1],
            bound_tags=bounds_tags[1],
        )

    return BoundsInfo(
        lower=lower_bound_info,
        upper=upper_bound_info,
    )

def get_float_bound(bound_info: Maybe[BoundInfo], row: pd.DataFrame) -> Maybe[float]:
    return (
        bound_info.flat_map(partial(eval_bound, row))
        .is_instance(float)
    )

def get_maybe_bound_info(bounds_info: BoundsInfo | None, get_bound_info: Callable[[BoundsInfo], Maybe[BoundInfo]]):
    return (
        Maybe(bounds_info)
        .flat_map(get_bound_info)
    )

def get_maybe_float_bound_elem(
    bounds_info: BoundsInfo | None,
    get_bound_info: Callable[[BoundsInfo], Maybe[BoundInfo]],
):
    return (
        get_maybe_bound_info(bounds_info, get_bound_info)
        .map(lambda bound_info: bound_info.bound_elem)
        .is_instance(float)
    )
