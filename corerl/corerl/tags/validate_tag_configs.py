import warnings
from functools import partial
from itertools import product

import numpy as np
from lib_utils.maybe import Maybe

from corerl.config import MainConfig
from corerl.environment.reward.config import Goal, JointGoal
from corerl.tags.components.bounds import (
    BoundedTag,
    BoundInfo,
    SafetyZonedTag,
    get_static_bound,
    get_widest_static_bounds,
)
from corerl.tags.setpoint import SetpointTagConfig
from corerl.tags.tag_config import TagConfig


def get_tag_value_permutations(tags: list[str], tag_cfgs: list[TagConfig]) -> list[tuple[float,...]]:
    tag_vals: list[np.ndarray] = [np.empty(1) for _ in range(len(tags))]
    for ind, tag_name in enumerate(tags):
        bounds = (
            Maybe.find(lambda tag_cfg, tag_name=tag_name: tag_cfg.name == tag_name, tag_cfgs)
            .is_instance(SafetyZonedTag)
            .map(partial(get_widest_static_bounds))
            .expect(f'Was unable to find tag config for tag: {tag_name}')
        )
        lo = bounds[0].unwrap()
        hi = bounds[1].unwrap()

        if lo is None:
            warnings.warn(
                message=f"{tag_name} has no specified lower bound. " \
                        f"Cannot generate value permutations to evaluate sympy function and compare bounds.",
                stacklevel=2,
            )
            return []

        if hi is None:
            warnings.warn(
                message=f"{tag_name} has no specified upper bound. " \
                        f"Cannot generate value permutations to evaluate sympy function and compare bounds.",
                stacklevel=2,
            )
            return []

        tag_vals[ind] = np.linspace(start=lo, stop=hi, num=11, endpoint=True).tolist()

    return list(product(*tag_vals))

def assert_bound_ordering(
    lower: BoundInfo | None,
    upper: BoundInfo | None,
    tag_cfgs: list[TagConfig],
    can_equal: bool = False,
):
    if lower is None or upper is None:
        return

    if isinstance(lower.bound_elem, float) and isinstance(upper.bound_elem, float):
        assert (
            lower.bound_elem <= upper.bound_elem if can_equal else lower.bound_elem < upper.bound_elem
        ), f"{lower.tag}'s {lower.direction} bound of the {lower.type} ({lower.bound_elem}) " \
           f"must be less than {upper.tag}'s {upper.direction} bound of the {upper.type} ({upper.bound_elem})"
    elif isinstance(lower.bound_elem, float) and isinstance(upper.bound_elem, str):
        assert upper.bound_func is not None
        assert upper.bound_tags is not None
        upper_tag_permutations = get_tag_value_permutations(upper.bound_tags, tag_cfgs)
        if len(upper_tag_permutations) == 0:
            warnings.warn(
                message=f"Cannot check that {lower.tag}'s {lower.direction} bound of the {lower.type} is less than " \
                        f"{upper.tag}'s {upper.direction} bound of the {upper.type} because {upper.tag}'s " \
                        f"{upper.direction} bound of the {upper.type}'s sympy function has tags that are unbounded",
                stacklevel=2,
            )
        for permutation in upper_tag_permutations:
            upper_val = upper.bound_func(*permutation)
            assert (
                lower.bound_elem <= upper_val if can_equal else lower.bound_elem < upper_val
            ), f"{lower.tag}'s {lower.direction} bound of the {lower.type} ({lower.bound_elem}) " \
               f"must be less than {upper.tag}'s {upper.direction} bound of the {upper.type} ({upper_val}), " \
               f"which was achieved with the following values {permutation} for the following tags {upper.bound_tags}" \
               f" in the function {upper.bound_elem}"
    elif isinstance(lower.bound_elem, str) and isinstance(upper.bound_elem, float):
        assert lower.bound_func is not None
        assert lower.bound_tags is not None
        lower_tag_permutations = get_tag_value_permutations(lower.bound_tags, tag_cfgs)
        if len(lower_tag_permutations) == 0:
            warnings.warn(
                message=f"Cannot check that {lower.tag}'s {lower.direction} bound of the {lower.type} is less than " \
                        f"{upper.tag}'s {upper.direction} bound of the {upper.type} because {lower.tag}'s " \
                        f"{lower.direction} bound of the {lower.type}'s sympy function has tags that are unbounded",
                stacklevel=2,
            )
        for permutation in lower_tag_permutations:
            lower_val = lower.bound_func(*permutation)
            assert (
                lower_val <= upper.bound_elem if can_equal else lower_val < upper.bound_elem
            ), f"{upper.tag}'s {upper.direction} bound of the {upper.type} ({upper.bound_elem}) " \
               f"must be greater than {lower.tag}'s {lower.direction} bound of the {lower.type} ({lower_val}), " \
               f"which was achieved with the following values {permutation} for the following tags {lower.bound_tags}" \
               f" in the function {lower.bound_elem}"
    elif isinstance(lower.bound_elem, str) and isinstance(upper.bound_elem, str):
        assert lower.bound_func is not None
        assert lower.bound_tags is not None
        assert upper.bound_func is not None
        assert upper.bound_tags is not None
        all_tags = list(set(lower.bound_tags + upper.bound_tags))
        lower_tag_inds = [all_tags.index(tag_str) for tag_str in lower.bound_tags]
        upper_tag_inds = [all_tags.index(tag_str) for tag_str in upper.bound_tags]
        tag_permutations = get_tag_value_permutations(all_tags, tag_cfgs)
        if len(tag_permutations) == 0:
            warnings.warn(
                message=f"Cannot check that {lower.tag}'s {lower.direction} bound of the {lower.type} is less than " \
                        f"{upper.tag}'s {upper.direction} bound of the {upper.type} because one of the bounds's " \
                        f"sympy functions has tags that are unbounded",
                stacklevel=2,
            )
        for permutation in tag_permutations:
            lower_tag_vals = [permutation[ind] for ind in lower_tag_inds]
            lower_val = lower.bound_func(*lower_tag_vals)
            upper_tag_vals = [permutation[ind] for ind in upper_tag_inds]
            upper_val = upper.bound_func(*upper_tag_vals)
            assert (
                lower_val <= upper_val if can_equal else lower_val < upper_val
            ), f"{upper.tag}'s {upper.direction} bound of the {upper.type} ({upper_val}) must be greater than " \
               f"{lower.tag}'s {lower.direction} bound of the {lower.type} ({lower_val}). {lower.tag}'s " \
               f"{lower.direction} bound of the {lower.type} was achieved with the following values {lower_tag_vals} " \
               f"for the following tags {lower.bound_tags} in the function {lower.bound_elem}. {upper.tag}'s " \
               f"{upper.direction} bound of the {upper.type} was achieved with the following values {upper_tag_vals} " \
               f"for the following tags {upper.bound_tags} in the function {upper.bound_elem}."

def non_empty_range_checks(tag_cfg: SafetyZonedTag, tag_cfgs: list[TagConfig]):
    if tag_cfg.operating_bounds_info is not None:
        # Operating Range Lower Bound < Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

    if tag_cfg.expected_bounds_info is not None:
        # Expected Range Lower Bound < Expected Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.expected_bounds_info.lower,
            upper=tag_cfg.expected_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

    if tag_cfg.red_bounds_info is not None:
        # Red Zone Lower Bound < Red Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.lower,
            upper=tag_cfg.red_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

    if tag_cfg.yellow_bounds_info is not None:
        # Yellow Zone Lower Bound < Yellow Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.yellow_bounds_info.lower,
            upper=tag_cfg.yellow_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

    if isinstance(tag_cfg, SetpointTagConfig) and tag_cfg.action_bounds_info is not None:
        # Action Lower Bound < Action Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.action_bounds_info.lower,
            upper=tag_cfg.action_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

def operating_vs_expected_range_checks(tag_cfg: SafetyZonedTag, tag_cfgs: list[TagConfig]):
    if tag_cfg.operating_bounds_info is not None and tag_cfg.expected_bounds_info is not None:
        # Operating Range Lower Bound <= Expected Range Lower Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.expected_bounds_info.lower,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Expected Range Upper Bound <= Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.expected_bounds_info.upper,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Operating Range Lower Bound < Expected Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.expected_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )
        # Expected Range Lower Bound < Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.expected_bounds_info.lower,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

def zone_bounds_vs_operating_range_checks(tag_cfg: SafetyZonedTag, tag_cfgs: list[TagConfig]):
    if tag_cfg.red_bounds_info is not None:
        assert tag_cfg.operating_bounds_info is not None
        # Operating Range Lower Bound < Red Zone Lower Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.red_bounds_info.lower,
            tag_cfgs=tag_cfgs,
        )
        # Red Zone Upper Bound < Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.upper,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )
        # Red Zone Lower Bound <= Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.lower,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Operating Range Lower Bound <= Red Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.red_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )

    if tag_cfg.yellow_bounds_info is not None:
        assert tag_cfg.operating_bounds_info is not None
        # Operating Range Lower Bound < Yellow Zone Lower Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.yellow_bounds_info.lower,
            tag_cfgs=tag_cfgs,
        )
        # Yellow Zone Upper Bound < Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.yellow_bounds_info.upper,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )
        # Yellow Zone Lower Bound <= Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.yellow_bounds_info.lower,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Operating Range Lower Bound <= Yellow Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.yellow_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )

def red_vs_yellow_zone_checks(tag_cfg: SafetyZonedTag, tag_cfgs: list[TagConfig]):
    if tag_cfg.red_bounds_info is not None and tag_cfg.yellow_bounds_info is not None:
        # Red Zone Lower Bound <= Yellow Zone Lower Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.lower,
            upper=tag_cfg.yellow_bounds_info.lower,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Yellow Zone Upper Bound <= Red Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.yellow_bounds_info.upper,
            upper=tag_cfg.red_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Red Zone Lower Bound <= Yellow Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.lower,
            upper=tag_cfg.yellow_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )
        # Yellow Zone Lower Bound <= Red Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.yellow_bounds_info.lower,
            upper=tag_cfg.red_bounds_info.upper,
            tag_cfgs=tag_cfgs,
            can_equal=True,
        )

def assert_valid_sympy_goals(cfg: MainConfig):
    """
    Ensure Goal.thresh is within the given tag's operating range when Goal.thresh is a sympy function.
    _assert_tag_in_range() already performs this check when Goal.thresh is a float.
    """
    def _evaluate_joint_goals(joint_goal: JointGoal):
        for sub_goal in joint_goal.goals:
            if isinstance(sub_goal, Goal):
                _assert_valid_sympy_goal(sub_goal)
            else:
                _evaluate_joint_goals(sub_goal)

    def _assert_valid_sympy_goal(goal: Goal):
        if isinstance(goal.thresh, str):
            assert goal.thresh_tags
            assert goal.thresh_func
            goal_tag = (
                Maybe.find(lambda tag_cfg: tag_cfg.name == goal.tag, cfg.pipeline.tags)
                .is_instance(BoundedTag)
                .expect(f"The tag used to define the Goal threshold ({goal.tag}) must have a TagConfig")
            )
            op_lo = get_static_bound(goal_tag.operating_bounds_info, lambda b: b.lower).unwrap()
            op_hi = get_static_bound(goal_tag.operating_bounds_info, lambda b: b.upper).unwrap()
            permutations = get_tag_value_permutations(goal.thresh_tags, cfg.pipeline.tags)
            for permutation in permutations:
                p_in_op_range = True
                thresh = goal.thresh_func(*permutation)
                if isinstance(op_lo, float):
                    p_in_op_range &= thresh >= op_lo
                if isinstance(op_hi, float):
                    p_in_op_range &= thresh <= op_hi
                assert p_in_op_range, (
                    f"The Goal: {goal.tag} {goal.op} {goal.thresh} violated the operating range of "
                    f"[{op_lo}, {op_hi}] when {goal.thresh_tags} had the following values: {permutation}."
                )

    if cfg.pipeline.reward:
        for priority in cfg.pipeline.reward.priorities:
            if isinstance(priority, Goal):
                _assert_valid_sympy_goal(priority)
            elif isinstance(priority, JointGoal):
                _evaluate_joint_goals(priority)

def validate_tag_configs(cfg: MainConfig):
    tag_cfgs = cfg.pipeline.tags
    assert_valid_sympy_goals(cfg)

    def check_bounds(tag_cfg: SafetyZonedTag):
        non_empty_range_checks(tag_cfg, tag_cfgs)
        operating_vs_expected_range_checks(tag_cfg, tag_cfgs)
        zone_bounds_vs_operating_range_checks(tag_cfg, tag_cfgs)
        red_vs_yellow_zone_checks(tag_cfg, tag_cfgs)

    for tag_cfg in tag_cfgs:
        Maybe(tag_cfg).is_instance(SafetyZonedTag).tap(check_bounds)
