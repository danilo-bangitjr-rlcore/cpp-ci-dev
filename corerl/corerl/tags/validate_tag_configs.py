import warnings
from collections.abc import Callable
from functools import partial
from itertools import product

import numpy as np
from lib_utils.maybe import Maybe

from corerl.config import MainConfig
from corerl.configs.tags.components.bounds import (
    BoundedTag,
    BoundInfo,
    BoundsInfo,
    Direction,
    SafetyZonedTag,
    get_static_bound,
    get_widest_static_bounds,
)
from corerl.configs.tags.setpoint import SetpointTagConfig
from corerl.configs.tags.tag_config import BasicTagConfig, TagConfig
from corerl.environment.reward.config import Goal, JointGoal
from corerl.utils.sympy import to_sympy


def get_tag_value_permutations(tags: list[str], tag_cfgs: list[TagConfig]) -> list[tuple[float,...]]:
    """
    Bounds defined as sympy functions are dynamic and depend upon the values of other tags.
    This function takes in 'tags', a list of tags in a sympy function and extracts their widest static bounds.
    These bounds are used to create a list of possible input permutations to the sympy function.
    """
    tag_vals: list[np.ndarray] = [np.empty(1) for _ in range(len(tags))]
    for ind, tag_name in enumerate(tags):
        bounds = (
            Maybe.find(lambda tag_cfg, tag_name=tag_name: tag_cfg.name == tag_name, tag_cfgs)
            .is_instance(SafetyZonedTag)
            .map(partial(get_widest_static_bounds))
            .unwrap()
        )
        if bounds is None:
            warnings.warn(
                message=f"{tag_name} has no TagConfig or is not a SafetyZonedTag. " \
                        f"Therefore, input permutations cannot be generated for the sympy function containing this tag",
                stacklevel=2,
            )
            return []

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
                    f"The threshold in the Goal: {goal.tag} {goal.op} {goal.thresh} violated the operating range of "
                    f"[{op_lo}, {op_hi}] when {goal.thresh_tags} had the following values: {permutation}."
                )

    if cfg.pipeline.reward:
        for priority in cfg.pipeline.reward.priorities:
            if isinstance(priority, Goal):
                _assert_valid_sympy_goal(priority)
            elif isinstance(priority, JointGoal):
                _evaluate_joint_goals(priority)

def red_zone_reflex_checks(tag_cfg: SafetyZonedTag, tag_cfgs: list[TagConfig]):
    if tag_cfg.red_zone_reaction is not None:
        for reflex in tag_cfg.red_zone_reaction:
            assert reflex.bounds_info
            # Red Zone Reflex Lower Bound < Red Zone Reflex Upper Bound
            assert_bound_ordering(
                lower=reflex.bounds_info.lower,
                upper=reflex.bounds_info.upper,
                tag_cfgs=tag_cfgs,
            )

            # Get TagConfig for action that the red zone reflex is applied to
            action_cfg = (
                Maybe.find(lambda cfg, tag_name=reflex.tag: cfg.name == tag_name, tag_cfgs)
                .is_instance(SafetyZonedTag)
                .expect(f"Was unable to find tag config for tag: {reflex.tag}")
            )
            assert action_cfg.operating_bounds_info
            # Operating Range Lower Bound <= Red Zone Reflex Lower Bound
            assert_bound_ordering(
                lower=action_cfg.operating_bounds_info.lower,
                upper=reflex.bounds_info.lower,
                tag_cfgs=tag_cfgs,
                can_equal=True,
            )
            # Red Zone Reflex Upper Bound <= Operating Range Upper Bound
            assert_bound_ordering(
                lower=reflex.bounds_info.upper,
                upper=action_cfg.operating_bounds_info.upper,
                tag_cfgs=tag_cfgs,
                can_equal=True,
            )
            # Operating Range Lower Bound < Red Zone Reflex Upper Bound
            assert_bound_ordering(
                lower=action_cfg.operating_bounds_info.lower,
                upper=reflex.bounds_info.upper,
                tag_cfgs=tag_cfgs,
            )
            # Red Zone Reflex Lower Bound < Operating Range Upper Bound
            assert_bound_ordering(
                lower=reflex.bounds_info.lower,
                upper=action_cfg.operating_bounds_info.upper,
                tag_cfgs=tag_cfgs,
            )

def assert_computed_tag_in_op_range(tag_cfg: BasicTagConfig | SetpointTagConfig, tag_cfgs: list[TagConfig]):
    if tag_cfg.value is not None and tag_cfg.operating_bounds_info is not None:
        op_lo = get_static_bound(tag_cfg.operating_bounds_info, lambda b: b.lower).unwrap()
        op_hi = get_static_bound(tag_cfg.operating_bounds_info, lambda b: b.upper).unwrap()
        _, computed_func, computed_func_tags = to_sympy(tag_cfg.value)
        permutations = get_tag_value_permutations(computed_func_tags, tag_cfgs)
        has_vals_in_op_range = False
        for permutation in permutations:
            p_in_op_range = True
            computed_val = computed_func(*permutation)
            if isinstance(op_lo, float):
                p_in_op_range &= computed_val >= op_lo
            if isinstance(op_hi, float):
                p_in_op_range &= computed_val <= op_hi
            has_vals_in_op_range |= p_in_op_range
            if has_vals_in_op_range:
                return

        if not has_vals_in_op_range:
            warnings.warn(
                message=f"Could not compute values for {tag_cfg.name} within operating range of [{op_lo}, {op_hi}]",
                stacklevel=2,
            )

def assert_consistent_non_redundant_goals(cfg: MainConfig):
    """
    Iterate through the list of Priorities and update the range of values tags can occupy.
    Throw warnings if there are redundancies and errors if there are inconsistencies.
    """
    def _evaluate_joint_goals(joint_goal: JointGoal, tag_bounds: dict[str, list[float | None]]):
        for sub_goal in joint_goal.goals:
            if isinstance(sub_goal, Goal):
                _check_goal_bounds(sub_goal, tag_bounds)
            else:
                _evaluate_joint_goals(sub_goal, tag_bounds)

    def _check_goal_bounds(goal: Goal, tag_bounds: dict[str, list[float | None]]):
        ind = 0 if goal.op == "up_to" else 1
        symbol = ">" if goal.op == "up_to" else "<"
        comparison = (lambda x, y: x < y) if goal.op == "up_to" else (lambda x, y: x > y)
        if goal.tag not in tag_bounds:
            tag_bounds[goal.tag] = [None, None]

        # Only compare sympy threshold to given tag's upper/lower range if it's a float
        if isinstance(goal.thresh, str) and isinstance(tag_bounds[goal.tag][ind], float):
            assert goal.thresh_tags
            assert goal.thresh_func
            permutations = get_tag_value_permutations(goal.thresh_tags, cfg.pipeline.tags)
            for permutation in permutations:
                thresh = goal.thresh_func(*permutation)
                if comparison(thresh, tag_bounds[goal.tag][ind]):
                    warnings.warn(
                        message=f"The Goal: {goal.tag} {goal.op} {goal.thresh} violates the higher priority goal "
                        f"of {goal.tag} {symbol}= {tag_bounds[goal.tag][ind]} when {goal.thresh_tags} "
                        f"has the following values: {permutation}.",
                        stacklevel=2,
                    )
        elif isinstance(goal.thresh, float):
            if isinstance(tag_bounds[goal.tag][ind], float):
                # Redundancy check
                assert comparison(tag_bounds[goal.tag][ind], goal.thresh), (
                    f"The Goal: {goal.tag} {goal.op} {goal.thresh} violates the higher priority goal "
                    f"of {goal.tag} {symbol} {tag_bounds[goal.tag][ind]}."
                )
            tag_bounds[goal.tag][ind] = goal.thresh
            goal_lo = tag_bounds[goal.tag][0]
            goal_hi = tag_bounds[goal.tag][1]
            if isinstance(goal_lo, float) and isinstance(goal_hi, float):
                # Consistency check
                assert goal_lo <= goal_hi, (
                    f"The Goals: {goal.tag} >= {goal_lo} and {goal.tag} <= {goal_hi} "
                    f"are inconsistent."
                )

    tag_ranges = {}
    if cfg.pipeline.reward:
        for priority in cfg.pipeline.reward.priorities:
            if isinstance(priority, Goal):
                _check_goal_bounds(priority, tag_ranges)
            elif isinstance(priority, JointGoal):
                _evaluate_joint_goals(priority, tag_ranges)

    return tag_ranges


def assert_consistent_goals_and_red_zones(cfg: MainConfig, goal_ranges: dict[str, list[float | None]]):
    """
    Ensure Goals do not violate red zones
    """
    def _get_goal_range_that_violates_red_zone(
        tag_cfg: SafetyZonedTag,
        red_dir: Direction,
        red_bound: float,
        goal_ind: int,
        goal_bound: float,
    ) -> tuple[float | None, float | None]:
        if (red_dir == Direction.Lower and goal_ind == 0) or (red_dir == Direction.Upper and goal_ind == 1):
            min_val = min(goal_bound, red_bound)
            max_val = max(goal_bound, red_bound)
        else:
            assert tag_cfg.operating_range
            op_ind = (goal_ind + 1) % 2
            if op_ind == 0:
                min_val = tag_cfg.operating_range[op_ind]
                max_val = goal_bound
            else:
                min_val = goal_bound
                max_val = tag_cfg.operating_range[op_ind]

        return min_val, max_val

    def _assert_goal_bound_within_red_bounds(
        tag_cfg: SafetyZonedTag,
        lens: Callable[[BoundsInfo], BoundInfo | None],
        comparison: Callable[[float, float], bool],
        ind: int,
    ):
        assert tag_cfg.red_bounds_info
        red_bound_info = lens(tag_cfg.red_bounds_info)
        goal_bound = goal_ranges[tag_cfg.name][ind]
        if red_bound_info and isinstance(goal_bound, float):
            # Can directly compare red zone boundary and goal boundary if they are both floats
            if isinstance(red_bound_info.bound_elem, float):
                min_val, max_val = _get_goal_range_that_violates_red_zone(
                    tag_cfg=tag_cfg,
                    red_dir=red_bound_info.direction,
                    red_bound=red_bound_info.bound_elem,
                    goal_ind=ind,
                    goal_bound=goal_bound,
                )
                assert comparison(goal_bound, red_bound_info.bound_elem), (
                    f"The Goals for {tag_cfg.name} are satisfied at values between [{min_val}, {max_val}] "
                    f"but this violates the red zone boundary ({red_bound_info.bound_elem})."
                )
            elif isinstance(red_bound_info.bound_elem, str):
                assert red_bound_info.bound_tags
                assert red_bound_info.bound_func
                permutations = get_tag_value_permutations(red_bound_info.bound_tags, cfg.pipeline.tags)
                for permutation in permutations:
                    red_bound_val = red_bound_info.bound_func(*permutation)
                    min_val, max_val = _get_goal_range_that_violates_red_zone(
                        tag_cfg=tag_cfg,
                        red_dir=red_bound_info.direction,
                        red_bound=red_bound_val,
                        goal_ind=ind,
                        goal_bound=goal_bound,
                    )
                    assert comparison(goal_bound, red_bound_val), (
                        f"The Goals for {tag_cfg.name} are satisfied at values between [{min_val}, {max_val}] "
                        f"but this violates the red zone boundary ({red_bound_info.bound_elem}) "
                        f"when {red_bound_info.bound_tags} have the following values: {permutation}. "
                    )

    for tag_name in goal_ranges:
        goal_tag_cfg = (
            Maybe.find(lambda tag_cfg, tag_name=tag_name: tag_cfg.name == tag_name, cfg.pipeline.tags)
            .is_instance(SafetyZonedTag)
            .expect(f"Was unable to find tag config for tag: {tag_name}")
        )
        if goal_tag_cfg.red_bounds_info is not None:
            comparisons = [
                # Compare red_lo and goal_lo
                ((lambda bounds_info: bounds_info.lower), (lambda goal_bound, red_bound: goal_bound > red_bound), 0),
                # Compare red_hi and goal_hi
                ((lambda bounds_info: bounds_info.upper), (lambda goal_bound, red_bound: goal_bound < red_bound), 1),
                # Compare red_lo and goal_hi
                ((lambda bounds_info: bounds_info.lower), (lambda goal_bound, red_bound: goal_bound > red_bound), 1),
                # Compare red_hi and goal_lo
                ((lambda bounds_info: bounds_info.upper), (lambda goal_bound, red_bound: goal_bound < red_bound), 0),
            ]

            for lens_func, comparison_func, goal_ind in comparisons:
                _assert_goal_bound_within_red_bounds(
                    tag_cfg=goal_tag_cfg,
                    lens=lens_func,
                    comparison=comparison_func,
                    ind=goal_ind,
                )

def validate_tag_configs(cfg: MainConfig):
    tag_cfgs = cfg.pipeline.tags
    assert_valid_sympy_goals(cfg)
    goal_ranges = assert_consistent_non_redundant_goals(cfg)
    assert_consistent_goals_and_red_zones(cfg, goal_ranges)

    def check_bounds(tag_cfg: SafetyZonedTag):
        non_empty_range_checks(tag_cfg, tag_cfgs)
        operating_vs_expected_range_checks(tag_cfg, tag_cfgs)
        zone_bounds_vs_operating_range_checks(tag_cfg, tag_cfgs)
        red_vs_yellow_zone_checks(tag_cfg, tag_cfgs)
        red_zone_reflex_checks(tag_cfg, tag_cfgs)

    for tag_cfg in tag_cfgs:
        Maybe(tag_cfg).is_instance(SafetyZonedTag).tap(check_bounds)
        Maybe(tag_cfg).is_instance(SetpointTagConfig).tap(partial(assert_computed_tag_in_op_range, tag_cfgs=tag_cfgs))
        Maybe(tag_cfg).is_instance(BasicTagConfig).tap(partial(assert_computed_tag_in_op_range, tag_cfgs=tag_cfgs))
