from functools import partial
from itertools import product
from typing import Any

import numpy as np
from lib_utils.maybe import Maybe

from corerl.config import MainConfig
from corerl.tags.components.bounds import BoundInfo, SafetyZonedTag, get_tag_bounds
from corerl.tags.tag_config import TagConfig


def get_tag_value_permutations(tags: list[str], tag_cfgs: list[TagConfig]) -> list[tuple[Any,...]]:
    tag_vals: list[np.ndarray] = [np.empty(1) for _ in range(len(tags))]
    for ind, tag_name in enumerate(tags):
        bounds = (
            Maybe.find(lambda tag_cfg, tag_name=tag_name: tag_cfg.name == tag_name, tag_cfgs)
            .is_instance(SafetyZonedTag)
            .map(partial(get_tag_bounds, row=None))
            .expect(f'Was unable to find tag config for tag: {tag_name}')
        )
        lo = bounds[0].expect(f'Was unable to find a lower bound for tag: {tag_name}')
        hi = bounds[1].expect(f'Was unable to find an upper bound for tag: {tag_name}')
        tag_vals[ind] = np.linspace(start=lo, stop=hi, num=11, endpoint=True)

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
        # Red Zone Lower Bound < Operating Range Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.lower,
            upper=tag_cfg.operating_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )
        # Operating Range Lower Bound < Red Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.operating_bounds_info.lower,
            upper=tag_cfg.red_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

def red_vs_yellow_zone_checks(tag_cfg: SafetyZonedTag, tag_cfgs: list[TagConfig]):
    if tag_cfg.red_bounds_info is not None and tag_cfg.yellow_bounds_info is not None:
        # Red Zone Lower Bound < Yellow Zone Lower Bound
        assert_bound_ordering(
            lower=tag_cfg.red_bounds_info.lower,
            upper=tag_cfg.yellow_bounds_info.lower,
            tag_cfgs=tag_cfgs,
        )
        # Yellow Zone Upper Bound < Red Zone Upper Bound
        assert_bound_ordering(
            lower=tag_cfg.yellow_bounds_info.upper,
            upper=tag_cfg.red_bounds_info.upper,
            tag_cfgs=tag_cfgs,
        )

def validate_tag_configs(cfg: MainConfig):
    tag_cfgs = cfg.pipeline.tags
    for tag_cfg in tag_cfgs:
        if isinstance(tag_cfg, SafetyZonedTag):
            operating_vs_expected_range_checks(tag_cfg, tag_cfgs)
            zone_bounds_vs_operating_range_checks(tag_cfg, tag_cfgs)
            red_vs_yellow_zone_checks(tag_cfg, tag_cfgs)
