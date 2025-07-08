from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal

from lib_config.config import MISSING, config, post_processor
from lib_defs.config_defs.tag_config import TagType
from pydantic import Field

from corerl.tags.components.bounds import (
    Bounds,
    BoundsFunction,
    BoundsTags,
    SafetyZonedTag,
)
from corerl.tags.components.computed import ComputedTag
from corerl.tags.components.opc import OPCTag
from corerl.tags.delta import DeltaTagConfig
from corerl.tags.meta import MetaTagConfig
from corerl.tags.seasonal import SeasonalTagConfig
from corerl.tags.setpoint import SetpointTagConfig
from corerl.utils.sympy import is_affine, to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig



# ----------------
# -- Tag Config --
# ----------------
@config()
class BasicTagConfig(
    SafetyZonedTag,
    ComputedTag,
    OPCTag,
):
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

    type: Literal[TagType.default] = TagType.default
    """
    Kind: optional external

    The type of values that this tag represents -- i.e. AI-controlled setpoints, lab tests,
    process values, etc. Specifying this value allows the data pipeline to pick smarter
    defaults.
    """

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
    def _additional_validations(self, cfg: MainConfig):
        # -----------------------------
        # -- Virtual tag validations --
        # -----------------------------
        if self.is_computed:
            assert self.value is not None, \
                "A value string must be specified for computed virtual tags."

            known_tags = {tag.name for tag in cfg.pipeline.tags}
            _, _, dependent_tags = to_sympy(self.value)

            for dep in dependent_tags:
                assert dep in known_tags, f"Virtual tag {self.name} depends on unknown tag {dep}."




TagConfig = BasicTagConfig | MetaTagConfig | SeasonalTagConfig | SetpointTagConfig | DeltaTagConfig


def get_scada_tags(cfgs: list[TagConfig]):
    return [
        tag_cfg
        for tag_cfg in cfgs
        if isinstance(tag_cfg, OPCTag)
        and not tag_cfg.is_computed
    ]

def in_taglist(name: str, taglist: Sequence[TagConfig]):
    for tc in taglist:
        if tc.name == name: return True
    return False
