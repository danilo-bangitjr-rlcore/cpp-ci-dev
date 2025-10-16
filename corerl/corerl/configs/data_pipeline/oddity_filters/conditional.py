
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lib_config.config import config, list_, post_processor

from corerl.configs.data_pipeline.oddity_filters.base import BaseOddityFilterConfig
from corerl.configs.data_pipeline.transforms import SympyConfig, TransformConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class ConditionalFilterConfig(BaseOddityFilterConfig):
    """
    - condition: transforms that produce a boolean mask that will be used to filter out rows that evaluate to True
    - filtered_tags: tag names that should be filtered if 'condition' evaluates to True. If set to 'all',
                     all tags are filtered.
    - excluded_tags: tags that shouldn't be filtered if 'condition' evaluates to True.
    """
    name: Literal['conditional'] = 'conditional'
    condition: list[TransformConfig] = list_()
    filtered_tags: list[str] | Literal['all'] = 'all'
    excluded_tags: list[str] = list_()

    @post_processor
    def _get_default_excluded_tags(self, cfg: MainConfig):
        from corerl.utils.sympy import to_sympy

        """
        Tags involved in a sympy condition are trusted implicitly and so shouldn't have their values filtered
        """
        for transform in self.condition:
            if isinstance(transform, SympyConfig):
                _, _, tag_names = to_sympy(transform.expression)
                self.excluded_tags = list(set(self.excluded_tags + tag_names))
