from typing import Literal

from lib_config.config import MISSING, config, post_processor
from lib_defs.config_defs.tag_config import TagType

from corerl.configs.tags.base import GlobalTagAttributes
from corerl.data_pipeline.transforms import NukeConfig


@config()
class MetaTagConfig(GlobalTagAttributes):
    name: str = MISSING
    type: Literal[TagType.meta] = TagType.meta


    # --------------
    # -- Defaults --
    # --------------
    @post_processor
    def _set_defaults(self, _: object):
        self.preprocess = []
        self.state_constructor = [NukeConfig()]
