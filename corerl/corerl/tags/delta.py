from typing import Literal

from lib_config.config import MISSING, config
from lib_defs.config_defs.tag_config import TagType

from corerl.tags.components.bounds import SafetyZonedTag


@config()
class DeltaTagConfig(
    SafetyZonedTag,
):
    name: str = MISSING
    type: Literal[TagType.delta] = TagType.delta
