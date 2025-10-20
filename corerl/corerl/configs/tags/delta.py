from typing import Literal

from lib_config.config import MISSING, config
from lib_defs.config_defs.tag_config import TagType

from corerl.configs.tags.components.bounds import SafetyZonedTag
from corerl.configs.tags.components.opc import OPCTag


@config()
class DeltaTagConfig(
    SafetyZonedTag,
    OPCTag,
):
    name: str = MISSING
    type: Literal[TagType.delta] = TagType.delta
    is_computed: bool = False
