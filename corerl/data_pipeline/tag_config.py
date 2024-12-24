from pydantic import Field
from corerl.configs.config import config, MISSING, list_
from corerl.data_pipeline.imputers.factory import ImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.null import NullConfig


@config()
class TagConfig:
    """This is the configuration for our OPC tags. A tag is roughly equivalent to a variable that is sent and received
    through an OPC server. It is typically a single primative datatype (e.g. float).

    From `PR#335 Discussion <https://github.com/rlcoretech/core-rl/pull/335#discussion_r1898067439>`_ a tag *may*
    simultaneously represent both an observation, a reward, and possibly also an action.
    """
    name: str = MISSING

    bounds: tuple[float | None, float | None] = (None, None)
    outlier: OddityFilterConfig = Field(default_factory=EMAFilterConfig, discriminator='name')
    imputer: ImputerConfig = Field(default_factory=IdentityImputerConfig, discriminator='name')
    reward_constructor: list[TransformConfig] = list_([NullConfig()])
    state_constructor: list[TransformConfig] | None = None
    is_action: bool = False
    is_meta: bool = False
    is_endogenous: bool = True
