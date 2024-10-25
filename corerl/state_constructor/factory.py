import corerl.state_constructor.examples as examples
from corerl.utils.hydra import DiscriminatedUnion

import gymnasium


def init_state_constructor(cfg: DiscriminatedUnion, env: gymnasium.Env):
    return examples.sc_group.dispatch(cfg, env)
