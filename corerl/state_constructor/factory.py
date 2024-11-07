import corerl.state_constructor.examples as examples
from corerl.utils.hydra import DiscriminatedUnion

def init_state_constructor(cfg: DiscriminatedUnion):
    return examples.sc_group.dispatch(cfg)
