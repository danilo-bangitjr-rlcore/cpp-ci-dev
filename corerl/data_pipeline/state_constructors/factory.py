from corerl.data_pipeline.state_constructors.base import BaseStateConstructor, BaseStateConstructorConfig, \
    state_constructor_group
import corerl.data_pipeline.state_constructors.identity  # noqa: F401


def init_state_constructor(cfg: BaseStateConstructorConfig) -> BaseStateConstructor:
    return state_constructor_group.dispatch(cfg)
