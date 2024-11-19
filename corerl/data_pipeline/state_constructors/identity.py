from corerl.data.data import Transition
from corerl.data_pipeline.state_constructors.base import BaseStateConstructor


class IdentityStateConstructor(BaseStateConstructor):
    def __call__(self, transitions: list[Transition]) -> list[Transition]:
        return transitions

    def reset(self) -> None:
        pass
