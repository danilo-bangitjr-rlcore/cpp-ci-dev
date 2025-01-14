from typing import Protocol


class Interaction(Protocol):
    def step(self) -> None: ...
