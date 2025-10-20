from typing import Any, Protocol

import jax


class CriticOutputs(Protocol):
    q: jax.Array

