from typing import NamedTuple, Protocol

import chex


class PolicyState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree | None = None
    group_opt_states: dict[str, chex.ArrayTree] | None = None


class ActorState(Protocol):
    actor: PolicyState
