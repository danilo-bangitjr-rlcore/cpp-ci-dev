

import time
import jax.numpy as jnp

from src.agent.components.buffer import EnsembleReplayBuffer
from src.interaction.transition_creator import Step, Transition

STATE_DIM = 5
ACTION_DIM = 3

buffer = EnsembleReplayBuffer()

def make_step(i: int)-> Step:
    return Step(
        jnp.ones(STATE_DIM)*i,
        jnp.ones(ACTION_DIM),
        1,
        jnp.ones(STATE_DIM)*(i+1),
        False,
        0.99,
    )


def make_transition(i: int):
    return Transition(
        [make_step(i), make_step(i+1)],
        1,
        0.99
    )


for i in range(10):
    transition = make_transition(i)
    buffer.add(transition)


rts = 0
for i in range(100):
    start = time.time()
    buffer.sample()
    rts += (time.time()-start)

print(rts/100)