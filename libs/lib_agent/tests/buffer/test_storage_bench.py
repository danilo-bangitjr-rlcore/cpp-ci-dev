from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.buffer.storage import ReplayStorage


def test_storage_integration(benchmark: BenchmarkFixture):
    rng = np.random.default_rng(0)
    storage = ReplayStorage(1000)

    class Step(NamedTuple):
        a: jax.Array
        b: jax.Array


    fake_step = Step(
        a=jnp.array(rng.random((256,))),
        b=jnp.array(rng.random((256,))),
    )

    # prefill storage so we don't need complicated accessor logic
    for _ in range(1000):
        storage.add(fake_step)


    def _inner(storage: ReplayStorage, step: Step, idxs: list[np.ndarray]):
        for _ in range(10):
            storage.add(step)
            storage.get_ensemble_batch(idxs)

    benchmark(
        _inner,
        storage,
        fake_step,
        [np.arange(256), np.arange(256, 512)],
    )
