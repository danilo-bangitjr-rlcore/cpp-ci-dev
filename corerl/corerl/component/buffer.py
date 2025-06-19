import datetime
from abc import abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from discrete_dists.distribution import Support
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from discrete_dists.utils.SumTree import SumTree
from lib_agent.buffer.storage import ReplayStorage
from lib_config.config import MISSING, computed, config
from lib_config.group import Group
from pydantic import Field

from corerl.data_pipeline.datatypes import DataMode, Transition
from corerl.state import AppState

if TYPE_CHECKING:
    from corerl.config import MainConfig



class JaxTransition(NamedTuple):
    last_action: jax.Array
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    gamma: jax.Array

    action_lo: jax.Array
    action_hi: jax.Array
    next_action_lo: jax.Array
    next_action_hi: jax.Array

    dp: jax.Array
    next_dp: jax.Array

    n_step_reward: jax.Array
    n_step_gamma: jax.Array

    state_dim: int
    action_dim: int

class SampleDistributionProtocol(Protocol):
    def size(self) -> int:
        """Return the number of elements in this buffer."""
        ...

    def probs(self, elements: np.ndarray) -> np.ndarray:
        """Calculate probabilities for the given elements."""
        ...

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample n elements using the provided random number generator."""
        ...

@config()
class BaseBufferConfig:
    name: Any = MISSING
    ensemble: int = MISSING
    ensemble_probability: float = 0.5
    seed: int = MISSING
    memory: int = 1_000_000
    n_most_recent: int = 1
    batch_size: int = 256
    id: str = ""

    @computed("seed")
    @classmethod
    def _seed(cls, cfg: "MainConfig"):
        return cfg.seed

    @computed("ensemble")
    @classmethod
    def _ensemble(cls, cfg: "MainConfig"):
        return cfg.feature_flags.ensemble

class BaseBuffer:
    def __init__(self, cfg: BaseBufferConfig, app_state: AppState):
        self._cfg = cfg

        self.seed = cfg.seed
        self.rng = np.random.default_rng(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size
        self.app_state = app_state
        assert cfg.n_most_recent <= self.batch_size
        self._n_most_recent = cfg.n_most_recent

        self._storage = ReplayStorage[JaxTransition](capacity=self.memory)

        self.id = cfg.id
        self._ens_dists: list[SampleDistributionProtocol] | None = None

        self._most_recent_online_idxs = deque(maxlen=cfg.n_most_recent)

    # ----------------------------- Public Interface ----------------------------- #

    @abstractmethod
    def feed(self, transitions: Sequence[Transition], data_mode: DataMode) -> np.ndarray:
        """
        Adds transitions to the buffer. Returns the indices of those elements in the buffer
        """

    def sample(self) -> JaxTransition:
        if not self.is_sampleable:
            raise Exception('One of the sub-distributions is empty.')

        ensemble_idxs: list[np.ndarray] = []
        assert self._ens_dists is not None
        for dist in self._ens_dists:
            idxs = dist.sample(self.rng, self.batch_size)
            idxs = self._add_n_most_recent(idxs)
            ensemble_idxs.append(idxs)

        return self._storage.get_ensemble_batch(ensemble_idxs)

    @property
    def size(self) -> list[int]:
        """
        Size of each sub-distribution.
        """
        assert self._ens_dists is not None
        return [d.size() for d in self._ens_dists]

    @property
    def is_sampleable(self) -> bool:
        """
        Checks to see whether the buffer is ready to be sampled
        """
        return min(self.size) > 0

    def get_batch(self, idxs: np.ndarray):
        """
        Given an array of indices, returns a TransitionBatch where the entries are the transitions
        at the given indices.
        """
        return self._storage.get_batch(idxs)

    # ---------------------------------- Helpers --------------------------------- #

    def _update_n_most_recent(self, idxs: np.ndarray, data_mode: DataMode) -> None:
        """
        Update the most_recent_idxs
        """
        if data_mode == DataMode.ONLINE:
            for i in idxs:
                self._most_recent_online_idxs.appendleft(int(i))

    def _feed(self, transitions: Sequence[Transition]) -> np.ndarray:
        """
        Adds data to buffer without modifying distributions.
        """
        idxs = np.empty(len(transitions), dtype=np.int64)
        for j, transition in enumerate(transitions):
            idxs[j] = self._storage.add(JaxTransition(
                last_action=transition.prior.action,
                state=transition.state,
                action=transition.action,
                reward=jnp.asarray(transition.reward),
                next_state=transition.next_state,
                gamma=jnp.asarray(transition.gamma),

                action_lo=transition.steps[0].action_lo,
                action_hi=transition.steps[0].action_hi,
                next_action_lo=transition.steps[-1].action_lo,
                next_action_hi=transition.steps[-1].action_hi,

                dp=jnp.asarray(transition.steps[0].dp),
                next_dp=jnp.asarray(transition.steps[-1].dp),

                n_step_reward=jnp.asarray(transition.n_step_reward),
                n_step_gamma=jnp.asarray(transition.n_step_gamma),
                state_dim=transition.state_dim,
                action_dim=transition.action_dim,
            ))

        return idxs

    def _get_ensemble_masks(self, batch_size: int) -> np.ndarray:
        """
        Computes whether each ensemble member should get each transition.
        """
        # generate a random mask for each ensemble member
        ensemble_masks = self.rng.random((self._cfg.ensemble, batch_size)) < self._cfg.ensemble_probability

        # for any data point not selected by any ensemble member, randomly select one member
        no_ensemble = ~ensemble_masks.any(axis=0)

        for idx in np.where(no_ensemble)[0]:
            random_member = self.rng.integers(0, self._cfg.ensemble)
            ensemble_masks[random_member, idx] = True
        return ensemble_masks

    def _add_n_most_recent(self, idxs: np.ndarray) -> np.ndarray:
        """
        Adds the n most recent online idxs to the beginning of sampled indices
        """
        for i, j in enumerate(self._most_recent_online_idxs):
            idxs[i] = j
        return idxs

    def _write_buffer_sizes(self):
        """
        Write the sizes of the sub buffers to metrics.
        """
        sizes = self.size
        for i, size in enumerate(sizes):
            self.app_state.metrics.write(self.app_state.agent_step, metric=f"buffer_{self.id}[{i}]_size", value=size)


# ---------------------------------------------------------------------------- #
#                             Mixed History Buffer                             #
# ---------------------------------------------------------------------------- #

class MaskedABDistribution:
    def __init__(self, support: int, left_prob: float, mask_prob: float):
        self._mask_prob = mask_prob

        self._online = Proportional(support)
        self._historical = Proportional(support)
        self._dist = MixtureDistribution(
            [
                SubDistribution(d=self._online, p=left_prob),
                SubDistribution(d=self._historical, p=1 - left_prob),
            ],
        )

    def size(self):
        # define the number of elements in this buffer
        # as the total number of non-zero elements in either
        # distribution --- represented by the sum of the sumtree
        # since elements are either 1 or 0
        return int(self._online.tree.total() + self._historical.tree.total())

    def probs(self, elements: np.ndarray) -> np.ndarray:
        return self._dist.probs(elements)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self._dist.sample(rng, n)

    def update(
        self,
        rng: np.random.Generator,
        elements: np.ndarray,
        mode: DataMode,
        ensemble_mask: np.ndarray,
    ):
        batch_size = len(elements)

        online_mask = np.full(batch_size, mode == DataMode.ONLINE)

        self._online.update(elements, ensemble_mask & online_mask)
        self._historical.update(elements, ensemble_mask & ~online_mask)


@config()
class MixedHistoryBufferConfig(BaseBufferConfig):
    name: Literal["mixed_history_buffer"] = "mixed_history_buffer"
    online_weight: float = 0.75


class MixedHistoryBuffer(BaseBuffer):
    def __init__(self, cfg: MixedHistoryBufferConfig, app_state: AppState):
        super().__init__(cfg, app_state)
        self._ens_dists = [
            MaskedABDistribution(
                self.memory,
                cfg.online_weight,
                cfg.ensemble_probability,
            ) for _ in range(cfg.ensemble)
        ]

    def feed(self, transitions: Sequence[Transition], data_mode: DataMode) -> np.ndarray:
        """
        Adds transitions to the buffer and updates distributions.
        """
        idxs = self._feed(transitions)

        batch_size = len(idxs)
        ensemble_masks = self._get_ensemble_masks(batch_size)

        assert self._ens_dists is not None
        for dist, mask in zip(self._ens_dists, ensemble_masks, strict=True):
            assert isinstance(dist, MaskedABDistribution)
            dist.update(self.rng, idxs, data_mode, mask)

        self._write_buffer_sizes()
        self._update_n_most_recent(idxs, data_mode)

        return idxs


# ---------------------------------------------------------------------------- #
#                              Recency Bias Buffer                             #
# ---------------------------------------------------------------------------- #

class Geometric(Proportional):
    def __init__(self, support: Support | int):
        if isinstance(support, int):
            support = (0, support)

        self._support = support
        rang = support[1] - support[0]
        self.tree = SumTree(rang)

    def discount(self, discount_factor: float):
        """
        Discount the values in the distribution by the given discount factor.
        """
        old_values = self.tree.get_values(np.arange(self._support[0], self._support[1]))
        new_values = old_values * discount_factor
        self.tree.update(np.arange(self._support[0], self._support[1]), new_values)


class MaskedUGDistribution:
    def __init__(self, support: int, left_prob: float, mask_prob: float):
        self._mask_prob = mask_prob

        self._uniform = Proportional(support)
        self._geometric = Geometric(support)
        self._dist = MixtureDistribution(
            [
                SubDistribution(d=self._uniform, p=left_prob),
                SubDistribution(d=self._geometric, p=1-left_prob),
            ],
        )

    def size(self):
        return int(self._uniform.tree.total())

    def probs(self, elements: np.ndarray) -> np.ndarray:
        return self._dist.probs(elements)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self._dist.sample(rng, n)

    def update_uniform(
        self,
        elements: np.ndarray,
        ensemble_mask: np.ndarray,
    ):
        self._uniform.update(elements, ensemble_mask)

    def update_geometric(
        self,
        elements: np.ndarray,
        initial_prob: np.ndarray,
    ):
        self._geometric.update(elements, initial_prob)

    def discount_geometric(
        self,
        discount: float,
    ):
        self._geometric.discount(discount)


@config()
class RecencyBiasBufferConfig(BaseBufferConfig):
    name: Literal["recency_bias_buffer"] = "recency_bias_buffer"
    uniform_weight: float = 0.01
    obs_period : datetime.timedelta = MISSING
    effective_episodes: float = 100.
    gamma: float = MISSING
    # if gamma = 0, use the fallback_gamma. RBB will use a discount_factor = fallback_gamma^{1/effective_episodes}
    # which means that the effective_episodes-th entry in the buffer has unformalized weight fallback_gamma
    # since discount_factor^effective_episodes = fallback_gamma
    fallback_gamma: float = 0.01

    @computed("obs_period")
    @classmethod
    def _obs_period(cls, cfg: "MainConfig"):
        return cfg.interaction.obs_period

    @computed('gamma')
    @classmethod
    def _gamma(cls, cfg: "MainConfig"):
        if cfg.agent.gamma == 0:
            return cls.fallback_gamma
        return cfg.agent.gamma


class RecencyBiasBuffer(BaseBuffer):
    def __init__(self, cfg: RecencyBiasBufferConfig, app_state: AppState):
        super().__init__(cfg, app_state)

        self._obs_period = np.timedelta64(cfg.obs_period, 'us')
        self._last_timestamp = None
        self._discount_factor = np.power(cfg.gamma, 1./cfg.effective_episodes)

        self._ens_dists = [
            MaskedUGDistribution(self.memory, cfg.uniform_weight, cfg.ensemble_probability) for _ in range(cfg.ensemble)
        ]

    def feed(self, transitions: Sequence[Transition], data_mode: DataMode) -> np.ndarray:
        """
        Adds transitions to the buffer.
        """

        idxs = self._feed(transitions)

        batch_size = len(idxs)
        if batch_size == 0:
            return idxs

        ensemble_masks = self._get_ensemble_masks(batch_size)

        # get the timestamps of the transitions
        timestamps = []
        for t in transitions:
            assert t.post.timestamp is not None
            utc_ts = t.post.timestamp.astimezone(datetime.UTC)
            naive_ts = utc_ts.replace(tzinfo=None)
            timestamps.append(naive_ts)
        timestamps = np.array(timestamps, dtype='datetime64[us]')

        curr_timestamp = max(
            np.max(timestamps),
            self._last_timestamp if self._last_timestamp is not None else np.max(timestamps),
        )

        # for each transition, get the time difference between the most recent timestamp and the current timestamp
        # this will be used to discount their initial probability
        steps_since_transition = (curr_timestamp - timestamps) / self._obs_period
        weights = np.power(self._discount_factor, steps_since_transition)

        assert self._ens_dists is not None
        for dist, mask in zip(self._ens_dists, ensemble_masks, strict=True):
            assert isinstance(dist, MaskedUGDistribution)
            # first, discount old elements according the amount of time that has passed since the last timestamp
            if self._last_timestamp is not None:
                steps_since_last_call = (curr_timestamp - self._last_timestamp) / self._obs_period
                dist.discount_geometric(self._discount_factor**steps_since_last_call)

            # add new elements
            dist.update_uniform(idxs, mask)
            # some of the transitions may have happened earlier than most_recent_ts, so discount them
            dist.update_geometric(idxs, weights*mask)

        self._last_timestamp = curr_timestamp

        self._write_buffer_sizes()
        self._update_n_most_recent(idxs, data_mode)

        return idxs

    def get_probability(self, ens_i: int, idxs: np.ndarray):
        assert 0 <= ens_i < self._cfg.ensemble
        assert self._ens_dists is not None
        return self._ens_dists[ens_i].probs(idxs)

BufferConfig = Annotated[(
    MixedHistoryBufferConfig
    | RecencyBiasBufferConfig
), Field(discriminator='name')]


buffer_group = Group[
    [AppState],
    MixedHistoryBuffer | RecencyBiasBuffer,
]()

buffer_group.dispatcher(MixedHistoryBuffer)
buffer_group.dispatcher(RecencyBiasBuffer)
