from corerl.utils.device import device
from omegaconf import DictConfig
from warnings import warn
import numpy as np
import torch
import random

from corerl.data.data import Transition, TransitionBatch
import pandas as pd
from corerl.sql_logging import sql_logging
from corerl.sql_logging.base_schema import SQLTransition
from sqlalchemy.orm import Session
from sqlalchemy import select
from omegaconf import DictConfig
import logging
from typing import List
from corerl.sql_logging.base_schema import TransitionInfo

logger = logging.getLogger(__name__)


class UniformBuffer:
    def __init__(self, cfg: DictConfig):
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size

        # Whether or not to use combined experience replay:
        #   https://arxiv.org/pdf/1712.01275
        self.combined = cfg.combined

        self.data = None
        self.pos = 0
        self.full = False

        if self.batch_size == 0:
            self.sample = self.sample_batch
        else:
            self.sample = self.sample_mini_batch

    @property
    def _last_pos(self):
        if self.pos == 0 and not self.full:
            return 0
        else:
            return self.pos - 1

    def feed(self, experience: Transition) -> None:
        if self.data is None:
            # Lazy instantiation
            data_size = _get_size(experience)
            self.data = [torch.empty((self.memory, *s), device=device.device) for s in data_size]

        for i, elem in enumerate(experience):
            self.data[i][self.pos] = _to_tensor(elem)

        self.pos += 1
        if not self.full and self.pos == self.memory:
            self.full = True
        self.pos %= self.memory

    def load(self, transitions: list[Transition]) -> None:
        assert len(transitions) > 0

        data_size = _get_size(transitions[0])
        self.data = [torch.empty((self.memory, *s)) for s in data_size]

        for transition in transitions:
            for i, elem in enumerate(transition):
                self.data[i][self.pos] = _to_tensor(elem)

            self.pos += 1
            if not self.full and self.pos == self.memory:
                self.full = True
            self.pos %= self.memory

        for i in range(len(self.data)):
            self.data[i] = self.data[i].to(device.device)

    def sample_mini_batch(self, batch_size: int = None) -> list[TransitionBatch]:
        if self.size == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = self.rng.randint(0, self.size, batch_size)

        if self.combined:
            sampled_indices[0] = self._last_pos

        sampled_data = [self.data[i][sampled_indices] for i in range(len(self.data))]

        return [self._prepare(sampled_data)]

    def sample_batch(self) -> list[TransitionBatch]:
        if self.size == 0:
            return None

        if self.full:
            sampled_data = self.data
        else:
            sampled_data = (self.data[i][: self.pos] for i in range(len(self.data)))

        return [self._prepare(sampled_data)]

    @property
    def size(self) -> list[int]:
        return [self.memory if self.full else self.pos]

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def get_all_data(self) -> list:
        return self.data

    def update_priorities(self, priority=None):
        pass

    def _prepare(self, batch: list) -> TransitionBatch:
        batch = TransitionBatch(*batch)
        return batch


class PriorityBuffer(UniformBuffer):
    def __init__(self, cfg: DictConfig):
        super(PriorityBuffer, self).__init__(cfg)
        self.priority = torch.zeros((self.memory,))
        warn("Priority buffer has not been tested yet")

    def feed(self, experience: Transition) -> None:
        super(PriorityBuffer, self).feed(experience)
        self.priority[self.pos] = 1.0

        if self.full:
            scale = self.priority.sum()
        else:
            scale = self.priority[: self.pos].sum()

        self.priority /= scale

    def sample_mini_batch(self, batch_size: int = None) -> list[TransitionBatch]:
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = self.rng.choice(
            self.size,
            batch_size,
            replace=False,
            p=(self.priority if self.full else self.priority[: self.pos]),
        )

        sampled_data = [self.data[i][sampled_indices] for i in range(len(self.data))]

        return [self._prepare(sampled_data)]

    def update_priorities(self, priority=None):
        if priority is None:
            raise NotImplementedError
        else:
            assert priority.shape == self.priority.shape
            self.priority = torch.tensor(priority)


class EnsembleUniformBuffer:
    def __init__(self, cfg: DictConfig):
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        random.seed(self.seed)
        self.batch_size = cfg.batch_size
        self.ensemble = cfg.ensemble  # Size of the ensemble
        self.data_subset = cfg.data_subset  # Percentage of all transitions added to a given buffer in the ensemble

        self.buffer_ensemble = [UniformBuffer(cfg) for _ in range(self.ensemble)]

        if self.batch_size == 0:
            self.sample = self.sample_batch
        else:
            self.sample = self.sample_mini_batch

    def feed(self, experience: Transition) -> None:
        for i in range(self.ensemble):
            if self.rng.rand() < self.data_subset:
                self.buffer_ensemble[i].feed(experience)

    def load(self, transitions: list[Transition]) -> None:
        num_transitions = len(transitions)
        assert num_transitions > 0

        subset_size = int(num_transitions * self.data_subset)

        ensemble_transitions = [random.sample(transitions, subset_size) for i in range(self.ensemble)]

        for i in range(self.ensemble):
            self.buffer_ensemble[i].load(ensemble_transitions[i])

    def sample_mini_batch(self, batch_size: int = None) -> list[TransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            ensemble_batch += self.buffer_ensemble[i].sample_mini_batch(batch_size)

        return ensemble_batch

    def sample_batch(self) -> list[TransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            ensemble_batch += self.buffer_ensemble[i].sample_batch()

        return ensemble_batch

    @property
    def size(self) -> list[int]:
        return [self.buffer_ensemble[i].size[0] for i in range(self.ensemble)]

    def reset(self) -> None:
        for i in range(self.ensemble):
            self.buffer_ensemble[i].reset()


def _to_tensor(elem):
    if (
            isinstance(elem, torch.Tensor)
            or isinstance(elem, np.ndarray)
            or isinstance(elem, list)
    ):
        return torch.tensor(elem)
    elif elem is None:
        return torch.empty((1, 0))
    else:
        return torch.tensor([elem])


def _get_size(experience: Transition) -> list[tuple]:
    size = []
    for elem in experience:
        if isinstance(elem, np.ndarray):
            size.append(elem.shape)
        elif elem is None:
            size.append((0,))
        elif isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, bool):
            size.append((1,))
        elif isinstance(elem, list):
            size.append((len(elem),))
        else:
            raise TypeError(f"unknown type {type(elem)}")

    return size


class SQLBuffer(UniformBuffer):
    def __init__(self, cfg: DictConfig):
        super(SQLBuffer, self).__init__(cfg)

        # Temp for debugging: TODO get from buffer cfg
        con_cfg = cfg.con_cfg
        self.engine = sql_logging.get_sql_engine(con_cfg, db_name=cfg["db_name"])

        self.session = Session(self.engine)
        self.transition_ids = []
        self._initial_idx = None
        self.only_new_transitions = cfg.only_new_transitions

    def register_session(self, session: Session):
        """
        Can be called if we need to override session
        """
        self.session = session
        self.engine = session.get_bind().engine
        self._initial_idx = self.get_initial_idx()

    @property
    def initial_idx(self):
        if not self.only_new_transitions:
            initial_idx = 0
        else:
            if self._initial_idx is None:
                initial_idx = self.get_initial_idx()
                self._initial_idx = initial_idx
            else:
                initial_idx = self._initial_idx
        return initial_idx

    def get_initial_idx(self):
        idx = self.session.scalar(
            select(SQLTransition.id).order_by(SQLTransition.id.desc())
        )
        if idx is None:
            idx = 0
        return idx

    def _transition_feed(self, experience: Transition, transition_infos: List[TransitionInfo] = None) -> None:
        sql_transition = SQLTransition(
            state=list(experience.state),
            action=list(experience.action),
            reward=experience.reward,
            next_state=list(experience.next_state),
        )
        if transition_infos is not None:
            sql_transition.transition_info.extend(transition_infos)
            sql_transition = self.session.merge(sql_transition)
        else:
            sql_transition = self.session.merge(sql_transition)
            self.session.add(sql_transition)

        self.session.commit()
        self.update_data()

    def _sql_transition_feed(self, experience: SQLTransition):

        # experience = self.session.merge(experience)
        self.session.add(experience)
        self.session.commit()
        self.update_data()

    def feed(self, experience: Transition, transition_infos: List[TransitionInfo] = None) -> None:
        if isinstance(experience, Transition):
            self._transition_feed(experience, transition_infos)
        elif isinstance(experience, SQLTransition):
            self._sql_transition_feed(experience)
        else:
            raise TypeError

    def _feed(self, experience: Transition) -> None:
        return super().feed(experience)

    def row_to_transition(self, row: pd.DataFrame) -> Transition:
        transition = Transition(
            obs=None,
            state=row.state.item(),
            action=row.action.item(),
            reward=row.reward.item(),
            n_step_reward=row.reward.item(),
            next_state=row.next_state.item(),
            boot_state=row.next_state.item(),  # WARNING: sql buffer only supports 1 step
            next_obs=None,
        )
        return transition

    def add_transitions(self, add_ids):
        add_df = pd.read_sql(
            select(SQLTransition).filter(SQLTransition.id.in_(add_ids)),
            con=self.engine,
        )

        for id in add_ids:
            row = add_df[add_df.id == id]
            if len(self.transition_ids) == self.memory:
                self.transition_ids[self.pos] = id
            else:
                self.transition_ids.append(id)

            self._feed(self.row_to_transition(row))

    def remove_transitions(self, remove_ids):
        keep_positions = []
        # TODO: Find a way to make this more efficient
        for pos, id in enumerate(self.transition_ids):
            if id not in remove_ids:
                keep_positions.append(pos)

        self.transition_ids = [self.transition_ids[pos] for pos in keep_positions]
        self.pos = len(keep_positions)
        for i in range(len(self.data)):
            self.data[i][: self.pos] = self.data[i][keep_positions]

    # def update_transition_ids(self):
    def update_data(self):
        """
        Queries transitions table to see if active trans_ids
        is different from self.transition_ids.

        If there is a difference, updates self.transition_ids to
        reflect new state of the table.
        """

        # get ids of active transitions in sql table
        df = pd.read_sql(
            select(SQLTransition.id).where(SQLTransition.exclude == False),
            con=self.engine,
        )

        # resolve differences
        table_ids = set(df[df["id"] > self.initial_idx]["id"])
        tid_set = set(self.transition_ids)
        if tid_set != table_ids:
            remove_ids = tid_set - table_ids  # ids that should be removed from buffer

            if len(remove_ids) > 0:
                self.remove_transitions(remove_ids)
                tid_set = set(self.transition_ids)  # recompute tid set after removal

            add_ids = table_ids - tid_set  # ids from table that arent in transition_ids
            if len(add_ids) > 0:
                self.add_transitions(add_ids)

    def load(self, transitions: list) -> None:
        for transition in transitions:
            self._feed(transition)
