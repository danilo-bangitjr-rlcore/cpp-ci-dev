from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field

import corerl.utils.dict as dict_u
from corerl.configs.config import config, list_
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.imputer_stage import BaseImputer, BaseImputerStageConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NormalizerConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.norm import Normalizer
from corerl.data_pipeline.transforms.trace import TraceConfig, TraceConstructor, TraceTemporalState
from corerl.utils.list import find_instance


@dataclass
class MaskedAETemporalState:
    trace_ts: TraceTemporalState = Field(default_factory=TraceTemporalState)
    last_trace: torch.Tensor | None = None
    num_outside_thresh: int = 0


@config()
class MaskedAEConfig(BaseImputerStageConfig):
    name: Literal['masked-ae'] = 'masked-ae'
    horizon: int = 10
    proportion_missing_tolerance: float = 0.5
    trace_values: list[float] = list_([0.5, 0.9, 0.95])

    buffer_size: int = 50_000
    batch_size: int = 256
    stepsize: float = 1e-4
    err_tolerance: float = 1e-3
    max_update_steps: int = 100
    training_missing_perc: float = 0.25


class MaskedAutoencoder(BaseImputer):
    def __init__(self, imputer_cfg: MaskedAEConfig, tag_cfgs: list[TagConfig]):
        super().__init__(imputer_cfg, tag_cfgs)
        self._imputer_cfg = imputer_cfg
        self._num_obs = len(tag_cfgs)
        self._num_traces = len(imputer_cfg.trace_values)

        norm_cfgs = _find_norm_cfgs(tag_cfgs)
        self._norms = {tag: Normalizer(cfg) for tag, cfg in norm_cfgs.items()}
        self._traces = TraceConstructor(TraceConfig(
            trace_values=imputer_cfg.trace_values,
        ))
        self._buffer = CircularBuffer(imputer_cfg.buffer_size)

        # build network to include a minor bottleneck,
        # critical to ensure it does not learn a trivial
        # identity function
        sizes = [
            int(2 * self._num_obs),
            int(self._num_obs),
            int(np.ceil(0.75 * self._num_obs)),
            int(self._num_obs),
            int(2 * self._num_obs),
        ]
        parts: list[nn.Module] = [
            nn.Linear((self._num_traces + 1) * self._num_obs, sizes[0]),
            nn.ReLU(),
        ]
        for i in range(1, len(sizes)):
            parts.append(nn.Linear(sizes[i-1], sizes[i]))
            parts.append(nn.ReLU())

        parts.append(nn.Linear(sizes[-1], self._num_obs))
        self._model = nn.Sequential(*parts)

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=imputer_cfg.stepsize,
            weight_decay=0.01,
        )


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ts = dict_u.assign_default(pf.temporal_state, StageCode.IMPUTER, MaskedAETemporalState)
        assert isinstance(ts, MaskedAETemporalState)

        df = pf.data.copy(deep=False)

        # first normalize all tags that we know how to,
        # to ensure the model weighs imputing each tag
        # equally. Unnormalized tags will get uneven weighting
        for tag, norm in self._norms.items():
            tag_data = df[[tag]].copy(deep=False)
            assert isinstance(tag_data, pd.DataFrame)
            carry = TransformCarry(df, tag_data, tag)
            carry, _ = norm(carry, None)
            df[tag] = carry.transform_data

        # try to recover traces from the temporal state
        # otherwise, start fresh
        if ts.last_trace is None:
            ts.last_trace = torch.zeros(self._num_traces * self._num_obs)

        # loop through data and impute one row at a time
        # this way we can use imputed values to compute
        # the traces
        for i in range(len(df)):
            row = df.iloc[[i]].copy(deep=False)

            # inputs to the NN are the current row
            # and the prior set of traces -- note use of
            # prior traces is still a valid summary of history
            raw_row = _row_to_tensor(row)
            inputs = torch.hstack((raw_row, ts.last_trace))

            num_nan = raw_row.isnan().sum().item()
            perc_nan = num_nan / self._num_obs

            should_impute = num_nan > 0
            can_impute = perc_nan <= self._imputer_cfg.proportion_missing_tolerance
            within_horizon = ts.num_outside_thresh <= self._imputer_cfg.horizon

            # only impute if
            #   1. We need to (i.e. there are missing values)
            #   2. We can (i.e. there aren't too many missing values)
            #   3. Or we are within our imputation horizon
            if should_impute and (can_impute or within_horizon):
                with torch.no_grad():
                    raw_row = self.impute(inputs)

            # if there is enough info to impute, there
            # is enough info to train the imputer.
            if can_impute:
                self._buffer.add(inputs)
                ts.num_outside_thresh = 0
            else:
                ts.num_outside_thresh += 1

            # update traces for use on next timestep
            carry = TransformCarry(df, row, '')
            carry, ts.trace_ts = self._traces(carry, ts.trace_ts)
            ts.last_trace = _row_to_tensor(carry.transform_data)

            # since we normalized to help training,
            # denormalize back to original tag space for rest
            # of the data pipeline
            pf.data.iloc[i] = self._denormalize(raw_row.numpy())

        self.train()
        return pf


    def impute(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass through the AE to get predicted
        values for *all* inputs. Then selectively grabs
        predictions only for the inputs that are NaN.
        """
        ae_predictions = self.forward(inputs)
        raw_row = inputs[:self._num_obs]
        ae_impute = torch.where(
            torch.isnan(raw_row),
            ae_predictions,
            raw_row,
        )

        return ae_impute


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        zero_impute = inputs.nan_to_num(nan=0)
        return self._model.forward(zero_impute)


    def train(self):
        steps = 0
        loss = torch.inf
        while loss > self._imputer_cfg.err_tolerance and steps < self._imputer_cfg.max_update_steps:
            steps += 1
            self._optimizer.zero_grad()
            batch = self._buffer.sample(self._imputer_cfg.batch_size)

            # Labels may have NaNs, ignore the
            # loss for the NaN components, then
            # set the NaNs to an arbitrary value
            labels = batch[:, :self._num_obs]
            mask = torch.isnan(labels)
            labels = labels.nan_to_num(nan=0)

            # To force the AE to learn in the presence of
            # missingness, need to fake some missingness
            # in the inputs only (i.e. learn a mapping from
            # missing value to non-missing value)
            sim_missing_mask = torch.rand_like(batch) < self._imputer_cfg.training_missing_perc
            batch = torch.where(sim_missing_mask, 0, batch)
            pred = self.forward(batch)

            # loss is a standard MSE except NaN labels
            # are masked out
            error = (~mask) * (pred - labels)
            nonzero = error.count_nonzero()

            # it shouldn't happen that we have only NaN values
            # however, there is some possibility due to the
            # sim_missing_mask above, so be a bit defensive
            if nonzero > 0:
                loss = torch.sum(error**2) / nonzero
                loss.backward()
                self._optimizer.step()


    def _denormalize(self, raw_row: np.ndarray):
        for i, (col, norm) in enumerate(self._norms.items()):
            raw_row[i] = norm.invert(raw_row[i], col)

        return raw_row


def _find_norm_cfgs(tag_cfgs: list[TagConfig]):
    out: dict[str, NormalizerConfig] = {}
    for tag in tag_cfgs:
        # look for a norm config anywhere,
        # preferring sc -> ac -> rc
        norm_cfg = find_instance(
            NormalizerConfig,
            (tag.state_constructor or []) +\
            (tag.action_constructor or []) +\
            (tag.reward_constructor)
        )

        if norm_cfg is not None:
            out[tag.name] = norm_cfg

    return out


def _row_to_tensor(row: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(
        row.iloc[0].to_numpy(np.float32)
    )

class CircularBuffer:
    def __init__(self, max_size: int):
        self._storage: dict[int, torch.Tensor] = {}
        self._i = 0
        self._max_size = max_size

    def add(self, data: torch.Tensor):
        self._storage[self._i] = data
        self._i = (self._i + 1) % self._max_size

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self._storage), size=batch_size, replace=True)
        out = [
            self._storage[idx] for idx in idxs
        ]
        return torch.vstack(out)
