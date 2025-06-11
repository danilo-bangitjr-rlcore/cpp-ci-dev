from dataclasses import dataclass, field
from typing import Literal

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_agent.network.networks as nets
import lib_utils.dict as dict_u
import numpy as np
import optax
import pandas as pd
from lib_agent.critic.qrc_critic import jax_u

from corerl.configs.config import config, list_
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.imputer_stage import BaseImputer, BaseImputerStageConfig
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.trace import TraceConfig, TraceConstructor, TraceTemporalState
from corerl.state import AppState


@dataclass
class MaskedAETemporalState:
    trace_ts: TraceTemporalState = field(default_factory=TraceTemporalState)
    last_trace: jax.Array | None = None
    num_outside_thresh: int = 0


@config()
class MaskedAEConfig(BaseImputerStageConfig):
    name: Literal["masked-ae"] = "masked-ae"
    horizon: int = 10
    proportion_missing_tolerance: float = 0.5
    trace_values: list[float] = list_([0.5, 0.9, 0.95])

    buffer_size: int = 50_000
    batch_size: int = 16
    warmup: int = 100 # min buffer size to update AE
    stepsize: float = 1e-5
    err_tolerance: float = 1e-3
    max_update_steps: int = 100
    training_missing_perc: float = 0.25
    fill_val: float = 0.0


class MaskedAutoencoder(BaseImputer):
    def __init__(self, imputer_cfg: MaskedAEConfig, app_state: AppState, tag_cfgs: list[TagConfig]):
        super().__init__(imputer_cfg, app_state, tag_cfgs)
        self._imputer_cfg = imputer_cfg
        self._obs_names = [t.name for t in tag_cfgs if t.type != TagType.meta]
        self._num_obs = len(self._obs_names)
        self._num_traces = len(imputer_cfg.trace_values)
        self._fill_val = imputer_cfg.fill_val

        self._traces = TraceConstructor(
            TraceConfig(
                trace_values=imputer_cfg.trace_values,
            ),
        )
        self._buffer = CircularBuffer(imputer_cfg.buffer_size)

        # build network to include a minor bottleneck,
        # critical to ensure it does not learn a trivial
        # identity function
        torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=256, activation="relu"),
                nets.LinearConfig(size=128, activation="relu"),
                nets.LinearConfig(size=int(np.ceil(0.5 * self._num_obs)), activation="relu"),
                nets.LinearConfig(size=128, activation="relu"),
                nets.LinearConfig(size=256, activation="relu"),
                nets.LinearConfig(size=int(self._num_obs), activation="relu"),
            ],
        )
        self._rng, init_rng = jax.random.split(jax.random.PRNGKey(1))
        self._net = hk.without_apply_rng(
            hk.transform(lambda x: nets.torso_builder(torso_cfg)(x)),
        )
        in_shape = (self._num_traces + 2) * self._num_obs
        self._params = self._net.init(init_rng, jnp.ones(in_shape))
        self._optim = optax.adam(learning_rate=imputer_cfg.stepsize)
        self._opt_state = self._optim.init(self._params)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ts = dict_u.assign_default(pf.temporal_state, StageCode.IMPUTER, MaskedAETemporalState)
        assert isinstance(ts, MaskedAETemporalState)

        df = pf.data.copy(deep=False)

        # try to recover traces from the temporal state
        # otherwise, start fresh
        if ts.last_trace is None:
            ts.last_trace = jnp.zeros(self._num_traces * self._num_obs)

        # loop through data and impute one row at a time
        # this way we can use imputed values to compute
        # the traces
        for i in range(len(df)):
            row = df.iloc[i].copy(deep=False)
            obs = row[self._obs_names].astype(np.float32)

            # inputs to the NN are the current row
            # and the prior set of traces -- note use of
            # prior traces is still a valid summary of history
            obs_jax = _series_to_jax(obs)
            obs_nanmask = jnp.isnan(obs_jax)
            inputs = jnp.hstack((obs_jax, ts.last_trace, obs_nanmask))

            num_nan = obs_nanmask.sum()
            perc_nan = num_nan / self._num_obs

            should_impute = num_nan > 0
            can_impute = perc_nan <= self._imputer_cfg.proportion_missing_tolerance
            within_horizon = ts.num_outside_thresh <= self._imputer_cfg.horizon

            # only impute if
            #   1. We need to (i.e. there are missing values)
            #   2. We can (i.e. there aren't too many missing values)
            #   3. Or we are within our imputation horizon
            will_impute = should_impute and (can_impute or within_horizon)
            if will_impute:
                obs_jax = self.impute(inputs, obs_nanmask)
                pf.data.loc[pf.data.index[i], self._obs_names] = np.asarray(obs_jax)
                obs[:] = obs_jax

            # if there is enough info to impute, there
            # is enough info to train the imputer.
            if can_impute:
                self._buffer.add(inputs)
                ts.num_outside_thresh = 0
            else:
                ts.num_outside_thresh += 1

            # update traces for use on next timestep
            carry = TransformCarry(df, obs.to_frame().T, "")
            carry, ts.trace_ts = self._traces(carry, ts.trace_ts)
            ts.last_trace = _series_to_jax(carry.transform_data.iloc[0])

            # log number of nans and whether imputation occurred
            self._app_state.metrics.write(self._app_state.agent_step, metric='AE-num_nan', value=num_nan)
            self._app_state.metrics.write(self._app_state.agent_step, metric='AE-imputed', value=will_impute)

        self.train()
        return pf

    def impute(self, inputs: jax.Array, obs_nanmask: jax.Array) -> jax.Array:
        """
        Runs a forward pass through the AE to get predicted
        values for *all* inputs. Then selectively grabs
        predictions only for the inputs that are NaN.
        """
        raw_obs = inputs[: self._num_obs]
        input_obs = jnp.where(obs_nanmask, self._fill_val, raw_obs)
        inputs = inputs.at[: self._num_obs].set(input_obs)
        ae_predictions = self._forward(self._params, inputs)
        return jnp.where(obs_nanmask, ae_predictions, raw_obs)

    @jax_u.method_jit
    def _forward(self, params: chex.ArrayTree, inputs: jax.Array) -> jax.Array:
        return self._net.apply(params, inputs)

    def train(self):
        if self._buffer.size < self._imputer_cfg.warmup:
            return
        steps = 0
        loss = jnp.inf
        batch_size = self._imputer_cfg.batch_size
        while loss > self._imputer_cfg.err_tolerance and steps < self._imputer_cfg.max_update_steps:
            steps += 1
            raw_input_batch = self._buffer.sample(batch_size)
            raw_obs_batch = raw_input_batch[:, : self._num_obs]

            # Labels may have NaNs. Store mask to
            # ignore loss for the NaN components and
            # to fill the missing values
            label_nanmask = jnp.isnan(raw_obs_batch)
            # if all labels are nan, skip this batch
            if (~label_nanmask).sum() == 0:
                continue
            # replace nans in the labels
            label_batch = raw_obs_batch.at[label_nanmask].set(self._fill_val)

            # To force the AE to learn in the presence of
            # missingness, need to fake some missingness
            # in the inputs only (i.e. learn a mapping from
            # missing value to non-missing value)
            self._rng, sample_rng = jax.random.split(self._rng)
            uniform_sample = jax.random.uniform(key=sample_rng, shape=(batch_size, self._num_obs))
            sim_missing_obs_mask = uniform_sample < self._imputer_cfg.training_missing_perc
            obs_batch = jnp.where(sim_missing_obs_mask, jnp.nan, raw_obs_batch)

            # update input to include simulated nans
            input_batch = raw_input_batch.at[:, : self._num_obs].set(obs_batch)
            # update missing indicator
            input_batch = input_batch.at[:, -self._num_obs :].set(sim_missing_obs_mask | label_nanmask)

            # replace nans in input - these could be simulated or real
            input_nanmask = jnp.isnan(input_batch)
            input_batch = input_batch.at[input_nanmask].set(self._fill_val)

            self._params, self._opt_state, loss = self._update(
                params=self._params,
                opt_state=self._opt_state,
                input_batch=input_batch,
                label_batch=label_batch,
                label_nanmask=label_nanmask,
            )
            self._app_state.metrics.write(self._app_state.agent_step, metric="AE-loss", value=loss)

    @jax_u.method_jit
    def _update(
        self,
        params: chex.ArrayTree,
        opt_state: chex.ArrayTree,
        input_batch: jax.Array,
        label_batch: jax.Array,
        label_nanmask: jax.Array,
    ):
        loss, grads = jax.value_and_grad(self._batch_loss)(params, input_batch, label_batch, label_nanmask)
        updates, new_opt_state = self._optim.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def _batch_loss(
        self,
        params: chex.ArrayTree,
        input_batch: jax.Array,
        label_batch: jax.Array,
        label_nanmask: jax.Array,
    ):
        losses = jax_u.vmap_except(self._loss, exclude=["params"])(params, input_batch, label_batch, label_nanmask)
        return losses.sum() / len(losses)

    def _loss(self, params: chex.ArrayTree, input: jax.Array, label: jax.Array, label_nanmask: jax.Array):
        pred = self._net.apply(params, input)
        # loss is a standard MSE except NaN labels
        # are masked out
        error = (~label_nanmask) * (pred - label)
        return jnp.sum(error**2)


def _series_to_jax(row: pd.Series) -> jax.Array:
    return jnp.asarray(row.to_numpy(np.float32))


class CircularBuffer:
    def __init__(self, max_size: int):
        self._storage: dict[int, jax.Array] = {}
        self._i = 0
        self.size = 0
        self._max_size = max_size

    def add(self, data: jax.Array):
        self._storage[self._i] = data
        self._i = (self._i + 1) % self._max_size
        self.size = min(self.size + 1, self._max_size)

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self._storage), size=batch_size, replace=True)
        out = [self._storage[idx] for idx in idxs]
        return jnp.vstack(out)
