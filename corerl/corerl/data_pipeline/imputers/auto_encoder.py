import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_agent.network.networks as nets
import lib_utils.dict as dict_u
import numpy as np
import optax
import pandas as pd
from lib_agent.buffer.storage import ReplayStorage
from lib_agent.critic.qrc_critic import jax_u
from lib_config.config import config, list_
from lib_defs.config_defs.tag_config import TagType

from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.imputer_stage import BaseImputer, BaseImputerStageConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.trace import TraceConfig, TraceConstructor, TraceTemporalState
from corerl.state import AppState

logger = logging.getLogger(__name__)

@dataclass
class MaskedAETemporalState:
    trace_ts: TraceTemporalState = field(default_factory=TraceTemporalState)
    last_trace: jax.Array | None = None
    num_outside_thresh: int = 0


@config(frozen=True)
class TrainingConfig:
    init_train_steps = 100
    batch_size: int = 64
    stepsize: float = 1e-4
    err_tolerance: float = 1e-4
    max_update_steps: int = 100
    training_missing_perc: float = 0.25

@config()
class MaskedAEConfig(BaseImputerStageConfig):
    name: Literal["masked-ae"] = "masked-ae"
    horizon: int = 10
    trace_values: list[float] = list_([0.5, 0.9, 0.95])
    buffer_size: int = 50_000

    fill_val: float = 0.0
    prop_missing_tol: float = 0.5
    train_cfg: TrainingConfig = field(default_factory=TrainingConfig)

class ImputeData(NamedTuple):
    obs: jax.Array # raw observation
    traces: jax.Array
    obs_nanmask: jax.Array # bool array set to True where raw obs was nan

class CircularBuffer:
    def __init__(self, max_size: int):
        self._storage = ReplayStorage[ImputeData](capacity=max_size)
        self._i = 0
        self._max_size = max_size

    @property
    def size(self):
        return self._storage.size()

    def add(self, data: ImputeData):
        self._storage.add(data)

    def sample_batches(self, batch_size: int, n_batches: int):
        idxs = [np.random.choice(self.size, size=batch_size, replace=True) for _ in range(n_batches)]
        return self._storage.get_ensemble_batch(idxs)

class MaskedAutoencoder(BaseImputer):
    def __init__(self, imputer_cfg: MaskedAEConfig, app_state: AppState, tag_cfgs: list[TagConfig]):
        super().__init__(imputer_cfg, app_state, tag_cfgs)
        self._dormant = True # dormant until NaN encountered online
        self._cfg = imputer_cfg
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
                nets.LinearConfig(size=2*self._num_obs, activation="relu"),
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
        self._optim = optax.adam(learning_rate=imputer_cfg.train_cfg.stepsize)
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
        total_nans = 0
        total_imputes = 0
        for i in range(len(df)):
            row = df.iloc[i].copy(deep=False)
            row_idx = pf.data.index[i]
            obs_series = row[self._obs_names].astype(np.float32)

            # inputs to the NN are the current row
            # and the prior set of traces -- note use of
            # prior traces is still a valid summary of history
            obs_jax = _series_to_jax(obs_series)
            obs_nanmask = jnp.isnan(obs_jax)
            obs = ImputeData(
                obs=obs_jax,
                traces=ts.last_trace,
                obs_nanmask=obs_nanmask,
            )

            num_nan = obs_nanmask.sum()
            total_nans += num_nan
            perc_nan = num_nan / self._num_obs

            should_impute = num_nan > 0
            can_impute = perc_nan <= self._cfg.train_cfg.training_missing_perc
            within_horizon = ts.num_outside_thresh <= self._cfg.horizon
            if len(df) == 1 and not (can_impute or within_horizon):
                logger.warning(f"Unable to impute at {row_idx}: "
                               f"{perc_nan*100:3.2f}% of observations are NaN. "
                               f"Tolerance is {self._cfg.train_cfg.training_missing_perc*100:3.2f}%")

            # only impute if
            #   1. We need to (i.e. there are missing values)
            #   2. We can (i.e. there aren't too many missing values)
            #   3. Or we are within our imputation horizon
            will_impute = should_impute and (can_impute or within_horizon)
            if will_impute:
                obs_jax = self.impute(obs)
                pf.data.loc[row_idx, self._obs_names] = np.asarray(obs_jax)
                obs_series[:] = obs_jax
                total_imputes += 1

            # if there is enough info to impute, there
            # is enough info to train the imputer.
            if can_impute:
                self._buffer.add(obs)
                ts.num_outside_thresh = 0
            else:
                ts.num_outside_thresh += 1

            # update traces for use on next timestep
            carry = TransformCarry(df, obs_series.to_frame().T, "")
            carry, ts.trace_ts = self._traces(carry, ts.trace_ts)
            ts.last_trace = _series_to_jax(carry.transform_data.iloc[0])

        # log number of nans and whether imputation occurred
        self._app_state.metrics.write(self._app_state.agent_step, metric='AE-num_nan', value=total_nans)
        self._app_state.metrics.write(self._app_state.agent_step, metric='AE-imputed', value=total_imputes)

        self.train()
        return pf

    def impute(self, data: ImputeData) -> jax.Array:
        """
        Runs a forward pass through the AE to get predicted
        values for *all* inputs. Then selectively grabs
        predictions only for the inputs that are NaN.
        """
        if self._dormant and self._buffer.size > 0:
            self._dormant = False
            logger.info("Imputation requested for the first time: AutoEncoder Imputer enabled.")
            for _ in range(self._cfg.train_cfg.init_train_steps): self.train()

        raw_obs = data.obs
        input_obs = jnp.where(data.obs_nanmask, self._fill_val, raw_obs)
        inputs = jnp.hstack((input_obs, data.traces, data.obs_nanmask))
        ae_predictions = self._forward(self._params, inputs)
        return jnp.where(data.obs_nanmask, ae_predictions, raw_obs)

    @jax_u.method_jit
    def _forward(self, params: chex.ArrayTree, inputs: jax.Array) -> jax.Array:
        return self._net.apply(params, inputs)

    def train(self):
        train_cfg = self._cfg.train_cfg
        if self._dormant:
            return

        batches = self._buffer.sample_batches(train_cfg.batch_size, train_cfg.max_update_steps)
        self._rng, train_rng = jax.random.split(self._rng)
        self._params, self._opt_state, loss = _train(
            params=self._params,
            opt_state=self._opt_state,
            rng=train_rng,
            batches=batches,
            train_cfg=train_cfg,
            fill_val=self._cfg.fill_val,
            net=self._net,
            optim=self._optim,
        )
        self._app_state.metrics.write(self._app_state.agent_step, metric="AE-loss", value=loss)

# jit on this function prevents a memory leak with jax.lax.while_loop
@partial(jax_u.jit, static_argnums=(4,5,6,7))
def _train(
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    rng: chex.PRNGKey,
    batches: ImputeData,
    train_cfg: TrainingConfig,
    fill_val: float,
    net: hk.Transformed,
    optim: optax.GradientTransformation,
):

    class TrainCarry(NamedTuple):
        params: chex.ArrayTree
        opt_state: chex.ArrayTree
        loss: float
        rng: chex.PRNGKey
        step: int

    def continue_training(carry: TrainCarry):
        return (carry.loss > train_cfg.err_tolerance) & (carry.step < train_cfg.max_update_steps)

    def train_step(carry: TrainCarry):
        raw_obs_batch = batches.obs[carry.step]
        trace_batch = batches.traces[carry.step]
        label_nanmask = batches.obs_nanmask[carry.step]

        # replace nans in the labels
        label_batch = jnp.where(label_nanmask, fill_val, raw_obs_batch)

        # To force the AE to learn in the presence of
        # missingness, need to fake some missingness
        # in the inputs only (i.e. learn a mapping from
        # missing value to non-missing value)
        rng, sample_rng = jax.random.split(carry.rng)
        uniform_sample = jax.random.uniform(key=sample_rng, shape=raw_obs_batch.shape)
        sim_missing_obs_mask = uniform_sample < train_cfg.training_missing_perc
        obs_batch = jnp.where(sim_missing_obs_mask, jnp.nan, raw_obs_batch)

        # update missing indicator
        missing_mask = sim_missing_obs_mask | label_nanmask
        # update input to include simulated nans
        input_batch = jnp.hstack((obs_batch, trace_batch, missing_mask))

        # replace nans in input - these could be simulated or real
        input_nanmask = jnp.isnan(input_batch)
        input_batch = jnp.where(input_nanmask, fill_val, input_batch)

        params, opt_state, loss = _update(
            params=carry.params,
            opt_state=carry.opt_state,
            input_batch=input_batch,
            label_batch=label_batch,
            label_nanmask=label_nanmask,
            net=net,
            optim=optim,
        )
        return TrainCarry(
            params=params,
            opt_state=opt_state,
            loss=loss,
            rng=rng,
            step=carry.step + 1,
        )

    carry = TrainCarry(
            params=params,
            opt_state=opt_state,
            loss=jnp.inf,
            rng=rng,
            step=0,
    )
    carry = jax.lax.while_loop(continue_training, train_step, carry)

    return carry.params, carry.opt_state, carry.loss

def _update(
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    input_batch: jax.Array,
    label_batch: jax.Array,
    label_nanmask: jax.Array,
    net: hk.Transformed,
    optim: optax.GradientTransformation,
):
    loss, grads = jax.value_and_grad(_batch_loss)(params, input_batch, label_batch, label_nanmask, net)
    updates, new_opt_state = optim.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

def _batch_loss(
    params: chex.ArrayTree,
    input_batch: jax.Array,
    label_batch: jax.Array,
    label_nanmask: jax.Array,
    net: hk.Transformed,
):
    losses = jax_u.vmap_except(_loss, exclude=["params", "net"])(params, input_batch, label_batch, label_nanmask, net)
    return losses.sum() / len(losses)

def _loss(params: chex.ArrayTree, input: jax.Array, label: jax.Array, label_nanmask: jax.Array, net: hk.Transformed):
    pred = net.apply(params, input)
    # loss is a standard MSE except NaN labels
    # are masked out
    error = (~label_nanmask) * (pred - label)
    return jnp.sum(error**2) / jnp.sum(~label_nanmask)


def _series_to_jax(row: pd.Series) -> jax.Array:
    return jnp.asarray(row.to_numpy(np.float32))
