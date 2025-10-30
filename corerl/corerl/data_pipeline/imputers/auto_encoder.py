import logging
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_agent.network.networks as nets
import lib_utils.dict as dict_u
import lib_utils.jax as jax_u
import numpy as np
import optax
from lib_agent.buffer.storage import ReplayStorage
from lib_defs.config_defs.tag_config import TagType
from lib_utils.named_array import NamedArray

from corerl.configs.data_pipeline.imputers.auto_encoder import MaskedAEConfig, TrainingConfig
from corerl.configs.data_pipeline.transforms.nuke import NukeConfig
from corerl.configs.data_pipeline.transforms.trace import TraceConfig
from corerl.configs.tags.tag_config import TagConfig
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.base import BaseImputer
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.trace import TraceConstructor, TraceTemporalState, log_trace_quality
from corerl.state import AppState

logger = logging.getLogger(__name__)


@dataclass
class MaskedAETemporalState:
    trace_ts: TraceTemporalState = field(default_factory=TraceTemporalState)
    last_trace: NamedArray | None = None
    num_outside_thresh: int = 0

class ImputeData(NamedTuple):
    obs: NamedArray # raw observation
    traces: NamedArray
    obs_nanmask: NamedArray # bool array set to True where raw obs was nan
    trace_nanmask: NamedArray # bool array set to True when trace was nan

class DebugInfo(NamedTuple):
    losses: jax.Array
    pred: jax.Array
    error: jax.Array


def _to_input(data: ImputeData):
    return jnp.hstack((data.obs.array, data.traces.array, data.obs_nanmask.array, data.trace_nanmask.array))

class CircularBuffer:
    def __init__(self, max_size: int):
        self._storage = ReplayStorage[ImputeData](capacity=max_size)
        self._i = 0
        self._max_size = max_size

    @property
    def size(self):
        return self._storage.size()

    def add(self, data: ImputeData):
        if jnp.all(data.obs_nanmask.array):
            logger.warning("Attempted to add all NaN obs to AE buffer - rejecting...")
            logger.warning(f"All NaN obs timestamp: {data.obs.timestamps}")
            return
        self._storage.add(data)

    def add_bulk(self, data: ImputeData):
        self._storage.add_bulk(data)

    def sample_batches(self, batch_size: int, n_batches: int):
        idxs = [np.random.choice(self.size, size=batch_size, replace=True) for _ in range(n_batches)]
        return self._storage.get_ensemble_batch(idxs)

def obs_filter(tag_config: TagConfig):
    """
    return False if the tag should be removed from obs
    """
    sc = tag_config.state_constructor
    if sc is None:
        return False
    if len(sc) == 1 and isinstance(sc[0], NukeConfig):
        return False
    if tag_config.type == TagType.meta:
        return False
    return True

class MaskedAutoencoder(BaseImputer):
    def __init__(self, imputer_cfg: MaskedAEConfig, app_state: AppState, tag_cfgs: list[TagConfig]):
        super().__init__(imputer_cfg, app_state, tag_cfgs)
        self._dormant = True # dormant until NaN encountered online
        self._cfg = imputer_cfg
        self._obs_names = [t.name for t in tag_cfgs if obs_filter(t)]
        self._num_obs = len(self._obs_names)
        self._num_traces = len(imputer_cfg.trace_values)
        self._fill_val = imputer_cfg.fill_val
        self._bulk_load_trigger = 1000
        self._min_buffer = 50
        self._debug = imputer_cfg.debug

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
        # input includes (obs, traces) + (obs_nanmask, trace_nanmask)
        in_shape = 2 * (1 + self._num_traces) * self._num_obs
        self._params = self._net.init(init_rng, jnp.ones(in_shape))
        self._optim = optax.adam(learning_rate=imputer_cfg.train_cfg.stepsize)
        self._opt_state = self._optim.init(self._params)
        assert not np.isnan(self._cfg.prop_missing_tol)

    def filter_nans(self, impute_data: ImputeData):
        num_nan_obs = impute_data.obs_nanmask.array.sum(axis=1)
        num_nan_trace = impute_data.trace_nanmask.array.sum(axis=1)

        perc_nan_obs = num_nan_obs / self._num_obs
        perc_nan_trace = num_nan_trace / (self._num_traces * self._num_obs)

        nan_tol = self._cfg.prop_missing_tol
        can_impute = (perc_nan_obs <= nan_tol) & (perc_nan_trace <= nan_tol)
        impute_data = ImputeData(
            obs=impute_data.obs[can_impute],
            traces=impute_data.traces[can_impute],
            obs_nanmask=impute_data.obs_nanmask[can_impute],
            trace_nanmask=impute_data.trace_nanmask[can_impute],
        )

        all_nan_mask = jnp.all(impute_data.obs_nanmask.array, axis=1)
        if jnp.all(all_nan_mask):
            return None

        return ImputeData(
            obs=impute_data.obs[~all_nan_mask],
            traces=impute_data.traces[~all_nan_mask],
            obs_nanmask=impute_data.obs_nanmask[~all_nan_mask],
            trace_nanmask=impute_data.trace_nanmask[~all_nan_mask],
        )

    def load(self, pf: PipelineFrame):
        logger.info("AE bulk loading data...")
        ts = dict_u.assign_default(pf.temporal_state, StageCode.IMPUTER, MaskedAETemporalState)
        assert isinstance(ts, MaskedAETemporalState)
        data = pf.data.copy(deep=False)[self._obs_names]
        obs = NamedArray.from_pandas(data)
        obs_nanmask = obs.set(jnp.isnan(obs.array))

        # precompute traces in jitted stream
        carry = TransformCarry(data, data.copy(deep=False), "")
        logger.info("\tAE computing traces...")
        carry, ts.trace_ts = self._traces(carry, ts.trace_ts)
        all_traces = NamedArray.from_pandas(carry.transform_data)
        trace_nanmask = all_traces.set(jnp.isnan(all_traces.array))
        ts.last_trace = all_traces[-1]

        impute_data = ImputeData(
            obs=obs.set(jnp.where(obs_nanmask.array, self._fill_val, obs.array)),
            traces=all_traces.set(jnp.where(trace_nanmask.array, self._fill_val, all_traces.array)),
            obs_nanmask=obs_nanmask,
            trace_nanmask=trace_nanmask,
        )
        impute_data = self.filter_nans(impute_data)
        if impute_data is None:
            logger.warning("Tried to load data with only NaN. No data added to buffer.")
            return

        # create example and add to buffer to init
        logger.info("\tAE adding impute data to buffer...")
        self._buffer.add_bulk(impute_data)

    def warmup(self):
        logger.info("Training autoencoder...")
        self._dormant = False
        for i in range(self._cfg.train_cfg.init_train_steps):
            logger.info(f"\tTrain step {i}")
            self.train()
        logger.info("Done training autoencoder")

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        bulk_load = len(pf.data) > self._bulk_load_trigger
        if bulk_load:
            logger.info(f"AE called with data of len greater than {self._bulk_load_trigger}. Bulk loading...")
            self.load(pf)
            self.warmup()
            self._dormant = False
            # if bulk loading, dont impute
            return pf

        ts = dict_u.assign_default(pf.temporal_state, StageCode.IMPUTER, MaskedAETemporalState)
        assert isinstance(ts, MaskedAETemporalState)

        df = pf.data.copy(deep=False)[self._obs_names]
        all_obs = NamedArray.from_pandas(df)

        # try to recover traces from the temporal state
        # otherwise, start fresh
        if ts.last_trace is None:
            obs_row = df.iloc[[0]][self._obs_names].copy(deep=False)
            carry = TransformCarry(df, obs_row, "")
            carry, ts.trace_ts = self._traces(carry, ts.trace_ts)
            ts.last_trace = NamedArray.from_pandas(carry.transform_data)[0]

        # loop through data and impute one row at a time
        # this way we can use imputed values to compute
        # the traces
        total_nan_obs = 0
        total_nan_trace = 0
        total_imputes = 0
        for i in range(len(df)):
            row = df.iloc[i].copy(deep=False)
            row_idx = pf.data.index[i]
            obs_row = row[self._obs_names].astype(np.float32).to_frame().T

            # inputs to the NN are the current row
            # and the prior set of traces -- note use of
            # prior traces is still a valid summary of history
            obs = all_obs[i]
            obs_nanmask = obs.set(jnp.isnan(obs.array))
            trace_nanmask = ts.last_trace.set(jnp.isnan(ts.last_trace.array))
            assert obs.names == obs_nanmask.names
            assert ts.last_trace.names == trace_nanmask.names
            impute_data = ImputeData(
                obs=obs.set(jnp.where(obs_nanmask.array, self._fill_val, obs.array)),
                traces=ts.last_trace.set(jnp.where(trace_nanmask.array, self._fill_val, ts.last_trace.array)),
                obs_nanmask=obs_nanmask,
                trace_nanmask=trace_nanmask,
            )

            num_nan_obs = obs_nanmask.array.sum()
            total_nan_obs += num_nan_obs
            perc_nan_obs = num_nan_obs / self._num_obs

            num_nan_trace = trace_nanmask.array.sum()
            total_nan_trace += num_nan_trace
            perc_nan_trace = num_nan_trace / (self._num_traces * self._num_obs)

            should_impute = num_nan_obs > 0
            nan_tol = self._cfg.prop_missing_tol
            can_impute = (perc_nan_obs <= nan_tol) and (perc_nan_trace <= nan_tol)
            within_horizon = ts.num_outside_thresh <= self._cfg.horizon
            if len(df) == 1 and not (can_impute or within_horizon):
                logger.warning(f"Unable to impute at {row_idx}: "
                               f"{perc_nan_obs*100:3.2f}% of observations are NaN. "
                               f"{perc_nan_trace*100:3.2f}% of traces are NaN. "
                               f"Tolerance is {self._cfg.train_cfg.training_missing_perc*100:3.2f}%")

            # only impute if
            #   1. We need to (i.e. there are missing values)
            #   2. We can (i.e. there aren't too many missing values)
            #   3. Or we are within our imputation horizon
            will_impute = should_impute and (can_impute or within_horizon) and self._buffer.size > 0
            if will_impute:
                obs = self.impute(impute_data)
                pf.data.loc[row_idx, self._obs_names] = np.asarray(obs)
                obs_row[:] = obs
                total_imputes += 1

            # if there is enough info to impute, there
            # is enough info to train the imputer.
            if can_impute:
                self._buffer.add(impute_data)
                ts.num_outside_thresh = 0
            else:
                ts.num_outside_thresh += 1

            # update traces for use on next timestep
            carry = TransformCarry(df, obs_row, "")
            carry, ts.trace_ts = self._traces(carry, ts.trace_ts)
            ts.last_trace = NamedArray.from_pandas(carry.transform_data)[0]

        # log number of nans and whether imputation occurred
        self._app_state.metrics.write(self._app_state.agent_step, metric='AE-num_nan_obs', value=total_nan_obs)
        self._app_state.metrics.write(self._app_state.agent_step, metric='AE-num_nan_trace', value=total_nan_trace)
        self._app_state.metrics.write(self._app_state.agent_step, metric='AE-imputed', value=total_imputes)
        log_trace_quality(self._app_state, prefix='AE', decays=self._cfg.trace_values, trace_ts=ts.trace_ts)

        self.train()
        return pf

    def impute(self, data: ImputeData) -> NamedArray:
        """
        Runs a forward pass through the AE to get predicted
        values for *all* inputs. Then selectively grabs
        predictions only for the inputs that are NaN.
        """
        if self._buffer.size < self._min_buffer:
            # pass nans forward until we have enough data to impute
            return data.obs.set(jnp.where(data.obs_nanmask.array, jnp.nan, data.obs.array))
        if self._dormant:
            self._dormant = False
            logger.info("Imputation requested for the first time: AutoEncoder Imputer enabled.")
            self.warmup()

        inputs = _to_input(data)
        ae_predictions = self._forward(self._params, inputs)
        return data.obs.set(jnp.where(data.obs_nanmask.array, ae_predictions, data.obs.array))

    @jax_u.method_jit
    def _forward(self, params: chex.ArrayTree, inputs: jax.Array) -> jax.Array:
        return self._net.apply(params, inputs)

    def train(self):
        train_cfg = self._cfg.train_cfg
        if self._dormant:
            return

        batches = self._buffer.sample_batches(train_cfg.batch_size, train_cfg.max_update_steps)

        if self._debug:
            for i, leaf in enumerate(jax.tree.leaves(batches)):
                if jnp.isnan(leaf).any():
                    logger.error(f"nan in AE training batch, leaf {i}")
                if jnp.isinf(leaf).any():
                    logger.error(f"inf in AE training batch, leaf {i}")

        self._rng, train_rng = jax.random.split(self._rng)
        self._params, self._opt_state, batch_losses, steps, debug_info = _train(
            params=self._params,
            opt_state=self._opt_state,
            rng=train_rng,
            batches=batches,
            train_cfg=train_cfg,
            fill_val=self._cfg.fill_val,
            net=self._net,
            optim=self._optim,
            debug=self._debug,
        )
        batch_losses = batch_losses[:steps]

        if self._debug and (
            jnp.isnan(batch_losses).any()
            or jnp.isnan(debug_info.losses).any()
            or jnp.isnan(debug_info.pred).any()
            or jnp.isnan(debug_info.error).any()
            or jnp.isinf(batch_losses).any()
            or jnp.isinf(debug_info.losses).any()
            or jnp.isinf(debug_info.pred).any()
            or jnp.isinf(debug_info.error).any()
        ):
            logger.error("nan or inf detected during AE training")

        self._app_state.metrics.write(self._app_state.agent_step, metric="AE-updates", value=steps)
        self._app_state.metrics.write(self._app_state.agent_step, metric="AE-loss", value=batch_losses[-1])


class TrainCarry(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree
    losses: jax.Array
    debug_info: DebugInfo
    rng: chex.PRNGKey
    step: int

# jit on this function prevents a memory leak with jax.lax.while_loop
@partial(jax_u.jit, static_argnums=(4, 5, 6, 7, 8))
def _train(
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    rng: chex.PRNGKey,
    batches: ImputeData,
    train_cfg: TrainingConfig,
    fill_val: float,
    net: hk.Transformed,
    optim: optax.GradientTransformation,
    debug: bool,
):

    batch_size = batches.obs.shape[1]
    input_size = batches.obs.shape[2]
    def continue_training(carry: TrainCarry):
        return (carry.losses[carry.step-1] > train_cfg.err_tolerance) & (carry.step < train_cfg.max_update_steps)

    def train_step(carry: TrainCarry):
        true_obs_batch = batches.obs[carry.step]
        true_obs_nanmask = batches.obs_nanmask[carry.step]
        true_trace_batch = batches.traces[carry.step]
        true_trace_nanmask = batches.trace_nanmask[carry.step]

        # To force the AE to learn in the presence of
        # missingness, need to fake some missingness
        # in the inputs only (i.e. learn a mapping from
        # missing value to non-missing value)
        rng, sample_rng = jax.random.split(carry.rng)
        uniform_sample = jax.random.uniform(key=sample_rng, shape=true_obs_batch.shape)
        sim_obs_nanmask = uniform_sample < train_cfg.training_missing_perc
        train_obs_batch = true_obs_batch.set(jnp.where(sim_obs_nanmask, fill_val, true_obs_batch.array))
        # update missing indicator
        train_obs_nanmask = true_obs_nanmask.set(sim_obs_nanmask | true_obs_nanmask.array)

        # simulate missingness of traces
        rng, sample_rng = jax.random.split(rng)
        uniform_sample = jax.random.uniform(key=sample_rng, shape=true_trace_batch.shape)
        sim_trace_nanmask = uniform_sample < train_cfg.training_missing_perc
        train_trace_batch = true_trace_batch.set(jnp.where(sim_trace_nanmask, fill_val, true_trace_batch.array))
        # update missing indicator
        train_trace_nanmask = true_trace_nanmask.set(sim_trace_nanmask | true_trace_nanmask.array)

        # update impute_data to include simulated nans
        train_impute_data = ImputeData(
            obs=train_obs_batch,
            traces=train_trace_batch,
            obs_nanmask=train_obs_nanmask,
            trace_nanmask=train_trace_nanmask,
        )
        input_batch = _to_input(train_impute_data)

        params, opt_state, loss, new_debug_info = _update(
            params=carry.params,
            opt_state=carry.opt_state,
            input_batch=input_batch,
            label_batch=true_obs_batch.array,
            label_nanmask=true_obs_nanmask.array,
            net=net,
            optim=optim,
            debug=debug,
        )

        if debug:
            new_losses = new_debug_info.losses.flatten()
            new_pred = new_debug_info.pred.flatten()
            new_error = new_debug_info.error.flatten()

            debug_info = DebugInfo(
                losses=carry.debug_info.losses.at[carry.step].set(new_losses),
                pred=carry.debug_info.pred.at[carry.step].set(new_pred),
                error=carry.debug_info.error.at[carry.step].set(new_error),
            )
        else:
            debug_info = carry.debug_info

        return TrainCarry(
            params=params,
            opt_state=opt_state,
            losses=carry.losses.at[carry.step].set(loss),
            debug_info=debug_info,
            rng=rng,
            step=carry.step + 1,
        )

    carry = TrainCarry(
        params=params,
        opt_state=opt_state,
        losses=jnp.inf * jnp.ones(train_cfg.max_update_steps),
        debug_info=DebugInfo(
            losses=jnp.empty((train_cfg.max_update_steps, batch_size)),
            pred=jnp.empty((train_cfg.max_update_steps, batch_size * input_size)),
            error=jnp.empty((train_cfg.max_update_steps, batch_size * input_size)),
        ),
        rng=rng,
        step=0,
    )
    carry = jax.lax.while_loop(continue_training, train_step, carry)

    return carry.params, carry.opt_state, carry.losses, carry.step, carry.debug_info

def _update(
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    input_batch: jax.Array,
    label_batch: jax.Array,
    label_nanmask: jax.Array,
    net: hk.Transformed,
    optim: optax.GradientTransformation,
    debug: bool,
):
    loss_and_debug, grads = jax.value_and_grad(_batch_loss, has_aux=True)(
        params, input_batch, label_batch, label_nanmask, net, debug,
    )
    loss = loss_and_debug[0]
    debug_info = loss_and_debug[1]
    updates, new_opt_state = optim.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, debug_info

def _batch_loss(
    params: chex.ArrayTree,
    input_batch: jax.Array,
    label_batch: jax.Array,
    label_nanmask: jax.Array,
    net: hk.Transformed,
    debug: bool,
):
    losses, pred, error = jax_u.vmap_except(_loss, exclude=["params", "net"])(
        params, input_batch, label_batch, label_nanmask, net,
    )
    debug_info = (
        DebugInfo(losses=losses, pred=pred, error=error)
        if debug
        else None
    )
    return losses.sum() / len(losses), debug_info


def _loss(params: chex.ArrayTree, input: jax.Array, label: jax.Array, label_nanmask: jax.Array, net: hk.Transformed):
    pred = net.apply(params, input)
    # loss is a standard MSE except NaN labels
    # are masked out
    error = (~label_nanmask) * (pred - label)
    return jnp.sum(error**2) / jnp.sum(~label_nanmask), pred, error
