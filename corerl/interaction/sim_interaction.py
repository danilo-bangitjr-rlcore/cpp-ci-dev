import logging

import numpy as np
import pandas as pd
from torch import Tensor

from corerl.agent.base import BaseAgent
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.eval.actor_critic import ActorCriticEval
from corerl.eval.monte_carlo import MonteCarloEvaluator
from corerl.eval.plotting.evals import plot_evals
from corerl.interaction.configs import SimInteractionConfig
from corerl.interaction.interaction import Interaction
from corerl.messages.events import Event, EventType
from corerl.state import AppState

logger = logging.getLogger(__file__)





class SimInteraction(Interaction):
    def __init__(
        self,
        cfg: SimInteractionConfig,
        app_state: AppState,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
    ):
        self._pipeline = pipeline
        self._env = env
        self._agent = agent
        self._app_state = app_state

        self._column_desc = pipeline.column_descriptions

        self._should_reset = False
        self._last_state: np.ndarray | None = None
        self._last_action: np.ndarray | None = None

        # evals
        self._monte_carlo_eval = MonteCarloEvaluator(
            app_state.cfg.eval_cfgs.monte_carlo,
            app_state,
            agent,
        )
        self._actor_critic_eval = ActorCriticEval(
            self._app_state.cfg.eval_cfgs.actor_critic,
            app_state,
            pipeline,
            agent,
            self._column_desc,
        )


    # -----------------------
    # -- Lifecycle Methods --
    # -----------------------
    def _on_get_obs(self):
        o = self._env.get_latest_obs()
        pipe_return = self._pipeline(o, data_mode=DataMode.ONLINE, reset_temporal_state=self._should_reset)
        self._agent.update_buffer(pipe_return)

        self._should_reset = bool(o['truncated'].any() or o['terminated'].any())

        # capture latest state
        self._last_state = (
            pipe_return.states
            .iloc[0]
            .to_numpy(dtype=np.float32)
        )

        self._last_action = (
            pipe_return.actions
            .iloc[0]
            .to_numpy(dtype=np.float32)
        )

        # log states
        self._write_to_metrics(pipe_return.states)


        # log rewards
        self._write_to_metrics(pipe_return.rewards)

        # perform evaluations
        self._monte_carlo_eval.execute(pipe_return, "online")
        label = str(self._app_state.agent_step)
        self._actor_critic_eval.execute([Tensor(self._last_state)], [Tensor(self._last_action)], label)
        plot_evals(
            app_state=self._app_state,
            step_start=self._app_state.agent_step,
            step_end=self._app_state.agent_step,
            labels=[label]
        )

        self._app_state.agent_step += 1

    def _on_update(self):
        self._agent.update()

    def _on_emit_action(self):
        sa = self._get_latest_state_action()
        assert sa is not None
        s, a = sa
        if not self._agent.cfg.delta_action:
            a = np.zeros_like(a)

        delta = self._agent.get_action(s)
        a_df = self._pipeline.action_constructor.assign_action_names(a, delta)
        a_df = self._pipeline.preprocessor.inverse(a_df)
        self._env.emit_action(a_df)

        # log actions
        self._write_to_metrics(a_df)

    # ------------------
    # -- No Event Bus --
    # ------------------
    def step(self):
        self._on_get_obs()
        self._on_update()
        self._on_emit_action()


    # ---------------
    # -- Event Bus --
    # ---------------
    def step_event(self, event: Event):
        logger.debug(f"Received step_event: {event}")
        match event.type:
            case EventType.step:
                self.step()

            case EventType.step_get_obs:
                self._on_get_obs()

            case EventType.step_agent_update:
                self._on_update()

            case EventType.step_emit_action:
                self._on_emit_action()

            case _:
                logger.warning(f"Unexpected step_event: {event}")

    # ---------
    # internals
    # ---------
    def _get_latest_state_action(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._last_state is None:
            logger.error("Tried to get interaction state, but none existed")
            return None

        if self._last_action is None:
            logger.error("Tried to get interaction action, but none existed")
            return None

        return self._last_state, self._last_action


    def _write_to_metrics(self, df: pd.DataFrame) -> None:
        if len(df) != 1:
            logger.error(f"unexpected df length: {len(df)}")

        for feat_name in df.columns:
            val = df[feat_name].values[0]
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=feat_name,
                value=val,
            )
