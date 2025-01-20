import logging

import numpy as np
from line_profiler import profile
from sqlalchemy.orm import Mapped, mapped_column

import corerl.sql_logging.base_schema as sql
from corerl.sql_logging.base_schema import SQLTransition, TransitionInfo

logger = logging.getLogger(__name__)


class CoagTransitionInfo(TransitionInfo):

    prev_uvt: Mapped[float] = mapped_column(nullable=True)
    new_uvt: Mapped[float] = mapped_column(nullable=True)
    target_uvt: Mapped[float] = mapped_column(nullable=True)
    dose: Mapped[float] = mapped_column(nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "normalized",
    }


class RawCoagTransitionInfo(CoagTransitionInfo):

    __mapper_args__ = {
        "polymorphic_identity": "raw",
    }


class CoagBanditSimEnv:
    """
    environment for controlling coag dosing
    """

    @profile
    def __init__(self) -> None:
        self.step_num = 0
        self.last_action = None
        self.uvt = 89

    def reset(self):
        self.state = [0.89, 0.95]

        info = None
        return np.array(self.state), info

    def ready_for_action(self):
        return True

    def start_step(self, action: list[float]):
        assert action.shape == (1,)
        # deploy new setpoint
        logger.info(f"Deploying action: {action}...")
        self.last_action = action
        # do something to the real world here

    def finish_step(self):
        """
        Here we've waited for the real world to respond to our action
        """
        action = self.last_action
        new_uvt = action[0] * (95 - 89) + 89

        # log raw observations
        raw_tinfo = RawCoagTransitionInfo(
            step=sql.stepper.step,
            prev_uvt=self.uvt,
            new_uvt=new_uvt,
            target_uvt=95,
            dose=action[0] * 15,
        )

        # normalize raw observations
        norm_tinfo = CoagTransitionInfo(
            step=sql.stepper.step,
            prev_uvt=raw_tinfo.prev_uvt / 100,
            new_uvt=raw_tinfo.new_uvt / 100,
            target_uvt=raw_tinfo.target_uvt / 100,
            dose=raw_tinfo.dose / 15,
        )

        # next state is based on normalized observations
        state = self.state
        action = action.tolist()  # list to make human readable in sql
        reward = -abs(norm_tinfo.target_uvt - norm_tinfo.new_uvt)
        next_state = [norm_tinfo.prev_uvt, norm_tinfo.target_uvt]

        transition = SQLTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            transition_info=[norm_tinfo, raw_tinfo],
            step=sql.stepper.step,
        )

        self.step_num += 1
        self.uvt = new_uvt
        self.state = next_state

        return transition
