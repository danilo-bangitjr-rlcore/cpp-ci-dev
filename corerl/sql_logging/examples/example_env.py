from line_profiler import profile
import numpy as np
import numpy as np
from corerl.sql_logging.base_schema import (
    Run,
    TransitionInfo,
    SQLTransition,
    Step
)

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
import logging

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

    def register_run(self, run: Run):
        self.run = run

    def get_step(self):
        step = Step(step_num=self.step_num, run=self.run)
        return step

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

        step = self.get_step() # this is a reference to an sql row TODO: test what happens if this gets called multiple times

        # log raw observations
        raw_tinfo = RawCoagTransitionInfo(
            step=step,
            prev_uvt=self.uvt,
            new_uvt=new_uvt,
            target_uvt=95,
            dose=action[0] * 15
        )

        # normalize raw observations
        norm_tinfo = CoagTransitionInfo(
            step=step,
            prev_uvt=raw_tinfo.prev_uvt / 100,
            new_uvt=raw_tinfo.new_uvt / 100,
            target_uvt=raw_tinfo.target_uvt / 100,
            dose=raw_tinfo.dose / 15
        )

        # next state is based on normalized observations
        state = self.state
        action = action.tolist() # list to make human readable in sql
        reward = -abs(norm_tinfo.target_uvt - norm_tinfo.new_uvt)
        next_state = [norm_tinfo.prev_uvt, norm_tinfo.target_uvt]

        transition = SQLTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            transition_info=[
                norm_tinfo,
                raw_tinfo
            ],
            step=step
        )

        self.step_num += 1
        self.uvt = new_uvt
        self.state = next_state

        return transition