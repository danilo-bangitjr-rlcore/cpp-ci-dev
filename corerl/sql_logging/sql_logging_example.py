import unittest
from corerl.sql_logging import sql_logging
from omegaconf import OmegaConf
from sqlalchemy_utils import database_exists, drop_database, create_database
from sqlalchemy import MetaData, select, ForeignKeyConstraint
import sqlalchemy
from corerl.component.network.factory import init_critic_network
import torch
import pandas as pd
from dataclasses import dataclass
from sqlalchemy.orm import Session
from hydra import compose, initialize
from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON, PickleType, DateTime
from typing import Any
from torch import Tensor
from datetime import datetime
import torch
import pandas as pd
from dataclasses import dataclass
from corerl.component.network.factory import init_critic_network

from typing_extensions import Annotated

from sqlalchemy.sql import func
from sqlalchemy.orm import mapped_column
from sqlalchemy import ForeignKeyConstraint

timestamp = Annotated[
    datetime,
    mapped_column(
      DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
]

@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s_next: torch.Tensor
    done: torch.Tensor
    terminate: torch.Tensor

    def to_numpy(self):
        numpy_transition = [
            self.s.numpy(),
            self.a.numpy(),
            float(self.r),
            self.s_next.numpy(),
            int(self.done),
            int(self.terminate),
        ]
        return numpy_transition


class Base(DeclarativeBase):
    """
    This base class tells SQLAlchemy to create SQL tables 
    for each class that inherits from it.
    """
    type_annotation_map = {
        Transition: PickleType, # -> backend datatype can vary
        list: JSON,
        dict: PickleType,
    }

class SQLTransition(Base):
    __tablename__ = "transitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    prev_uvt: Mapped[float]
    new_uvt: Mapped[float]
    target_uvt: Mapped[float]
    dose: Mapped[float]
    state: Mapped[list]
    action: Mapped[list]
    reward: Mapped[list]
    next_state: Mapped[list] = mapped_column(nullable=True)
    transition: Mapped[Transition]
    exclude: Mapped[bool] = mapped_column(default=False)
    raw_transition_info: Mapped["RawTransition"] = relationship(back_populates="transitions")
    run: Mapped[int]
    step: Mapped[int]
    __table_args__ = (
        ForeignKeyConstraint(
            ["run", "step"], ["raw_transition_info.run", "raw_transition_info.step"]
        ),
    )

class RawTransition(Base):
    __tablename__ = "raw_transition_info"
    run: Mapped[int] = mapped_column(primary_key=True)
    step: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    prev_uvt: Mapped[float]
    new_uvt: Mapped[float]
    target_uvt: Mapped[float]
    dose: Mapped[float]
    transitions: Mapped[List["SQLTransition"]] = relationship(back_populates="raw_transition_info")

class CriticWeights(Base):
    __tablename__ = "critic_weights"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    critic_weights: Mapped[dict] # can do pickletype

def get_transition():
    state = torch.tensor([1, 2], dtype=torch.float)
    action = torch.tensor(0.5, dtype=torch.float)
    reward = torch.tensor(-1.0, dtype=torch.float)
    state_next = torch.tensor([1, 2], dtype=torch.float)

    transition = Transition(state, action, reward, state_next, False, False)
    return transition

def init_critic():
    with initialize(version_base=None, config_path="../../config/agent/critic/critic_network"):
        critic_cfg = compose(config_name="ensemble", overrides=["base.arch=[16,16]", "ensemble=1"])

    critic = init_critic_network(cfg=critic_cfg, input_dim=21, output_dim=1)
        
    return critic

def get_run_id(engine):
    df = pd.read_sql(
        select(RawTransition.run).order_by(RawTransition.run.desc()).limit(1),
        con=engine,
    )

    run_id = df["run"].iloc[0]
    new_run_id = run_id + 1

    return new_run_id

def main():
    con_cfg = OmegaConf.load('config/db/sql/credentials_vpn.yaml') # add your own credentials at this path
    engine = sql_logging.get_sql_engine(con_cfg, db_name='orm_test_db')

    if not database_exists(engine.url):
        create_database(engine.url)
    Base.metadata.create_all(engine)

    # example of writing transitions to db
    raw_transition = RawTransition(
        run=get_run_id(engine), 
        step=0, 
        prev_uvt=89, 
        new_uvt=91, 
        target_uvt=90, 
        dose=6
    )
    
    new_transition = SQLTransition( 
        prev_uvt=raw_transition.prev_uvt/100,
        new_uvt=raw_transition.new_uvt/100,
        target_uvt=raw_transition.target_uvt/100,
        dose=raw_transition.dose/15
    )

    raw_transition.transitions.append(new_transition)

    new_transition.action = new_transition.dose
    new_transition.state = [new_transition.prev_uvt, new_transition.target_uvt]
    new_transition.reward = abs(raw_transition.new_uvt - raw_transition.target_uvt)
    new_transition.transition = get_transition()
    
    with Session(engine) as session:
        session.add(raw_transition)
        session.commit()

    # example of storing critic weights to db
    critic_before_db = init_critic()

    critic_weights = CriticWeights(critic_weights=critic_before_db.state_dict())
    with Session(engine) as session:
        session.add(critic_weights)
        session.commit()

    # fetch weights with sql query
    df = pd.read_sql(select(CriticWeights).order_by(CriticWeights.ts.desc()).limit(1), con=engine)

    loaded_stated_dict = df["critic_weights"].iloc[0]

    critic_after_db = init_critic()
    critic_after_db.load_state_dict(loaded_stated_dict)
    
    test_input = torch.rand((1,21))
    assert torch.isclose(critic_after_db(test_input)[0], critic_before_db(test_input)[0])

if __name__ == "__main__":
    main()