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
from rlcorelib.root.control.src.network.factory import init_critic_network

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