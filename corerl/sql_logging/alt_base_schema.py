from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON, PickleType, DateTime
from datetime import datetime
import torch
from dataclasses import dataclass

from typing_extensions import Annotated

from sqlalchemy.sql import func
from sqlalchemy.orm import mapped_column
from sqlalchemy import ForeignKeyConstraint


timestamp = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    ),
]

class Base(DeclarativeBase):
    # type_annotation_map = {
    #     list: JSON,
    #     dict: PickleType,
    # }
    pass


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    hparams: Mapped[List["HParam"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    transition_info: Mapped[List["TransitionInfo"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class HParam(Base):
    __tablename__ = "hparams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    val = mapped_column(JSON)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"))
    run: Mapped["Run"] = relationship(back_populates="hparams")


class SQLTransition(Base):
    __tablename__ = "transitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]

    # Data
    state = mapped_column(JSON)
    action = mapped_column(JSON)
    reward = mapped_column(JSON)
    next_state = mapped_column(JSON, nullable=True)
    # transition: Mapped[Transition]
    exclude: Mapped[bool] = mapped_column(default=False)

    # Foreign Keys
    # run_id: Mapped[int]
    # step: Mapped[int]

    # Object relationship
    transition_info: Mapped[List["TransitionInfo"]] = relationship(
        back_populates="transition",
    )

class TransitionInfo(Base):
    # __abstract__ = True
    __tablename__ = "transition_info"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"))
    step: Mapped[int] # = mapped_column(primary_key=True)
    insert_ts: Mapped[timestamp]
    type: Mapped[str] = mapped_column(String(100))

    trans_id: Mapped[int] = mapped_column(ForeignKey("transitions.id"))
    trans_ts: Mapped[timestamp]

    # Data
    """
    Inherit from this class and add your data here.
    For example:
    
    prev_uvt: Mapped[float]
    new_uvt: Mapped[float]
    target_uvt: Mapped[float]
    dose: Mapped[float]
    """

    # Relationships
    transition: Mapped["SQLTransition"] = relationship(
        back_populates="transition_info"
    )

    run: Mapped["Run"] = relationship(back_populates="transition_info")

    __mapper_args__ = {
        "polymorphic_identity": "transition_info",
        "polymorphic_on": "type",
    }

class CriticWeights(Base):
    __tablename__ = "critic_weights"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    critic_weights = mapped_column(PickleType)
