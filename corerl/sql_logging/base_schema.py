from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON, PickleType, DateTime
from datetime import datetime

from typing_extensions import Annotated

from sqlalchemy.sql import func
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy.dialects.mysql import LONGBLOB

timestamp = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    ),
]


class Base(DeclarativeBase):
    # can define type_annotation_map here
    type_annotation_map = {
        dict: JSON,
    }


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    hparams: Mapped[List["HParam"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    steps: Mapped[List["Step"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class HParam(Base):
    __tablename__ = "hparams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    val = mapped_column(JSON)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"))
    run: Mapped["Run"] = relationship(back_populates="hparams")


class Step(Base):
    __tablename__ = "steps"
    ts: Mapped[timestamp]
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"), primary_key=True)
    step_num: Mapped[int] = mapped_column(primary_key=True)

    # Object relationships
    run: Mapped["Run"] = relationship(back_populates="steps")
    transitions: Mapped[List["SQLTransition"]] = relationship(
        back_populates="step", cascade="all, delete-orphan"
    )
    transition_info: Mapped[List["TransitionInfo"]] = relationship(
        back_populates="step", cascade="all, delete-orphan"
    )
    losses: Mapped[List["Loss"]] = relationship(
        back_populates="step", cascade="all, delete-orphan"
    )
    network_weights: Mapped[List["NetworkWeights"]] = relationship(
        back_populates="step", cascade="all, delete-orphan"
    )
    grad_info: Mapped[List["GradInfo"]] = relationship(
        back_populates="step", cascade="all, delete-orphan"
    )


class SQLTransition(Base):
    __tablename__ = "transitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]

    run_id: Mapped[int] = mapped_column(nullable=True)
    step_num: Mapped[int] = mapped_column(nullable=True)

    # Data
    state = mapped_column(JSON)
    action = mapped_column(JSON)
    reward = mapped_column(JSON)
    next_state = mapped_column(JSON, nullable=True)

    exclude: Mapped[bool] = mapped_column(default=False)

    # Object relationships
    transition_info: Mapped[List["TransitionInfo"]] = relationship(
        back_populates="transition",
    )

    step: Mapped["Step"] = relationship(back_populates="transitions")

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "step_num"], ["steps.run_id", "steps.step_num"]
        ),
    )


class Loss(Base):
    __tablename__ = "losses"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]

    run_id: Mapped[int] = mapped_column(nullable=True)
    step_num: Mapped[int] = mapped_column(nullable=True)
    # Data
    loss: Mapped[float]
    type: Mapped[str] = mapped_column(String(100))

    # Relationships
    step: Mapped["Step"] = relationship(back_populates="losses")

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "step_num"], ["steps.run_id", "steps.step_num"]
        ),
    )


class TransitionInfo(Base):
    __tablename__ = "transition_info"
    id: Mapped[int] = mapped_column(primary_key=True)
    insert_ts: Mapped[timestamp]

    run_id: Mapped[int] = mapped_column(nullable=True)
    step_num: Mapped[int] = mapped_column(nullable=True)

    type: Mapped[str] = mapped_column(String(100))

    trans_id: Mapped[int] = mapped_column(ForeignKey("transitions.id"))
    trans_ts: Mapped[timestamp]

    # Relationships
    transition: Mapped["SQLTransition"] = relationship(back_populates="transition_info")
    step: Mapped["Step"] = relationship(back_populates="transition_info")

    __mapper_args__ = {
        "polymorphic_identity": "transition_info",
        "polymorphic_on": "type",
    }

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "step_num"], ["steps.run_id", "steps.step_num"]
        ),
    )


class NetworkWeights(Base):
    __tablename__ = "network_weights"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]

    run_id: Mapped[int] = mapped_column(nullable=True)
    step_num: Mapped[int] = mapped_column(nullable=True)
    type: Mapped[str] = mapped_column(String(100))

    # data
    state_dict = mapped_column(PickleType(impl=LONGBLOB))

    # Relationships
    step: Mapped["Step"] = relationship(back_populates="network_weights")

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "step_num"], ["steps.run_id", "steps.step_num"]
        ),
    )


class GradInfo(Base):
    __tablename__ = "grad_info"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]

    run_id: Mapped[int] = mapped_column(nullable=True)
    step_num: Mapped[int] = mapped_column(nullable=True)
    type: Mapped[str] = mapped_column(String(100))

    # data
    data = mapped_column(PickleType(impl=LONGBLOB))

    # Relationships
    step: Mapped["Step"] = relationship(back_populates="grad_info")

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "step_num"], ["steps.run_id", "steps.step_num"]
        ),
    )


stepper = None


class SQLStepper:
    def __init__(self, run: Run):
        super().__init__()
        self.run = run
        self.step_num = 0
        self.step = Step(step_num=self.step_num, run=self.run)

    def increment_step(self):
        self.step_num += 1
        self.step = Step(step_num=self.step_num, run=self.run)


def init_stepper(run: Run):
    global stepper
    stepper = SQLStepper(run)
