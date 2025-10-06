from datetime import datetime
from typing import Annotated

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import JSON, DateTime

timestamp = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    ),
]


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[timestamp]
    hparams: Mapped[list["HParam"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )


class HParam(Base):
    __tablename__ = "hparams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    val = mapped_column(JSON)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"))
    run: Mapped["Run"] = relationship(back_populates="hparams")
