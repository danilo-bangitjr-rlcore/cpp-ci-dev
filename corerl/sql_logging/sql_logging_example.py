from corerl.sql_logging import sql_logging
from omegaconf import OmegaConf
from sqlalchemy_utils import database_exists, drop_database, create_database
from sqlalchemy import select
from corerl.component.network.factory import init_critic_network
from sqlalchemy.orm import Session
from hydra import compose, initialize
from sqlalchemy.orm import mapped_column, Mapped
from corerl.component.network.factory import init_critic_network

from sqlalchemy.orm import mapped_column
import numpy as np
import torch
import pandas as pd

# NOTE: import of base schema objects
from corerl.sql_logging.base_schema import (
    SQLTransition,
    TransitionInfo,
    Base,
    Run,
    HParam,
    NetworkWeights,
)

from corerl.component.buffer.buffers import SQLBuffer

"""
Definition of custom subclasses to store env-specific observation data.
See section "OPTIONAL: storing additional transition info" in main()
"""


class RawCoagTransitionInfo(TransitionInfo):

    # add the addition info you want to store here
    prev_uvt: Mapped[float] = mapped_column(nullable=True)
    new_uvt: Mapped[float] = mapped_column(nullable=True)
    target_uvt: Mapped[float] = mapped_column(nullable=True)
    dose: Mapped[float] = mapped_column(nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "raw",  # NOTE: this will appear in the "type" column
    }


class CoagTransitionInfo(RawCoagTransitionInfo):
    __mapper_args__ = {
        "polymorphic_identity": "normalized",  # NOTE: here we just give a name to distinguish
    }


def init_critic():
    with initialize(
        version_base=None, config_path="../../config/agent/critic/critic_network"
    ):
        critic_cfg = compose(
            config_name="ensemble", overrides=["base.arch=[256,256]", "ensemble=1"]
        )

    critic = init_critic_network(cfg=critic_cfg, input_dim=21, output_dim=1)

    return critic


def main():
    """
    Basics: connecting to db and storing transitions
    """
    # Transitions are considered a fundamental object in this schema.
    # This means you can write simple transitions to an sql database without
    # being forced to add any additional logging

    # first read the sql credentials and create the test database
    test_db_name = "orm_test_db"
    con_cfg = OmegaConf.load("config/db/sql/credentials_vpn.yaml")
    engine = sql_logging.get_sql_engine(con_cfg, db_name=test_db_name)

    # here we remove the test_db if it already exists
    # to ensure repeatability
    if database_exists(engine.url):
        drop_database(engine.url)

    create_database(engine.url)  # creates database
    Base.metadata.create_all(
        engine
    )  # creates empty tables corresponding to the base schema

    # the session object enables writing/reading from the database
    session = Session(engine)

    # now you can add transitions!
    new_transition = SQLTransition(
        state=list(np.random.random(size=(2,))),  # NOTE: arrays converted to list here
        action=np.random.random(),
        next_state=list(np.random.random(size=(2,))),
        reward=np.random.random(),
    )
    # Arrays are converted to lists in the table for readability in SQL
    # You can directly give numpy arrays (it's up to you!), but then your tables won't be human readable.

    # write the transition to the db
    session.add(new_transition)
    session.commit()  # NOTE: you have to commit after adding!

    # We could add two transitions and then commit both:
    for _ in range(2):
        new_transition = SQLTransition(
            state=list(
                np.random.random(size=(2,))
            ),  # NOTE: arrays converted to list here
            action=np.random.random(),
            next_state=list(np.random.random(size=(2,))),
            reward=np.random.random(),
        )
        session.add(new_transition)  # doesn't write yet

    session.commit()  # writes two rows to the db

    """
    OPTIONAL: experiment tracking
    """

    # first write to the run table to keep track of this experiment
    # while we're at it, let's track the hyperparameters
    my_hparams = {
        "step_size": 0.001,
        "arch": [256, 256],  # architecture
        "agent_name": "Billy",
    }  # this dict might come from a config in practice

    run = Run(hparams=[HParam(name=name, val=val) for name, val in my_hparams.items()])
    session.add(run)
    session.commit()  # since hparams belong to the run, we only needed to add run

    """
    OPTIONAL: storing additional transition info (such as raw observations)
    NOTE: depends on experiment tracking at the moment (connected to run table)
    """
    # You can attach info to your transitions
    # the info will appear in another table "transition_info"
    # you could for example include raw observations

    # let's say we are doing coag dosing at Drayton Valley
    # we want to store uvt before and after the transition,
    # along with the coag dosing rate and target uvt

    # we defined subclasses of TransitionInfo at the top of this file
    # NOTE: you will need to make sure these class definitions are in scope
    # before you call Base.metadata.create_all() (which creates the tables).

    # create instances of the transition info classes
    # raw observations
    raw_tinfo = RawCoagTransitionInfo(
        step=0,  # interaction step
        prev_uvt=89,
        new_uvt=91,
        target_uvt=92,
        dose=10,
        run=run,  # NOTE: this connects the transitioninfo to the run table
    )

    # normalized observations
    norm_tinfo = CoagTransitionInfo(
        step=0,
        prev_uvt=raw_tinfo.prev_uvt / 100,
        new_uvt=raw_tinfo.new_uvt / 100,
        target_uvt=raw_tinfo.target_uvt / 100,
        dose=raw_tinfo.dose / 15,
        run=run,  # NOTE: this connects the transitioninfo to the run table
    )

    # create a new transition
    # let's make sure we use the norm_tinfo data for consistency
    # (but you could add anything to state, action, etc.)
    new_transition = SQLTransition(
        state=[norm_tinfo.prev_uvt, norm_tinfo.target_uvt],  #
        action=norm_tinfo.dose,
        next_state=[norm_tinfo.new_uvt, norm_tinfo.target_uvt],
        reward=abs(norm_tinfo.target_uvt - norm_tinfo.new_uvt),
    )

    # attach the raw and normalized observations to this transition
    new_transition.transition_info.extend(
        [raw_tinfo, norm_tinfo]
    )  # we are just extending a list here
    session.add(new_transition)  # doesn't write yet
    session.commit()  # this writes the transition and the transition info!

    """
    OPTIONAL: The SQLBuffer
    
    There is an SQLBuffer that will automatically sync to the transition table.
    The primary use case here is if you want to mark transitions to be ignored online.
    It can also be useful if you want to relabel (e.g., renormalize) or alter transitions in the buffer online.

    WARNING: You will need to modify your code to avoid calling buffer.feed().
    Instead call buffer.update_data() with no args
    """
    buffer_cfg = OmegaConf.load("config/agent/buffer/sql_buffer.yaml")
    buffer_cfg["db_name"] = test_db_name  # NOTE: adding this outside of yaml file
    buffer = SQLBuffer(buffer_cfg)
    buffer.update_data()  # we already have a few transitions in the db

    # get a batch
    batch = buffer.sample_batch()
    print(batch.state)

    # mark a transition to be ignored
    # NOTE: we can do this from a different process (or manually),
    # which could be handy online
    exclusion_ids = [2]
    remove_ids(exclusion_ids, session)

    # now update the buffer to remove the excluded transition,
    # and confirm it doesn't exist in a new batch
    buffer.update_data()
    batch = buffer.sample_batch()
    print(batch.state)

    """
    OPTIONAL: Network weight logging
    """
    critic_before_db = init_critic()  # initialize random critic

    # create row in network_weights table
    critic_weights = NetworkWeights(
        state_dict=critic_before_db.state_dict(), type="critic"
    )

    # save weights to db
    session.add(critic_weights)
    session.commit()

    # fetch weights with sql query
    df = pd.read_sql(
        select(NetworkWeights).order_by(NetworkWeights.ts.desc()).limit(1), con=engine
    )

    loaded_stated_dict = df["state_dict"].iloc[0]

    # load weights and confirm we get the same output
    critic_after_db = init_critic()
    critic_after_db.load_state_dict(loaded_stated_dict)

    test_input = torch.rand((1, 21))
    assert torch.isclose(
        critic_after_db(test_input)[0], critic_before_db(test_input)[0]
    )


def remove_ids(exclusion_ids, session):
    excluded_transitions = session.execute(
        select(SQLTransition).filter(SQLTransition.id.in_(exclusion_ids)),
    )

    for (transition,) in excluded_transitions:
        transition.exclude = True
        session.add(transition)

    session.commit()


if __name__ == "__main__":
    main()
