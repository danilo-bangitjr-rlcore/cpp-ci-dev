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
import numpy as np
from corerl.component.buffer.buffers import SQLBuffer

import logging

logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
from corerl.sql_logging.alt_base_schema import SQLTransition, Base, NetworkWeights

logger = logging.getLogger(__name__)

def init_critic():
    with initialize(version_base=None, config_path="../config/agent/critic/critic_network"):
        critic_cfg = compose(config_name="ensemble", overrides=["base.arch=[256,256]", "ensemble=1"])

    critic = init_critic_network(cfg=critic_cfg, input_dim=21, output_dim=1)
        
    return critic

class TestORMSchema(unittest.TestCase):

    def setUp(self) -> None:
        pass

    @classmethod
    def setUpClass(cls):
        cls.remove_db = False
        cls.con_cfg = OmegaConf.load(
            "config/db/sql/credentials_vpn.yaml"
        )  # add your own credentials at this path
        cls.engine = sql_logging.get_sql_engine(cls.con_cfg, db_name="orm_test_db")
        if database_exists(cls.engine.url):
            drop_database(cls.engine.url)

        create_database(cls.engine.url)
        Base.metadata.create_all(cls.engine)

    def test_transition(self):

        with Session(self.engine) as session:
            for i in range(5):
                new_transition = SQLTransition(
                    state=list(np.random.random(size=(2,))),
                    action=np.random.random(),
                    next_state=list(np.random.random(size=(2,))),
                    reward=np.random.random(),
                )

                session.add(new_transition)

            session.commit()

    def test_critic_weight_logging(self):
        critic_before_db = init_critic()
        critic_weights = NetworkWeights(state_dict=critic_before_db.state_dict(), type="critic")
        with Session(self.engine) as session:
            session.add(critic_weights)
            session.commit()

        # fetch weights with sql query
        df = pd.read_sql(select(NetworkWeights).order_by(NetworkWeights.ts.desc()).limit(1), con=self.engine)
    
        loaded_stated_dict = df["state_dict"].iloc[0]

        critic_after_db = init_critic()
        critic_after_db.load_state_dict(loaded_stated_dict)
        
        test_input = torch.rand((1,21))
        self.assertTrue(torch.isclose(critic_after_db(test_input)[0], critic_before_db(test_input)[0]))
    
    @classmethod
    def tearDownClass(cls):
        if cls.remove_db:
            if database_exists(cls.engine.url):
                drop_database(cls.engine.url)


class TestSQLBuffer(unittest.TestCase):

    def setUp(self) -> None:
        # pass
        rows = self.session.scalars(select(SQLTransition))
        for row in rows:
            self.session.delete(row)

        self.add_rand_transitions(n=5)
        self.session.commit()

    @classmethod
    def setUpClass(cls):
        cls.remove_db = False
        cls.con_cfg = OmegaConf.load("config/db/sql/credentials_vpn.yaml")
        cls.engine = sql_logging.get_sql_engine(cls.con_cfg, db_name="orm_test_db2")

        cls.session = Session(cls.engine)

        if database_exists(cls.engine.url):
            drop_database(cls.engine.url)

        create_database(cls.engine.url)
        Base.metadata.create_all(cls.engine)

    def test_update_data(self):

        buffer_cfg = OmegaConf.load("config/agent/buffer/sql_buffer.yaml")
        buffer = SQLBuffer(buffer_cfg)
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 5)
        print(batch)

    def test_remove_data(self):
        base_idx = self.get_base_idx()
        buffer_cfg = OmegaConf.load("config/agent/buffer/sql_buffer.yaml")
        buffer = SQLBuffer(buffer_cfg)
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 5)

        exclusion_id = base_idx + 3
        # exclude transition
        excluded_transitions = self.session.execute(
            select(SQLTransition).where(SQLTransition.id == exclusion_id)
        )

        for (transition,) in excluded_transitions:
            transition.exclude = True
            self.session.add(transition)

        self.session.commit()

        # make sure tha buffer only gives 4 transitions
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 4)
        self.assertFalse(exclusion_id in buffer.transition_ids)

    def test_remove_add_remove_add(self):

        base_idx = self.get_base_idx()
        buffer_cfg = OmegaConf.load("config/agent/buffer/sql_buffer.yaml")
        buffer = SQLBuffer(buffer_cfg)
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 5)
        exclusion_ids = [base_idx + 4, base_idx + 2]
        # exclude transition
        self.remove_ids(exclusion_ids)

        # make sure tha buffer only gives 4 transitions
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 3)
        for exclusion_id in exclusion_ids:
            self.assertFalse(exclusion_id in buffer.transition_ids)

        # add 5 more transitions
        self.add_rand_transitions(n=5)

        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 8)

        exclusion_ids.extend([base_idx + 1, base_idx + 7, base_idx + 3])
        # remove more
        self.remove_ids(exclusion_ids)
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 5)
        for exclusion_id in exclusion_ids:
            self.assertFalse(exclusion_id in buffer.transition_ids)

        # add 2 more
        self.add_rand_transitions(n=2)

        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 7)
        for exclusion_id in exclusion_ids:
            self.assertFalse(exclusion_id in buffer.transition_ids)

        # sql table still has all the transitions
        num_rows = self.session.query(SQLTransition).count()
        self.assertTrue(num_rows == 12)  # started with 5, added 5 and 2
        # TODO: also check the data (e.g., state) in the batch matches original transitions
        # Kerrick checked this manually but not programatically

    def test_full_buffer(self):
        base_idx = self.get_base_idx()
        buffer_cfg = OmegaConf.load("config/agent/buffer/sql_buffer.yaml")
        buffer = SQLBuffer(buffer_cfg)
        memory = 10
        buffer.memory = memory
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 5)
        exclusion_ids = [base_idx + 4, base_idx + 2]
        # exclude transition
        self.remove_ids(exclusion_ids)

        # make sure tha buffer only gives 3 transitions
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 3)
        for exclusion_id in exclusion_ids:
            self.assertFalse(exclusion_id in buffer.transition_ids)

        # add 10 more transitions
        self.add_rand_transitions(n=10)

        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == memory)

        exclusion_ids.extend([base_idx + 1, base_idx + 7, base_idx + 3])
        # remove more
        self.remove_ids(exclusion_ids)
        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(
            len(batch.state) == memory
        )  # there were 15 in table, excluded 2 and then 3
        for exclusion_id in exclusion_ids:
            self.assertFalse(exclusion_id in buffer.transition_ids)

        # add 10 more
        self.add_rand_transitions(n=10)

        buffer.update_data()
        batch = buffer.sample_batch()
        self.assertTrue(len(batch.state) == 10)
        for exclusion_id in exclusion_ids:
            self.assertFalse(exclusion_id in buffer.transition_ids)

        # sql table still has all the transitions
        num_rows = self.session.query(SQLTransition).count()
        self.assertTrue(num_rows == 25)  # started with 5, added 10 and 10
        # TODO: also check the data (e.g., state) in the batch matches original transitions
        # Kerrick checked this manually but not programatically

    def remove_ids(self, exclusion_ids):
        excluded_transitions = self.session.execute(
            select(SQLTransition).filter(SQLTransition.id.in_(exclusion_ids)),
        )

        for (transition,) in excluded_transitions:
            transition.exclude = True
            self.session.add(transition)

        self.session.commit()

    def add_rand_transitions(self, n):
        for i in range(n):
            new_transition = SQLTransition(
                state=list(np.random.random(size=(2,))),
                action=np.random.random(),
                next_state=list(np.random.random(size=(2,))),
                reward=np.random.random(),
            )

            self.session.add(new_transition)

        self.session.commit()

    def get_base_idx(self):
        idx = self.session.scalar(
            select(SQLTransition.id).order_by(SQLTransition.id.asc())
        )
        return idx

    @classmethod
    def tearDownClass(cls):
        if cls.remove_db:
            if database_exists(cls.engine.url):
                drop_database(cls.engine.url)


if __name__ == "__main__":
    unittest.main()
