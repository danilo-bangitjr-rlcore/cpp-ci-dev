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

import logging
logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
from assets.coag_sim_schema import Base, SQLTransition, RawTransition, CriticWeights

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

def get_transition():
    state = torch.tensor([1, 2], dtype=torch.float)
    action = torch.tensor(0.5, dtype=torch.float)
    reward = torch.tensor(-1.0, dtype=torch.float)
    state_next = torch.tensor([1, 2], dtype=torch.float)

    transition = Transition(state, action, reward, state_next, False, False)
    return transition

def init_critic():
    with initialize(version_base=None, config_path="../config/agent/critic/critic_network"):
        critic_cfg = compose(config_name="ensemble", overrides=["base.arch=[16,16]", "ensemble=1"])

    critic = init_critic_network(cfg=critic_cfg, input_dim=21, output_dim=1)
        
    return critic

    
class TestORMSchema(unittest.TestCase):

    def setUp(self) -> None:
        pass

    @classmethod
    def setUpClass(cls):
        cls.remove_db = False
        cls.con_cfg = OmegaConf.load('config/db/sql/credentials_vpn.yaml') # add your own credentials at this path
        cls.engine = sql_logging.get_sql_engine(cls.con_cfg, db_name='orm_test_db')
        if database_exists(cls.engine.url):
            drop_database(cls.engine.url)
        
        create_database(cls.engine.url)
        Base.metadata.create_all(cls.engine)
    
    def test_critic_weight_logging(self):
        critic_before_db = init_critic()

        critic_weights = CriticWeights(critic_weights=critic_before_db.state_dict())
        with Session(self.engine) as session:
            session.add(critic_weights)
            session.commit()

        # fetch weights with sql query
        df = pd.read_sql(select(CriticWeights).order_by(CriticWeights.ts.desc()).limit(1), con=self.engine)
    
        loaded_stated_dict = df["critic_weights"].iloc[0]

        critic_after_db = init_critic()
        critic_after_db.load_state_dict(loaded_stated_dict)
        
        test_input = torch.rand((1,21))
        self.assertTrue(torch.isclose(critic_after_db(test_input)[0], critic_before_db(test_input)[0]))

    def test_raw_transition_info(self):
        
        raw_transition = RawTransition(
            run=0, 
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
        
        with Session(self.engine) as session:
            session.add(raw_transition)
            session.commit()

            
    @classmethod
    def tearDownClass(cls):
        if cls.remove_db:
            if database_exists(cls.engine.url):
                drop_database(cls.engine.url)

if __name__ == '__main__':
    unittest.main()