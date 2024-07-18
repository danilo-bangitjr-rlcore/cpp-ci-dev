import torch
import numpy as np
import logging
import numpy as np
from collections.abc import MutableMapping

import hydra
from omegaconf import DictConfig, OmegaConf
from corerl.sql_logging.examples.example_env import CoagBanditSimEnv

from corerl.sql_logging.base_schema import (
    Base,
    Run,
    HParam,
    NetworkWeights,
    Loss,
    GradInfo
)
from corerl.sql_logging import sql_logging

from sqlalchemy.orm import Session
from sqlalchemy_utils import database_exists, drop_database, create_database
from sqlalchemy import select
import corerl.utils.freezer as fr
from pathlib import Path
from corerl.agent.factory import init_agent

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@hydra.main(
    version_base="1.3", config_path="../../../config/", config_name="bandit_config",
)
def main(cfg: DictConfig) -> None:

    logger.info(
        f"Output dir: {(out := hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)}"
    )

    session, run = setup_sql_logging(cfg)
    run_agent(cfg, run, session)

def run_agent(cfg: DictConfig, run, session) -> None:

    # setup env
    env = CoagBanditSimEnv()
    env.register_run(run)

    # setup agent
    agent = init_agent(cfg.agent, state_dim=2, action_dim=1)
    agent.critic_buffer.session = session  # override session (chane to register fn)

    # get initial state and action
    state, info = env.reset()
    initial_action = agent.get_action(state)

    env.start_step(initial_action)

    do_update = True
    step_num = -1

    # run this agent forever
    while True:
        logger.info(f"step: {step_num+1}")

        # here we could wait for some condition based on the env
        if env.ready_for_action():

            # get transition
            transition = env.finish_step()
            state = np.array(transition.next_state)

            # take an action as soon as possible (could instead wait until after update depending on use case)
            action = agent.get_action(state)
            env.start_step(action)

            agent.update_buffer(transition)
            step_num = transition.step.step_num

        if do_update:
            agent.update()
            
            # log loss (here we just use random, will rely on hooks)
            transition.step.losses.append(
                Loss(loss=float(np.random.random()), step=transition.step, type="train")
            )


        # here the logging of critic weights and grads is included for completeness
        # however, these would best be handled inside the critic itself (e.g., with hooks)
        if step_num % 2 == 0:
            log_critic_weights = True
        else:
            log_critic_weights = False

        if log_critic_weights:
            
            critic_weights = NetworkWeights(
                state_dict=agent.q_critic.model.state_dict(),
                type="critic",
                step=transition.step,
            )
            
            session.add(critic_weights)
        
            grads = get_ensemble_grad(agent)
            if grads is not None:
                grad_info = GradInfo(type='grad', data=grads, step=transition.step)
                session.add(grad_info)

            session.commit()


# utils
def setup_sql_logging(cfg):
    """
    This could live in a util file
    """
    
    con_cfg = cfg.agent.buffer.con_cfg
    flattened_cfg = prep_cfg_for_db(OmegaConf.to_container(cfg), to_remove=[])
    engine = sql_logging.get_sql_engine(con_cfg, db_name=cfg.agent.buffer.db_name)

    if database_exists(engine.url):
        drop_database(engine.url)

    create_database(engine.url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        run = Run(
            hparams=[HParam(name=name, val=val) for name, val in flattened_cfg.items()]
        )
        session.add(run)
        session.commit()

        run_id = session.scalar(select(Run.run_id).order_by(Run.run_id.desc()))
        logger.info(f"{run_id=}")

        return session, run

def get_ensemble_grad(agent):
    batch = agent.critic_buffer.sample_batch()
    if batch is not None:
        loss = agent.compute_q_loss(batch)
        agent.q_critic.ensemble_backward(loss)

        grads = []
        for p in agent.q_critic.model.parameters():
            grads.append(p.grad.flatten())
        grads = torch.cat(grads)
    else:
        grads = None

    return grads

def flatten_dict(dictionary: dict, parent_key: str='', separator: str='_') -> dict:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def prep_cfg_for_db(cfg: dict, to_remove: list[str]) -> dict:
    for key in to_remove:
        del cfg[key]
    return flatten_dict(cfg)

if __name__ == "__main__":
    main()