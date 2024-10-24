import torch
import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from corerl.sql_logging.examples.example_env import CoagBanditSimEnv
from corerl.sql_logging.base_schema import (
    NetworkWeights,
    Loss,
    GradInfo,
)
from corerl.sql_logging.sql_logging import setup_sql_logging

from corerl.agent.factory import init_agent
import corerl.sql_logging.base_schema as sql

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@hydra.main(
    version_base="1.3",
    config_path="../../../config/",
    config_name="bandit_config",
)
def main(cfg: DictConfig) -> None:
    run_agent(cfg)


def run_agent(cfg: DictConfig) -> None:
    session, run = setup_sql_logging(cfg)
    sql.init_stepper(run)

    # setup env
    env = CoagBanditSimEnv()

    # setup agent
    agent = init_agent(cfg.agent, state_dim=2, action_dim=1)
    agent.critic_buffer.session = session  # override session (change to register fn)

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
            step_num = sql.stepper.step_num

        if do_update:
            agent.update()

            # log loss (here we just use random, will rely on hooks)
            sql.stepper.step.losses.append(
                Loss(loss=float(np.random.random()), type="train")
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
                step=sql.stepper.step,
            )

            session.add(critic_weights)

            grads = get_ensemble_grad(agent)
            if grads is not None:
                grad_info = GradInfo(type="grad", data=grads, step=sql.stepper.step)
                session.add(grad_info)

            session.commit()

        sql.stepper.increment_step()


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


if __name__ == "__main__":
    main()
