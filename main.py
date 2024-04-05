import hydra

from tqdm import tqdm
from omegaconf import DictConfig
from gymnasium.spaces.utils import flatdim

from root.agent.factory import init_agent
from root.environment.factory import init_environment
from root.state_constructor.factory import init_state_constructor
from root.interaction.factory import init_interaction


@hydra.main(version_base=None, config_name='config', config_path="config")
def main(cfg: DictConfig) -> None:
    env = init_environment(cfg.env)
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)
    action_dim = flatdim(env.action_space)

    state, info = env.reset()
    state_dim = sc.get_state_dim(state)  # gets state_dim dynamically
    agent = init_agent(cfg.agent, state_dim, action_dim)

    max_steps = cfg.experiment.max_steps
    for step in tqdm(range(max_steps)):
        action = agent.get_action(state)
        next_state, reward, terminated, truncate, env_info = interaction.step(action)
        transition = (state, action, reward, next_state, terminated, truncate)
        agent.update_buffer(transition)
        agent.update()
        state = next_state


if __name__ == "__main__":
    main()
