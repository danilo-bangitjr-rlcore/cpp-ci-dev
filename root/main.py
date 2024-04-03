import hydra

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces.utils import flatdim

from root.agent.factory import init_agent
from root.environment.factory import init_environment
from root.environment.wrapper.torch_wrapper import TorchWrapper


@hydra.main(version_base=None, config_name='config', config_path="config")
def main(cfg: DictConfig) -> None:
    seed = cfg.experiment.seed
    env = init_environment(cfg.env, seed)

    state_dim, action_dim = flatdim(env.observation_space), flatdim(
        env.action_space)  # TODO: get state and action dim from env wrapper

    agent = init_agent(cfg.agent, state_dim, action_dim, seed)

    max_steps = cfg.experiment.max_steps
    state, info = env.reset()

    for step in tqdm(range(max_steps)):
        action = agent.get_action(state)  # TODO: don't love all the extra stuff being returned
        next_state, reward, terminated, truncate, env_info = env.step(action)

        transition = (state, action, reward, next_state, terminated, truncate)
        agent.update_buffer(transition)
        agent.update()
        state = next_state


if __name__ == "__main__":
    main()
