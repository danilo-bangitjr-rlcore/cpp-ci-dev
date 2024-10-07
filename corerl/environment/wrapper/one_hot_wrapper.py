import numpy as np
import gymnasium


class OneHotWrapper:
    """
    Translates actions as one-hot encodings to integers.
    """
    def __init__(self, env: gymnasium.Env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self) -> tuple[np.ndarray, dict]:
        return self.env.reset()

    def step(self, action: np.ndarray):
        action = np.argmax(action)
        action = np.array([action])
        return self.env.step(action)

    def render(self):
        return self.env.render()
