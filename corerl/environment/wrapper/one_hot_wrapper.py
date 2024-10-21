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
        a = np.argmax(action)
        a = np.array([a])
        return self.env.step(a)

    def render(self):
        return self.env.render()
