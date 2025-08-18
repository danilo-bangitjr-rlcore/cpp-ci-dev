
import numpy as np


class PIController:
    def __init__(self, K: float, Ti: float, Tt: float, mv_lim: list[float]):
        self.K = K
        self.Ti = Ti
        self.Tt = Tt
        self.mv_lim = mv_lim
        self.int_error = 0.
        self.int_windup = 0.
        self.max_int = 1000*np.max(np.abs(self.mv_lim))
        self.nom_out = np.mean(self.mv_lim)

    def step(self, error: float, dt: float) -> float:
        self.int_error += (error / self.Ti + self.int_windup / self.Tt) *dt
        self.int_error = np.clip(self.int_error, -self.max_int, self.max_int)

        raw_output = self.nom_out + self.K * (error + self.int_error)
        clipped_output = np.clip(raw_output, self.mv_lim[0]+1e-6, self.mv_lim[1]-1e-6)
        self.int_windup = (clipped_output - raw_output)

        return clipped_output

    def reset(self):
        self.int_error = 0
        self.int_windup = 0
