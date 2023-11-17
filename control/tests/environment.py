import sys
sys.path.insert(0, "../")
import numpy as np
import unittest
import lib.environment.three_tanks as tt

SEED = 42

class TestThreeTanks(unittest.TestCase):
    def test_setpoint(self):
        env = tt.ThreeTankEnv(seed=SEED, lr_constrain=1e-6)
        env.reset()
        self.assertEqual(env.setpoint, 3, "After reset, the setpoint should go to 3")

    def test_change_action_change_pid(self):
        env = tt.TTChangeAction(seed=SEED, lr_constrain=1e-6, constant_pid=False)
        env.reset()
        self.assertEqual(env.internal_timeout, 10, "The internal timeout is 10 when the agent can change pid")
        self.assertEqual(env.internal_iterations, 100, "The internal iterations is 100 when the agent can change pid")

    def test_change_action_constant_pid(self):
        env = tt.TTChangeAction(seed=SEED, lr_constrain=1e-6, constant_pid=True)
        env.reset()
        self.assertEqual(env.internal_timeout, 1, "The internal timeout is 1 when pid is constant")
        self.assertEqual(env.internal_iterations, 1000, "The internal iterations is 1000 when pid is constant")

    def test_direct_action_change_pid(self):
        env = tt.TTAction(seed=SEED, lr_constrain=1e-6, constant_pid=False)
        env.reset()
        self.assertEqual(env.internal_timeout, 10, "The internal timeout is 10 when the agent can change pid")
        self.assertEqual(env.internal_iterations, 100, "The internal iterations is 100 when the agent can change pid")

    def test_direct_action_constant_pid(self):
        env = tt.TTAction(seed=SEED, lr_constrain=1e-6, constant_pid=True)
        env.reset()
        self.assertEqual(env.internal_timeout, 1, "The internal timeout is 1 when pid is constant")
        self.assertEqual(env.internal_iterations, 1000, "The internal iterations is 1000 when pid is constant")

    def test_discrete_delta_pid(self):
        env = tt.TTChangeActionDiscrete(0.1, seed=SEED, lr_constrain=1e-6, constant_pid=True)
        env.reset()
        self.assertEqual(env.internal_timeout, 1, "The internal timeout is 1 when pid is constant")
        self.assertEqual(env.internal_iterations, 1000, "The internal iterations is 1000 when pid is constant")
        sp, r, done, _, _ = env.step(0)
        self.assertEqual(sp[0], -0.1, "action 0 reduces PID")
        self.assertEqual(sp[1], -0.1, "action 0 reduces PID")

        sp, r, done, _, _ = env.step(1)
        self.assertEqual(sp[0], -0.1, "action 1 does not change PID")
        self.assertEqual(sp[1], -0.1, "action 1 does not change PID")

        sp, r, done, _, _ = env.step(2)
        self.assertEqual(sp[0], 0, "action 1 does not change PID")
        self.assertEqual(sp[1], 0, "action 1 does not change PID")

if __name__ == '__main__':
    unittest.main()