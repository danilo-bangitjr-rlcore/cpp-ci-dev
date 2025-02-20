from test.behavior.bsuite import BSuiteTestCase


class StandStillMountainCar(BSuiteTestCase):
    name = 'stand still mountain car'
    config = 'test/behavior/stand_still_mountain_car/config.yaml'

    lower_bounds = { 'reward': -0.21 }
    upper_warns = { 'avg_critic_loss': 0.006, 'actor_loss': 2.22 }
