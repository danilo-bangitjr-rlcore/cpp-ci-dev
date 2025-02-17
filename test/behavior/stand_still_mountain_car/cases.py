from test.behavior.bsuite import BSuiteTestCase


class StandStillMountainCar(BSuiteTestCase):
    name = 'stand still mountain car'
    config = 'test/behavior/stand_still_mountain_car/config.yaml'

    lower_bounds = { 'reward': -0.05 }
    upper_warns = { 'critic_loss': 0.003, 'actor_loss': -0.5 }
