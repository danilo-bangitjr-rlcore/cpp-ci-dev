from test.behavior.bsuite import BehaviourCategory, BSuiteTestCase


class StandStillMountainCar(BSuiteTestCase):
    name = 'stand still mountain car'
    config = 'test/behavior/mountain_car/stand_still.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    lower_bounds = { 'reward': -0.13 }
    upper_bounds = { 'actor_var': 0.0001 }
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -0.57 }


class MountainCar(BSuiteTestCase):
    name = 'hill balancing mountain car'
    config = 'test/behavior/mountain_car/goal.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    # should learn to consistently bias towards slightly right-leaning actions
    lower_bounds = { 'reward': -0.43, 'ACTION-action-0': 0.38 }
    upper_warns = { 'avg_critic_loss': 0.112, 'actor_loss': -0.9 }
