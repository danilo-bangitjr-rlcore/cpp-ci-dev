from test.behavior.bsuite import BehaviourCategory, BSuiteTestCase


class AllNonstationaryTest(BSuiteTestCase):
    name = 'all_nonstationary'
    config = 'test/behavior/nonstationary_world/all_nonstationary.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.5}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }

class UpstreamSensorStepTest(BSuiteTestCase):
    name = 'upstream_sensor_step'
    config = 'test/behavior/nonstationary_world/upstream_sensor_step.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.5}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }

class DownstreamSensorStepTest(BSuiteTestCase):
    name = 'upstream_sensor_step'
    config = 'test/behavior/nonstationary_world/downstream_sensor_step.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.5}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }
