from test.behavior.bsuite import BSuiteTestCase


class SaturationTest(BSuiteTestCase):
    name = 'saturation'
    config = 'test/behavior/saturation/config.yaml'

    lower_bounds = { 'reward': -0.078 }
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class DeltaSaturationTest(SaturationTest):
    name = 'delta saturation'
    config = 'test/behavior/saturation/delta.yaml'

    lower_bounds = { 'reward': -0.19 }
    upper_warns = { 'avg_critic_loss': 0.005, 'actor_loss': -1.0 }


class DelayedSaturationTest(BSuiteTestCase):
    name = 'delayed saturation'
    config = 'test/behavior/saturation/direct_delayed.yaml'

    lower_bounds = { 'reward': -0.19 }
    upper_warns = { 'avg_critic_loss': 0.002, 'actor_loss': -1.55 }
