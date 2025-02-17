from test.behavior.bsuite import BSuiteTestCase


class SaturationTest(BSuiteTestCase):
    name = 'saturation'
    config = 'test/behavior/saturation/config.yaml'

    lower_bounds = { 'reward': -0.25 }
    upper_warns = { 'critic_loss': 0.05, 'actor_loss': -1.0 }


class DeltaSaturationTest(SaturationTest):
    name = 'delta saturation'
    config = 'test/behavior/saturation/delta.yaml'
