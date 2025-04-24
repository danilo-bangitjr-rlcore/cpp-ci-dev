from test.behavior.bsuite import BSuiteTestCase


class SaturationTest(BSuiteTestCase):
    name = 'saturation'
    config = 'test/behavior/saturation/config.yaml'

    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class GoalSaturationTest(BSuiteTestCase):
    name = 'saturation goal'
    config = 'test/behavior/saturation/saturation_goals.yaml'

    lower_bounds = { 'reward': -0.5}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class DelayedSaturationTest(BSuiteTestCase):
    name = 'delayed saturation'
    config = 'test/behavior/saturation/direct_delayed.yaml'

    lower_bounds = { 'reward': -0.19 }
    upper_warns = { 'avg_critic_loss': 0.002, 'actor_loss': -1.55 }

class MultiActionSaturationTest(BSuiteTestCase):
    name = 'multi action saturation'
    config = 'test/behavior/saturation/multi_action_config.yaml'

    lower_bounds = { 'reward': -0.1}

class ExpandingBoundsSaturationTest(BSuiteTestCase):
    name = 'expanding bounds saturation'
    config = 'test/behavior/saturation/expanding_bounds.yaml'
    required_features = {'action_bounds'}
    lower_bounds = { 'reward': -0.085}

class SlowExpandingBoundsSaturationTest(BSuiteTestCase):
    name = 'slow expanding bounds saturation'
    config = 'test/behavior/saturation/slow_expanding_bounds.yaml'
    required_features = {'action_bounds'}
    lower_bounds = { 'reward': -0.085}

class SetpointChangeSaturationTest(BSuiteTestCase):
    name = 'setpoint change saturation'
    config = 'test/behavior/saturation/setpoint_change.yaml'
    required_features = {'action_bounds'}
    lower_bounds = { 'reward': -0.085}
