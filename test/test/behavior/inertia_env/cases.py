from test.behavior.bsuite import BehaviourCategory, BSuiteTestCase


class InertiaEnv1DDirectTest(BSuiteTestCase):
    name = 'inertia_env_1d_direct'
    config = 'test/behavior/inertia_env/inertia_env_1d_direct.yaml'

    # Agent converges to rewards of about -0.145 when the policy erroneously
    # becomes deterministic along the bottom of the action's operating range
    lower_bounds = {'reward': -0.1}
    category = {BehaviourCategory.EXPLORATION}


class InertiaEnv1DDeltaTest(BSuiteTestCase):
    name = 'inertia_env_1d_delta'
    config = 'test/behavior/inertia_env/inertia_env_1d_delta.yaml'

    # Agent converges to rewards of about -0.145 when the policy erroneously
    # becomes deterministic along the bottom of the action's operating range
    lower_bounds = {'reward': -0.1}
    category = {BehaviourCategory.EXPLORATION}
