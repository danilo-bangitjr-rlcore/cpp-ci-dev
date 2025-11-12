from test.behavior.bsuite import BehaviourCategory, BSuiteTestCase


class DistractionWorldTest(BSuiteTestCase):
    name = 'distraction_world'
    config = 'test/behavior/distraction_world/distraction_world.yaml'

    lower_bounds = { 'reward': -0.1 }
    category = {BehaviourCategory.REPRESENTATION}
