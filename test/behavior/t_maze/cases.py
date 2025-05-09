from test.behavior.bsuite import BSuiteTestCase


class ShortTMazeTest(BSuiteTestCase):
    name = 'short_t_maze'
    config = 'test/behavior/t_maze/short.yaml'
    lower_bounds = { 'reward': -0.1}

class MediumTMazeTest(BSuiteTestCase):
    name = 'short_t_maze'
    config = 'test/behavior/t_maze/medium.yaml'
    lower_bounds = { 'reward': -0.1}

class LongTMazeTest(BSuiteTestCase):
    name = 'short_t_maze'
    config = 'test/behavior/t_maze/long.yaml'
    lower_bounds = { 'reward': -0.1}
