from test.behavior.bsuite import BSuiteTestCase


class ShortTMazeTest(BSuiteTestCase):
    name = 'short_t_maze'
    config = 'test/behavior/t_maze/short.yaml'
    # goes right with actions at least 0.95 until end zone then takes correct action with probability 0.95
    lower_bounds = {'reward': -0.005*2 - 0.05}

class MediumTMazeTest(BSuiteTestCase):
    name = 'medium_t_maze'
    config = 'test/behavior/t_maze/medium.yaml'
    # goes right with actions at least 0.95 until end zone then takes correct action with probability 0.95
    lower_bounds = {'reward': -0.005*5 - 0.05}

class LongTMazeTest(BSuiteTestCase):
    name = 'long_t_maze'
    config = 'test/behavior/t_maze/long.yaml'
    # goes right with actions at least 0.95 until end zone then takes correct action with probability 0.95
    lower_bounds = {'reward': -0.005*10 - 0.05}
