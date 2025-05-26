

from test.behavior.bsuite import BSuiteTestCase


class CalibrationTest(BSuiteTestCase):
    name = 'calibration'
    config = 'test/behavior/calibration/calibration.yaml'

    # The best possible reward after calibration is half of -0.571
    lower_bounds = {'reward': -0.3}
