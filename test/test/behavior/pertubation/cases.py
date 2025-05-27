import logging

from test.behavior.bsuite import BSuiteTestCase

logger = logging.getLogger(__name__)

class SaturationPerturbationTest(BSuiteTestCase):
    name = 'saturation perturbation'
    config = 'test/behavior/pertubation/saturation_pertub.yaml'
    # Define bounds similar to regular saturation test
    lower_bounds = {'reward': -0.085}


class MultiActionSaturationPerturbationTest(BSuiteTestCase):
    name = 'multi action saturation perturbation'
    config = 'test/behavior/pertubation/multi_action_saturation_pertub.yaml'


    lower_bounds = {'reward': -0.1}


class StandStillMountainCarPerturbationTest(BSuiteTestCase):
    name = 'standstill mountain car perturbation'
    config = 'test/behavior/pertubation/standstill_montain_car_pertub.yaml'

    lower_bounds = {'reward': -0.13}
