import pytest
from sqlalchemy import Engine

from test.behavior.bsuite import BSuiteTestCase


class StandStillMountainCar(BSuiteTestCase):
    name = 'stand still mountain car'
    config = 'test/behavior/stand_still_mountain_car/config.yaml'

    lower_bounds = { 'reward': -0.05 }
    upper_warns = { 'critic_loss': 0.003, 'actor_loss': -0.5 }


@pytest.mark.parametrize('test_case', [
    StandStillMountainCar(),
])
@pytest.mark.timeout(600)
def test_stand_still_mountain_car(
    test_case: BSuiteTestCase,
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
):
    port = tsdb_engine.url.port
    assert port is not None
    test_case.execute_test(tsdb_engine, port, tsdb_tmp_db_name)
