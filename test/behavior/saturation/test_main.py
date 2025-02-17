import pytest
from sqlalchemy import Engine

from test.behavior.bsuite import BSuiteTestCase


class SaturationTest(BSuiteTestCase):
    name = 'saturation'
    config = 'test/behavior/saturation/config.yaml'

    lower_bounds = { 'reward': -0.25 }
    upper_warns = { 'critic_loss': 0.05, 'actor_loss': -1.0 }


class DeltaSaturationTest(SaturationTest):
    name = 'delta saturation'
    config = 'test/behavior/saturation/delta.yaml'


@pytest.mark.parametrize('test_case', [
    SaturationTest(),
    DeltaSaturationTest(),
])
@pytest.mark.timeout(900)
def test_saturation(
    test_case: BSuiteTestCase,
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
):
    port = tsdb_engine.url.port
    assert port is not None
    test_case.execute_test(tsdb_engine, port, tsdb_tmp_db_name)
