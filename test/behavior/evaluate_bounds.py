import sys
from pathlib import Path
from typing import NamedTuple

sys.path.append(str(Path.cwd()))

from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from rlevaluation.backend.statistics import PercentileBootstrapResult, ToleranceIntervalResult
from rlevaluation.statistics import percentile_bootstrap_ci, tolerance_interval

from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine
from test.behavior.bsuite import BSuiteTestCase
from test.behavior.test_bsuite import TEST_CASES


class TestResults(NamedTuple):
    ti: ToleranceIntervalResult
    ci: PercentileBootstrapResult


def _eval_test_case(test_case: BSuiteTestCase, seed: int):
    cfg = SQLEngineConfig(
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='localhost',
        port=5432,
    )
    db_name = f'bsuite_test_{test_case.name}_{seed}'
    engine = get_sql_engine(cfg, db_name)

    all_metrics = (
        test_case.upper_bounds
        | test_case.lower_bounds
        | test_case.upper_warns
        | test_case.lower_warns
    )

    test_case._overrides |= { 'experiment.seed': seed}

    metrics_table = test_case.execute_test(engine, 5432, db_name)
    metric_values: dict[str, float] = {}
    for metric in all_metrics:
        value = test_case.summarize_over_time(metric, metrics_table)
        metric_values[metric] = value

    return metric_values


def _test_over_seeds(test_case: BSuiteTestCase, pool: Pool, seeds: int):
    results = pool.map(partial(_eval_test_case, test_case), range(seeds))

    df = pd.DataFrame(results)

    stats: dict[str, TestResults] = {}

    rng = np.random.default_rng(0)

    for metric in df.columns:
        stats[metric] = TestResults(
            ti=tolerance_interval(
                df[metric].to_numpy(),
                # confidence level
                alpha=0.1,
                # coverage percentage
                beta=0.95,
            ),
            ci=percentile_bootstrap_ci(
                rng,
                df[metric].to_numpy(),
                # confidence level
                alpha=0.05,
            ),
        )


    return stats


def main():
    pool = Pool(16)

    all_results: list[dict[str, TestResults]] = []
    for test_case in TEST_CASES:
        got = _test_over_seeds(test_case, pool, 32)
        all_results.append(got)

    with open("out.txt", 'w') as f:
        for result, test_case in zip(all_results, TEST_CASES, strict=True):
            f.write('-' * 50)
            f.write(test_case.name)
            f.write("\n")
            for metric in result:
                res = result[metric]
                f.write(f'{metric} - ti: {res.ti.tol}')
                f.write("\n")
                f.write(f'{metric} - ci: {res.ci.sample_stat} in {res.ci.ci}')
                f.write("\n")



if __name__ == "__main__":
    main()
