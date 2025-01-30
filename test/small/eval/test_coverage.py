import numpy as np
import pandas as pd
import pytest

from corerl.eval.coverage import (
    ActionSamplerConfig,
    BaseCoverageConfig,
    BaseSamplerConfig,
    CoverageProtocol,
    Dataset,
    KDECoverage,
    NeighboursCoverage,
    NeighboursCoverageConfig,
    UniformActionSampler,
    UniformDatasetSampler,
    get_norm_const,
    sample_epsilon_ball,
)


@pytest.fixture
def dataset():
    data = pd.DataFrame(
        {
            "tag-1": np.arange(0, 1, 0.1),
            "tag-2": np.array([0, 1] * 5),
        }
    )
    dataset = Dataset(
        data=data,
    )
    return dataset


@pytest.fixture
def tiny_dataset():
    data = pd.DataFrame(
        {
            "tag-1": [0, 1],
        }
    )
    dataset = Dataset(
        data=data,
    )
    return dataset


class DumbCoverage:
    def __init__(self, cfg: BaseCoverageConfig):
        self.cfg = cfg
        self.normalization = None

    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.dataset is not None
        return np.array([0.5])

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.normalization is not None
        return self.unnorm_cov(state_action) / self.normalization

    def fit(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.normalization = get_norm_const(self, dataset, self.cfg.epsilon, self.cfg.n_norm_samples)


def test_sample_epsilon_ball():
    """
    Samples vector in epsilon ball around center and makes sure they are right shape and length.
    """
    center = np.array([0.5, 0.5])
    epsilon = 0.1
    n_samples = 100
    samples = sample_epsilon_ball(center, epsilon, n_samples)
    assert samples.shape == (n_samples, len(center)), "Sample shape mismatch"
    assert np.allclose(np.linalg.norm(samples - center, axis=1), epsilon), "Samples should be within epsilon distance"


def test_get_norm_const(dataset: Dataset):
    """
    Tests that the normalization constant is greater than 0 for the DumbCoverage function.
    """
    cfg = BaseCoverageConfig(epsilon=0.1, n_norm_samples=100)
    test_coverage = DumbCoverage(cfg)
    test_coverage.fit(dataset)

    normalization = get_norm_const(test_coverage, dataset, cfg.epsilon, cfg.n_norm_samples)
    assert normalization > 0, "Normalization should be greater than 0"
    assert isinstance(normalization, float), "Normalization should be a float value"


def test_kde_coverage(dataset: Dataset):
    cfg = BaseCoverageConfig(epsilon=0.1, n_norm_samples=100)
    kde_coverage = KDECoverage(cfg)
    kde_coverage.fit(dataset)

    # Test the __call__ method for arbitrary state_action pair
    state_action = np.array([[0.5, 0.5]])
    coverage_value = kde_coverage.cov(state_action)
    assert coverage_value > 0, "Coverage value should be greater than 0"

    # Test that the normalization is greater than 0
    normalization = get_norm_const(kde_coverage, dataset, cfg.epsilon, cfg.n_norm_samples)
    assert normalization > 0, "Normalization should be greater than 0"


def _test_uniform_dataset_sampler(sampler: UniformDatasetSampler, coverage_fn: CoverageProtocol, dataset: Dataset):
    """
    Tests the UniformDataetSampler's ability to evaluate action coverage on a dataset.

    This test performs the following steps:
    * Evaluates the coverage on the provided dataset and asserts that the coverage value is between 0 and 1.
    * Creates a test dataset with low coverage and evaluates its coverage is greater than 1.
    """

    # Evaluate coverage on the dataset itself (should be very well-covered)
    coverage_value_1 = sampler.eval(dataset, coverage_fn)
    assert 0 <= coverage_value_1, "Coverage value should greater than zero."
    assert abs(coverage_value_1 - 1) <= 1.1, "Coverage value should be around 1."

    # Evaluate coverage on a test dataset which is very far from the original dataset.
    test_data = pd.DataFrame(
        {
            "tag-1": [10, 10],
        }
    )
    test_dataset = Dataset(
        data=test_data,
    )

    coverage_value_2 = sampler.eval(test_dataset, coverage_fn)
    assert coverage_value_2 >= 1, "Coverage value should be larger than 1."
    assert coverage_value_2 >= coverage_value_1


def test_kde_with_uniform_dataset_sampler(tiny_dataset: Dataset):
    np.random.seed(42)
    coverage_cfg = BaseCoverageConfig(epsilon=0.1, n_norm_samples=1000)
    sampler_cfg = BaseSamplerConfig(n_state_samples=1)

    kde_coverage = KDECoverage(coverage_cfg)
    kde_coverage.fit(tiny_dataset)
    sampler = UniformDatasetSampler(sampler_cfg)
    _test_uniform_dataset_sampler(sampler, kde_coverage, tiny_dataset)


def test_neighbours_with_uniform_dataset_sampler(tiny_dataset: Dataset):
    np.random.seed(42)
    coverage_cfg = NeighboursCoverageConfig(epsilon=0.1, n_norm_samples=1000, n_neighbours=1, metric="l2")
    sampler_cfg = BaseSamplerConfig(n_state_samples=1)

    neighbours_coverage = NeighboursCoverage(coverage_cfg)
    neighbours_coverage.fit(tiny_dataset)
    sampler = UniformDatasetSampler(sampler_cfg)
    _test_uniform_dataset_sampler(sampler, neighbours_coverage, tiny_dataset)


def _test_uniform_action_sampler(coverage_fn: CoverageProtocol, dataset: Dataset, cfg: ActionSamplerConfig):
    """
    Tests the UniformActionSampler's ability to evaluate action coverage on a dataset.

    This test performs the following steps:
    * Evaluates the coverage on the provided dataset and asserts that the coverage value is greater than 0.
    * Creates a test dataset with high action coverage and evaluates its coverage.
    * Creates a test dataset with low action coverage and evaluates its coverage.
    * Asserts that the coverage value for the high action coverage dataset is less than or equal to the coverage value for the low action coverage dataset.
    """
    sampler = UniformActionSampler(cfg)

    # Evaluate coverage on the dataset itself
    coverage_value_0 = sampler.eval(dataset, coverage_fn)
    assert coverage_value_0 > 0

    half_dataset_size = len(dataset.data) // 2
    test_tag_2 = list(np.random.uniform(0, 1, half_dataset_size))

    # make a test dataset with only the states with high action coverage
    test_data_high_cov = pd.DataFrame(
        {
            "tag-1": [0] * half_dataset_size,  # "observation" tag
            "tag-2": test_tag_2,  # "action" tag
        }
    )
    test_dataset_high_cov = Dataset(data=test_data_high_cov, action_tags=["tag-2"])

    coverage_value_1 = sampler.eval(test_dataset_high_cov, coverage_fn)

    # make a test dataset with only the states with low action coverage
    test_data_low_cov = pd.DataFrame(
        {
            "tag-1": [1] * half_dataset_size,  # "observation" tag
            "tag-2": test_tag_2,  # "action" tag
        }
    )
    test_dataset_low_cov = Dataset(data=test_data_low_cov, action_tags=["tag-2"])

    coverage_value_2 = sampler.eval(test_dataset_low_cov, coverage_fn)

    assert coverage_value_1 <= coverage_value_2


def test_kde_with_uniform_action_sampler():
    np.random.seed(0)
    coverage_cfg = BaseCoverageConfig(epsilon=0.01, n_norm_samples=1000)
    sampler_cfg = ActionSamplerConfig(n_state_samples=1, n_action_samples=100)

    half_dataset_size = 100
    data = pd.DataFrame(
        {
            "tag-1": [0] * half_dataset_size + [1] * half_dataset_size,
            "tag-2": list(np.random.uniform(0, 1, half_dataset_size)) + [0] * half_dataset_size,
        }
    )
    dataset = Dataset(data=data, action_tags=["tag-2"])

    kde_coverage = KDECoverage(coverage_cfg)
    kde_coverage.fit(dataset)
    _test_uniform_action_sampler(kde_coverage, dataset, sampler_cfg)


def test_neighbours_with_uniform_action_sampler():
    np.random.seed(42)
    coverage_cfg = NeighboursCoverageConfig(epsilon=0.1, n_norm_samples=1000, n_neighbours=1, metric="l2")
    sampler_cfg = ActionSamplerConfig(n_state_samples=1, n_action_samples=100)

    half_dataset_size = 50
    data = pd.DataFrame(
        {
            "tag-1": [0] * half_dataset_size + [1] * half_dataset_size,
            "tag-2": list(np.random.uniform(0, 1, half_dataset_size)) + [0] * half_dataset_size,
        }
    )
    dataset = Dataset(data=data, action_tags=["tag-2"])

    neighbours_coverage = NeighboursCoverage(coverage_cfg)
    neighbours_coverage.fit(dataset)
    _test_uniform_action_sampler(neighbours_coverage, dataset, sampler_cfg)
